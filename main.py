"""
NoBluff Backend - High-Stakes Lie Detection API
================================================
Production-ready FastAPI server with AI failover pattern.
Deployed on Render.com.

UPDATES:
- Added "INCONCLUSIVE" state for silence/noise.
- Enhanced System Prompt for "Forensic" accuracy.
- Enforced Native JSON response mode for Gemini.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =============================================================================
# CONFIGURATION
# =============================================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Use Gemini 1.5 Flash for speed, or Pro for max intelligence
# Flash is recommended for real-time "viral" feel.
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash")

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nobluff")

# Shared HTTP Client (Connection Pooling)
http_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=60.0) # Increased timeout for deep analysis
    logger.info("Global HTTP Client Initialized")
    yield
    if http_client:
        await http_client.aclose()
        logger.info("Global HTTP Client Closed")

# =============================================================================
# DOMAIN MODELS
# =============================================================================

class Verdict(str, Enum):
    BLUFF = "BLUFF"
    NO_BLUFF = "NO BLUFF"
    INCONCLUSIVE = "INCONCLUSIVE" # New state for silence/noise

@dataclass
class AnalysisResult:
    """Internal result from AI provider analysis."""
    verdict: Verdict
    confidence: float
    reason: str
    provider: str

class AnalysisResponse(BaseModel):
    """API response schema."""
    verdict: str = Field(..., description="BLUFF, NO BLUFF, or INCONCLUSIVE")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    reason: str = Field(..., description="Punchy explanation for UI")

class ErrorResponse(BaseModel):
    error: str
    code: str
    detail: Optional[str] = None

# =============================================================================
# AI PROVIDER ABSTRACTION
# =============================================================================

class AIProviderError(Exception):
    def __init__(self, provider: str, message: str, retriable: bool = True):
        self.provider = provider
        self.message = message
        self.retriable = retriable
        super().__init__(f"[{provider}] {message}")

class AIProvider(ABC):
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    async def analyze_audio(self, audio_data: bytes, mime_type: str) -> AnalysisResult:
        pass
    
    def _build_analysis_prompt(self) -> str:
        """
        ADVANCED PROMPT v2:
        - Detects Silence/Noise
        - Focuses on Acoustic Biomarkers (Pitch, Jitter, Latency)
        """
        return """You are a Forensic Psycho-Acoustic Analyst. Your job is to detect deception in audio.

STEP 1: AUDIO QUALITY CHECK
- If the audio is silent, just background noise, music, or the speech is unintelligible:
  RETURN verdict: "INCONCLUSIVE" and reason: "No clear speech detected."

STEP 2: DECEPTION ANALYSIS
Analyze the following biomarkers:
1. PITCH VOLATILITY: sudden spikes or cracks in voice.
2. LATENCY: unusual pauses before answering.
3. OVER-EXPLANATION: unnecessary details (a sign of fabrication).
4. TONE: Does the emotional tone match the words?

OUTPUT FORMAT (JSON ONLY):
{
    "verdict": "BLUFF" | "NO BLUFF" | "INCONCLUSIVE",
    "confidence": 0.0 to 1.0,
    "reason": "Short, punchy verdict (Max 8 words). E.g., 'High vocal stress detected.'"
}

Be skeptical. If indicators are mixed, lean towards the stronger acoustic signal.
"""

    def _parse_verdict_response(self, text: str) -> Tuple[Verdict, float, str]:
        cleaned = text.strip()
        if "```" in cleaned:
            cleaned = re.sub(r"```json|```", "", cleaned).strip()
        
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"JSON Parse Error. Raw: {text}")
            # Fallback for emergency regex
            return Verdict.INCONCLUSIVE, 0.0, "Analysis Error. Try Again."

        # Parse Verdict
        v_str = str(data.get("verdict", "")).upper().strip()
        
        if "INCONCLUSIVE" in v_str:
            return Verdict.INCONCLUSIVE, 0.0, data.get("reason", "Audio unclear.")
        elif "NO" in v_str or "TRUTH" in v_str:
            return Verdict.NO_BLUFF, float(data.get("confidence", 0.9)), data.get("reason", "No deception markers.")
        elif "BLUFF" in v_str or "LIE" in v_str:
            return Verdict.BLUFF, float(data.get("confidence", 0.9)), data.get("reason", "Deception detected.")
        
        return Verdict.INCONCLUSIVE, 0.0, "Unclear result."

# =============================================================================
# GEMINI SERVICE (Primary - Multimodal)
# =============================================================================

class GeminiService(AIProvider):
    API_BASE = "https://generativelanguage.googleapis.com/v1beta"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    @property
    def name(self) -> str: return "Gemini"
    
    async def analyze_audio(self, audio_data: bytes, mime_type: str) -> AnalysisResult:
        # Normalize MIME for Gemini
        gemini_mime = "audio/mp4" if "m4a" in mime_type.lower() else mime_type
        
        audio_b64 = base64.standard_b64encode(audio_data).decode("utf-8")
        
        payload = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": gemini_mime, "data": audio_b64}},
                    {"text": self._build_analysis_prompt()}
                ]
            }],
            "generationConfig": {
                "temperature": 0.2, # Low temp for analytical precision
                "response_mime_type": "application/json" # Enforce strict JSON
            }
        }
        
        url = f"{self.API_BASE}/{GEMINI_MODEL_NAME}:generateContent?key={self.api_key}"
        
        try:
            response = await http_client.post(url, json=payload)
            
            if response.status_code != 200:
                logger.error(f"[Gemini] Error {response.status_code}: {response.text}")
                raise Exception(f"Gemini API Error {response.status_code}")
            
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            
            verdict, confidence, reason = self._parse_verdict_response(text)
            return AnalysisResult(verdict, confidence, reason, "Gemini")
            
        except Exception as e:
            logger.error(f"[Gemini] Exception: {e}")
            raise AIProviderError("Gemini", str(e))

# =============================================================================
# OPENAI SERVICE (Backup)
# =============================================================================

class OpenAIService(AIProvider):
    API_BASE = "https://api.openai.com/v1"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    @property
    def name(self) -> str: return "OpenAI"
    
    async def analyze_audio(self, audio_data: bytes, mime_type: str) -> AnalysisResult:
        # 1. Transcribe (Whisper)
        files = {
            "file": ("audio.m4a", audio_data, mime_type),
            "model": (None, "whisper-1"),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            transcript_res = await http_client.post(f"{self.API_BASE}/audio/transcriptions", headers=headers, files=files)
            
            if transcript_res.status_code != 200:
                raise Exception("Whisper Failed")
                
            transcript = transcript_res.json().get("text", "")
            
            if len(transcript.strip()) < 2:
                # Handle silence explicitly in fallback
                return AnalysisResult(Verdict.INCONCLUSIVE, 0.0, "No speech detected.", "OpenAI")

            # 2. Analyze Text
            chat_payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": self._build_analysis_prompt()},
                    {"role": "user", "content": f"TRANSCRIPT: {transcript}"}
                ],
                "response_format": { "type": "json_object" }
            }
            json_headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            
            chat_res = await http_client.post(f"{self.API_BASE}/chat/completions", headers=json_headers, json=chat_payload)
            content = chat_res.json()["choices"][0]["message"]["content"]
            
            verdict, confidence, reason = self._parse_verdict_response(content)
            return AnalysisResult(verdict, confidence, reason, "OpenAI")
            
        except Exception as e:
            raise AIProviderError("OpenAI", str(e))

# =============================================================================
# ORCHESTRATOR
# =============================================================================

class AnalysisOrchestrator:
    def __init__(self):
        self.providers = []
        if GOOGLE_API_KEY: self.providers.append(GeminiService(GOOGLE_API_KEY))
        if OPENAI_API_KEY: self.providers.append(OpenAIService(OPENAI_API_KEY))
    
    async def analyze(self, audio_data: bytes, mime_type: str) -> AnalysisResult:
        for provider in self.providers:
            try:
                return await provider.analyze_audio(audio_data, mime_type)
            except Exception as e:
                logger.warning(f"{provider.name} failed: {e}")
                continue
        
        # If all fail
        raise HTTPException(503, "Service Unavailable")

orchestrator = AnalysisOrchestrator()

# =============================================================================
# APP
# =============================================================================

app = FastAPI(title="NoBluff API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_endpoint(
    file: UploadFile = File(...),
    mime_type: str = Form(...)
):
    try:
        content = await file.read()
        if len(content) < 1000: # Simple byte check for empty files
             return AnalysisResponse(verdict="INCONCLUSIVE", confidence=0.0, reason="Audio too short/empty.")
             
        result = await orchestrator.analyze(content, mime_type)
        return AnalysisResponse(
            verdict=result.verdict.value,
            confidence=result.confidence,
            reason=result.reason
        )
    except Exception as e:
        logger.exception("Error")
        if isinstance(e, HTTPException): raise e
        raise HTTPException(500, "Internal Server Error")
