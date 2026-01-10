"""
NoBluff Backend - High-Stakes Lie Detection API
================================================
Production-ready FastAPI server with AI failover pattern.
Deployed on Render.com.

UPDATES:
- UPGRADED Model to 'gemini-3-flash-preview' (Correct ID found in docs).
- TUNED for Short Audio (1-3s) analysis.
- MAINTAINED OpenAI Failover.
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

# CRITICAL FIX: The correct ID for Jan 2026
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("nobluff")

http_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=60.0)
    yield
    if http_client: await http_client.aclose()

# =============================================================================
# DOMAIN MODELS
# =============================================================================

class Verdict(str, Enum):
    BLUFF = "BLUFF"
    NO_BLUFF = "NO_BLUFF"
    INCONCLUSIVE = "INCONCLUSIVE"

@dataclass
class AnalysisResult:
    verdict: Verdict
    confidence: float
    reason: str
    provider: str

class AnalysisResponse(BaseModel):
    verdict: str
    confidence: float
    reason: str

# =============================================================================
# AI PROVIDER ABSTRACTION
# =============================================================================

class AIProvider(ABC):
    @abstractmethod
    async def analyze_audio(self, audio_data: bytes, mime_type: str) -> AnalysisResult: pass
    @property
    @abstractmethod
    def name(self) -> str: pass
    
    def _build_analysis_prompt(self) -> str:
        return """You are a Forensic Psycho-Acoustic Analyst.

INSTRUCTIONS:
1.  **Analyze EVERYTHING.** Even a sigh, a grunt, or a single "No" is valid data.
2.  **Short Audio is OK.** Do not return INCONCLUSIVE just because it's short.
3.  **Detect Non-Verbal Cues.** Heavy breathing = Stress. Long pause = Deception.

OUTPUT JSON:
{
    "verdict": "BLUFF" | "NO_BLUFF" | "INCONCLUSIVE",
    "confidence": 0.0-1.0,
    "reason": "Max 8 words. E.g. 'Heavy sigh detected.'"
}"""

    def _parse_verdict_response(self, text: str) -> Tuple[Verdict, float, str]:
        cleaned = text.strip()
        if "```" in cleaned: cleaned = re.sub(r"```json|```", "", cleaned).strip()
        
        try:
            data = json.loads(cleaned)
        except:
            # Fallback regex if JSON fails
            if "BLUFF" in text.upper(): return Verdict.BLUFF, 0.9, "Stress detected."
            if "NO" in text.upper(): return Verdict.NO_BLUFF, 0.9, "Calm tone."
            return Verdict.INCONCLUSIVE, 0.0, "Analysis Error."

        v_str = str(data.get("verdict", "")).upper().strip()
        
        if "INCONCLUSIVE" in v_str: return Verdict.INCONCLUSIVE, 0.0, data.get("reason", "Audio unclear.")
        if "NO" in v_str: return Verdict.NO_BLUFF, float(data.get("confidence", 0.95)), data.get("reason", "Truthful tone.")
        if "BLUFF" in v_str: return Verdict.BLUFF, float(data.get("confidence", 0.95)), data.get("reason", "Deception markers.")
        
        return Verdict.INCONCLUSIVE, 0.0, "Unclear result."

# =============================================================================
# GEMINI SERVICE (Primary: Gemini 3 Flash Preview)
# =============================================================================

class GeminiService(AIProvider):
    API_BASE = "https://generativelanguage.googleapis.com/v1beta"
    def __init__(self, api_key: str): self.api_key = api_key
    @property
    def name(self) -> str: return "Gemini 3 Flash"
    
    async def analyze_audio(self, audio_data: bytes, mime_type: str) -> AnalysisResult:
        # Handle "models/" prefix requirement
        model_id = GEMINI_MODEL_NAME if GEMINI_MODEL_NAME.startswith("models/") else f"models/{GEMINI_MODEL_NAME}"
        gemini_mime = "audio/mp4" if "m4a" in mime_type.lower() else mime_type
        
        payload = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": gemini_mime, "data": base64.standard_b64encode(audio_data).decode("utf-8")}},
                    {"text": self._build_analysis_prompt()}
                ]
            }],
            "generationConfig": { "temperature": 0.2, "response_mime_type": "application/json" }
        }
        
        try:
            response = await http_client.post(f"{self.API_BASE}/{model_id}:generateContent?key={self.api_key}", json=payload)
            if response.status_code != 200:
                logger.error(f"Gemini Error: {response.text}")
                raise Exception(f"Error {response.status_code}")
            
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            verdict, confidence, reason = self._parse_verdict_response(text)
            return AnalysisResult(verdict, confidence, reason, "Gemini 3")
        except Exception as e:
            raise Exception(str(e))

# =============================================================================
# OPENAI SERVICE (Backup)
# =============================================================================

class OpenAIService(AIProvider):
    API_BASE = "https://api.openai.com/v1"
    def __init__(self, api_key: str): self.api_key = api_key
    @property
    def name(self) -> str: return "OpenAI"
    
    async def analyze_audio(self, audio_data: bytes, mime_type: str) -> AnalysisResult:
        files = { "file": ("audio.m4a", audio_data, mime_type), "model": (None, "whisper-1") }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            # 1. Whisper
            transcript_res = await http_client.post(f"{self.API_BASE}/audio/transcriptions", headers=headers, files=files)
            if transcript_res.status_code != 200: raise Exception("Whisper Failed")
            transcript = transcript_res.json().get("text", "")
            
            # REMOVED LENGTH CHECK: Analyze everything
            
            # 2. GPT-4o Analysis
            chat_payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": self._build_analysis_prompt()},
                    {"role": "user", "content": f"TRANSCRIPT OF AUDIO: '{transcript}'. If empty, assume non-verbal sounds."}
                ],
                "response_format": { "type": "json_object" }
            }
            
            chat_res = await http_client.post(f"{self.API_BASE}/chat/completions", headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, json=chat_payload)
            content = chat_res.json()["choices"][0]["message"]["content"]
            
            verdict, confidence, reason = self._parse_verdict_response(content)
            return AnalysisResult(verdict, confidence, reason, "OpenAI")
            
        except Exception as e:
            raise Exception(str(e))

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
        
        # If all fail, return a fallback result object
        return AnalysisResult(Verdict.INCONCLUSIVE, 0.0, "System Busy.", "Fallback")

orchestrator = AnalysisOrchestrator()

# =============================================================================
# APP
# =============================================================================

app = FastAPI(title="NoBluff API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST"], allow_headers=["*"])

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_endpoint(file: UploadFile = File(...), mime_type: str = Form(...)):
    try:
        content = await file.read()
        # Ultra-low threshold: 50 bytes (Header only is usually 44 bytes)
        if len(content) < 50:
             return AnalysisResponse(verdict="INCONCLUSIVE", confidence=0.0, reason="Mic error.")
        
        result = await orchestrator.analyze(content, mime_type)
        
        return AnalysisResponse(verdict=result.verdict.value, confidence=result.confidence, reason=result.reason)
    except Exception as e:
        logger.exception("Final Error")
        return AnalysisResponse(verdict="INCONCLUSIVE", confidence=0.0, reason="Analysis failed.")
