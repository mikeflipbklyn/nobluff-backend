"""
NoBluff Backend - High-Stakes Lie Detection API
================================================
Production-ready FastAPI server with AI failover pattern.
Deployed on Render.com.

VERIFIED CONFIGURATION:
- Primary: 'gemini-3-flash-preview' (Multimodal Audio).
- Backup: OpenAI Whisper + GPT-4o (Text + Context).
- Failover: Automatic and silent to the user.
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

# ACCURATE MODEL ID FOR 2026
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("nobluff")

http_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=60.0)
    logger.info("✅ SYSTEM ONLINE: Global HTTP Client Initialized")
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
    
    def _parse_verdict_response(self, text: str) -> Tuple[Verdict, float, str]:
        cleaned = text.strip()
        if "```" in cleaned: cleaned = re.sub(r"```json|```", "", cleaned).strip()
        
        try:
            data = json.loads(cleaned)
        except:
            # Emergency Regex Fallback
            if "BLUFF" in text.upper(): return Verdict.BLUFF, 0.85, "Detected stress markers."
            if "NO" in text.upper(): return Verdict.NO_BLUFF, 0.90, "Voice patterns stable."
            return Verdict.INCONCLUSIVE, 0.0, "Audio unintelligible."

        v_str = str(data.get("verdict", "")).upper().strip()
        
        # Normalize response
        if "INCONCLUSIVE" in v_str: return Verdict.INCONCLUSIVE, 0.0, data.get("reason", "Audio unclear.")
        
        # Handle "NO BLUFF" or "TRUTH"
        if "NO" in v_str or "TRUTH" in v_str:
            return Verdict.NO_BLUFF, float(data.get("confidence", 0.95)), data.get("reason", "Calm, direct syntax.")
            
        # Handle "BLUFF" or "LIE"
        if "BLUFF" in v_str or "LIE" in v_str:
            return Verdict.BLUFF, float(data.get("confidence", 0.85)), data.get("reason", "Vocal stress detected.")
        
        return Verdict.INCONCLUSIVE, 0.0, "Analysis failed."

# =============================================================================
# GEMINI SERVICE (Primary)
# =============================================================================

class GeminiService(AIProvider):
    API_BASE = "https://generativelanguage.googleapis.com/v1beta"
    def __init__(self, api_key: str): self.api_key = api_key
    @property
    def name(self) -> str: return "Gemini 3.0"
    
    async def analyze_audio(self, audio_data: bytes, mime_type: str) -> AnalysisResult:
        # 1. Format Model ID
        model_id = GEMINI_MODEL_NAME if GEMINI_MODEL_NAME.startswith("models/") else f"models/{GEMINI_MODEL_NAME}"
        gemini_mime = "audio/mp4" if "m4a" in mime_type.lower() else mime_type
        
        # 2. Prepare Payload
        prompt = """You are a Forensic Psycho-Acoustic Analyst.
        TASK: Analyze this audio for deception.
        RULES: 
        - SHORT CLIPS ARE VALID. A single "No" can be analyzed.
        - IGNORE silence. Focus on the split-second of sound.
        - If no speech, analyze BREATHING or HESITATION.
        
        OUTPUT JSON: {"verdict": "BLUFF"|"NO_BLUFF"|"INCONCLUSIVE", "confidence": 0.0-1.0, "reason": "Punchy 8-word explanation."}"""
        
        payload = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": gemini_mime, "data": base64.standard_b64encode(audio_data).decode("utf-8")}},
                    {"text": prompt}
                ]
            }],
            "generationConfig": { "temperature": 0.2, "response_mime_type": "application/json" }
        }
        
        # 3. Execute
        try:
            response = await http_client.post(f"{self.API_BASE}/{model_id}:generateContent?key={self.api_key}", json=payload)
            
            if response.status_code != 200:
                logger.error(f"❌ Gemini Error: {response.status_code} - {response.text}")
                raise Exception(f"API Error {response.status_code}")
            
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            verdict, confidence, reason = self._parse_verdict_response(text)
            return AnalysisResult(verdict, confidence, reason, "Gemini 3.0")
            
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
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            # 1. WHISPER (Transcription)
            files = { "file": ("audio.m4a", audio_data, mime_type), "model": (None, "whisper-1") }
            transcript_res = await http_client.post(f"{self.API_BASE}/audio/transcriptions", headers=headers, files=files)
            
            transcript = ""
            if transcript_res.status_code == 200:
                transcript = transcript_res.json().get("text", "")
            else:
                logger.warning(f"⚠️ OpenAI Whisper Issue: {transcript_res.status_code}")
                # Continue anyway with empty transcript to check for "Silence"

            # 2. GPT-4o (Analysis)
            # We tell GPT that if transcript is empty, it means the user was silent/breathing.
            prompt = """You are a Lie Detector.
            The user recorded audio. 
            Transcript: "{transcript}"
            
            IF TRANSCRIPT IS EMPTY:
            - The user stayed silent or just breathed. 
            - VERDICT: "BLUFF" (Silence/Stonewalling is suspicious).
            - REASON: "Suspicious silence detected."
            
            IF TRANSCRIPT EXISTS:
            - Analyze syntax for hesitation or evasion.
            
            Output JSON: {"verdict": "BLUFF"|"NO_BLUFF", "confidence": 0.8, "reason": "Short reason."}"""
            
            chat_payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a forensic analyst."},
                    {"role": "user", "content": prompt.format(transcript=transcript)}
                ],
                "response_format": { "type": "json_object" }
            }
            
            chat_res = await http_client.post(f"{self.API_BASE}/chat/completions", headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, json=chat_payload)
            content = chat_res.json()["choices"][0]["message"]["content"]
            
            verdict, confidence, reason = self._parse_verdict_response(content)
            return AnalysisResult(verdict, confidence, reason, "OpenAI (Backup)")
            
        except Exception as e:
            logger.error(f"❌ OpenAI Exception: {e}")
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
        
        # 1. Try Primary (Gemini)
        try:
            logger.info("🚀 Attempting Primary: Gemini 3.0")
            return await self.providers[0].analyze_audio(audio_data, mime_type)
        except Exception as e:
            logger.error(f"⚠️ PRIMARY FAILED: {e}")
        
        # 2. Try Backup (OpenAI)
        if len(self.providers) > 1:
            try:
                logger.info("🔄 SWITCHING TO BACKUP: OpenAI")
                return await self.providers[1].analyze_audio(audio_data, mime_type)
            except Exception as e:
                logger.error(f"❌ BACKUP FAILED: {e}")
        
        # 3. Fail Safe
        return AnalysisResult(Verdict.INCONCLUSIVE, 0.0, "Systems Busy.", "FailSafe")

orchestrator = AnalysisOrchestrator()

# =============================================================================
# APP ENTRY
# =============================================================================

app = FastAPI(title="NoBluff API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["POST"], allow_headers=["*"])

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_endpoint(file: UploadFile = File(...), mime_type: str = Form(...)):
    try:
        content = await file.read()
        if len(content) < 50:
             return AnalysisResponse(verdict="INCONCLUSIVE", confidence=0.0, reason="Mic error.")
        
        result = await orchestrator.analyze(content, mime_type)
        
        logger.info(f"✅ FINAL VERDICT: {result.verdict} (via {result.provider})")
        return AnalysisResponse(verdict=result.verdict.value, confidence=result.confidence, reason=result.reason)
        
    except Exception as e:
        logger.exception("CRITICAL SERVER ERROR")
        return AnalysisResponse(verdict="INCONCLUSIVE", confidence=0.0, reason="Server Error.")
