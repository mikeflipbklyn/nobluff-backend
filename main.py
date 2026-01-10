"""
NoBluff Backend - High-Stakes Lie Detection API
================================================
Production-ready FastAPI server with AI failover pattern.
Deployed on Render.com with gunicorn.

Architecture:
    Client -> /analyze -> GeminiService -> (on failure) -> OpenAIService -> Response
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

# Gemini 3 Flash Preview - Latest model
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "models/gemini-3-flash-preview")

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
    http_client = httpx.AsyncClient(timeout=45.0)
    logger.info(f"NoBluff API Started | Gemini Model: {GEMINI_MODEL_NAME}")
    yield
    if http_client:
        await http_client.aclose()
        logger.info("HTTP Client Closed")

# =============================================================================
# DOMAIN MODELS
# =============================================================================

class Verdict(str, Enum):
    BLUFF = "BLUFF"
    NO_BLUFF = "NO BLUFF"

@dataclass
class AnalysisResult:
    verdict: Verdict
    confidence: float
    reason: str
    provider: str

class AnalysisResponse(BaseModel):
    verdict: str = Field(..., description="BLUFF or NO BLUFF")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    reason: str = Field(..., description="Punchy explanation for UI")

class ErrorResponse(BaseModel):
    error: str
    code: str
    detail: Optional[str] = None

# =============================================================================
# EXCEPTIONS
# =============================================================================

class AIProviderError(Exception):
    def __init__(self, provider: str, message: str, retriable: bool = True):
        self.provider = provider
        self.message = message
        self.retriable = retriable
        super().__init__(f"[{provider}] {message}")

class RateLimitError(AIProviderError):
    def __init__(self, provider: str):
        super().__init__(provider, "Rate limit exceeded", retriable=True)

class ServiceUnavailableError(AIProviderError):
    def __init__(self, provider: str, detail: str = ""):
        super().__init__(provider, f"Service unavailable: {detail}", retriable=True)

class InvalidResponseError(AIProviderError):
    def __init__(self, provider: str, detail: str = ""):
        super().__init__(provider, f"Invalid response: {detail}", retriable=False)

# =============================================================================
# AI PROVIDER ABSTRACTION
# =============================================================================

class AIProvider(ABC):
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    async def analyze_audio(self, audio_data: bytes, mime_type: str) -> AnalysisResult:
        pass
    
    def _build_analysis_prompt(self) -> str:
        return """You are 'NoBluff', a ruthless lie detector AI. Analyze the audio tone, hesitation, and stress markers.

OUTPUT REQUIREMENTS:
1. Verdict: 'BLUFF' (Lie/High Stress) or 'NO BLUFF' (Truth/Calm).
2. Confidence: 0.0 to 1.0.
3. Reason: A SHORT, PUNCHY verdict (Max 8 words). Do not be polite. Be clinical.
   - Good: "Voice cracks detected. High stress indicators."
   - Good: "Tone consistent. No deception found."
   - Bad: "The speaker seems to be hesitating which might suggest..."

RESPONSE FORMAT (JSON ONLY):
{
    "verdict": "BLUFF",
    "confidence": 0.95,
    "reason": "Micro-tremors detected in pitch."
}"""

    def _parse_verdict_response(self, text: str) -> Tuple[Verdict, float, str]:
        cleaned = text.strip()
        if "```" in cleaned:
            cleaned = re.sub(r"```json|```", "", cleaned).strip()
        
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning(f"JSON Decode Failed. Attempting Regex. Raw: {text[:50]}...")
            verdict_match = re.search(r'"verdict"\s*:\s*"(BLUFF|NO BLUFF)"', text, re.I)
            conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
            reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', text)
            
            if verdict_match:
                data = {
                    "verdict": verdict_match.group(1).upper(),
                    "confidence": float(conf_match.group(1)) if conf_match else 0.5,
                    "reason": reason_match.group(1) if reason_match else "Analysis Complete."
                }
            else:
                raise InvalidResponseError(self.name, "Could not parse JSON or Regex")

        verdict_str = str(data.get("verdict", "")).upper().strip()
        verdict = Verdict.BLUFF if "BLUFF" in verdict_str and "NO" not in verdict_str else Verdict.NO_BLUFF
        if "NO" in verdict_str or "TRUTH" in verdict_str: verdict = Verdict.NO_BLUFF
        
        confidence = float(data.get("confidence", 0.5))
        reason = str(data.get("reason", "Inconclusive data."))[:50]
        
        return verdict, confidence, reason

# =============================================================================
# GEMINI SERVICE (Primary) - Using Gemini 3
# =============================================================================

class GeminiService(AIProvider):
    API_BASE = "https://generativelanguage.googleapis.com/v1beta"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    @property
    def name(self) -> str: return "Gemini"
    
    async def analyze_audio(self, audio_data: bytes, mime_type: str) -> AnalysisResult:
        logger.info(f"[Gemini] Sending {len(audio_data)} bytes to {GEMINI_MODEL_NAME}")
        
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
                "temperature": 0.2,
                "response_mime_type": "application/json"
            }
        }
        
        url = f"{self.API_BASE}/{GEMINI_MODEL_NAME}:generateContent?key={self.api_key}"
        
        try:
            response = await http_client.post(url, json=payload)
            
            if response.status_code != 200:
                error_text = response.text[:200]
                logger.error(f"[Gemini] Error {response.status_code}: {error_text}")
                if response.status_code == 429: raise RateLimitError("Gemini")
                if response.status_code == 404:
                    logger.error(f"[Gemini] Model not found: {GEMINI_MODEL_NAME}")
                raise ServiceUnavailableError("Gemini", f"HTTP {response.status_code}")
            
            data = response.json()
            try:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError):
                raise InvalidResponseError("Gemini", "Unexpected API structure")

            verdict, confidence, reason = self._parse_verdict_response(text)
            logger.info(f"[Gemini] Success: {verdict.value} ({confidence:.0%})")
            return AnalysisResult(verdict, confidence, reason, "Gemini")
            
        except httpx.RequestError as e:
            raise ServiceUnavailableError("Gemini", str(e))

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
        logger.info(f"[OpenAI] Transcribing {len(audio_data)} bytes")
        
        files = {
            "file": ("audio.m4a", audio_data, mime_type),
            "model": (None, "whisper-1"),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            transcript_res = await http_client.post(
                f"{self.API_BASE}/audio/transcriptions",
                headers=headers, files=files
            )
            if transcript_res.status_code != 200:
                raise ServiceUnavailableError("OpenAI Whisper", transcript_res.text)
            
            transcript = transcript_res.json().get("text", "")
            if not transcript: raise InvalidResponseError("OpenAI", "Empty transcript")
            
            logger.info(f"[OpenAI] Transcript: {transcript[:50]}...")
            
            chat_payload = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": self._build_analysis_prompt()},
                    {"role": "user", "content": f"TRANSCRIPT: {transcript}"}
                ],
                "response_format": { "type": "json_object" }
            }
            
            json_headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            chat_res = await http_client.post(
                f"{self.API_BASE}/chat/completions",
                headers=json_headers, json=chat_payload
            )
            
            content = chat_res.json()["choices"][0]["message"]["content"]
            verdict, confidence, reason = self._parse_verdict_response(content)
            
            logger.info(f"[OpenAI] Success: {verdict.value} ({confidence:.0%})")
            return AnalysisResult(verdict, confidence, reason, "OpenAI")
            
        except httpx.RequestError as e:
            raise ServiceUnavailableError("OpenAI", str(e))

# =============================================================================
# ORCHESTRATOR
# =============================================================================

class AnalysisOrchestrator:
    def __init__(self):
        self.providers = []
        if GOOGLE_API_KEY: self.providers.append(GeminiService(GOOGLE_API_KEY))
        if OPENAI_API_KEY: self.providers.append(OpenAIService(OPENAI_API_KEY))
        logger.info(f"Orchestrator initialized with {len(self.providers)} providers")
    
    async def analyze(self, audio_data: bytes, mime_type: str) -> AnalysisResult:
        errors = []
        for provider in self.providers:
            try:
                logger.info(f"Trying provider: {provider.name}")
                return await provider.analyze_audio(audio_data, mime_type)
            except AIProviderError as e:
                logger.warning(f"Provider {provider.name} failed: {e}")
                errors.append(f"{provider.name}: {str(e)}")
                if not e.retriable: break
        
        raise HTTPException(
            status_code=503,
            detail={"error": "Analysis failed", "providers_tried": errors}
        )

orchestrator = AnalysisOrchestrator()

# =============================================================================
# APP
# =============================================================================

app = FastAPI(title="NoBluff API", version="2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "providers": [p.name for p in orchestrator.providers],
        "model": GEMINI_MODEL_NAME
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_endpoint(
    file: UploadFile = File(...),
    mime_type: str = Form(...)
):
    try:
        content = await file.read()
        if len(content) > 25 * 1024 * 1024:
            raise HTTPException(413, "File too large")
        
        logger.info(f"Received {len(content)} bytes ({mime_type})")
        result = await orchestrator.analyze(content, mime_type)
        
        return AnalysisResponse(
            verdict=result.verdict.value,
            confidence=result.confidence,
            reason=result.reason
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unhandled error")
        raise HTTPException(500, str(e))
