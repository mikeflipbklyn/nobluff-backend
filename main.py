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
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

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

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("nobluff")


# =============================================================================
# DOMAIN MODELS
# =============================================================================


class Verdict(str, Enum):
    BLUFF = "BLUFF"
    NO_BLUFF = "NO BLUFF"


@dataclass
class AnalysisResult:
    """Internal result from AI provider analysis."""
    verdict: Verdict
    confidence: float
    reason: str
    provider: str


class AnalysisResponse(BaseModel):
    """API response schema."""
    verdict: str = Field(..., description="BLUFF or NO BLUFF")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    reason: str = Field(..., description="Explanation for the verdict")


class ErrorResponse(BaseModel):
    """Standardized error response."""
    error: str
    code: str
    detail: Optional[str] = None


# =============================================================================
# EXCEPTIONS
# =============================================================================


class AIProviderError(Exception):
    """Base exception for AI provider failures."""
    def __init__(self, provider: str, message: str, retriable: bool = True):
        self.provider = provider
        self.message = message
        self.retriable = retriable
        super().__init__(f"[{provider}] {message}")


class RateLimitError(AIProviderError):
    """Rate limit exceeded."""
    def __init__(self, provider: str):
        super().__init__(provider, "Rate limit exceeded", retriable=True)


class ServiceUnavailableError(AIProviderError):
    """Service temporarily unavailable."""
    def __init__(self, provider: str, detail: str = ""):
        super().__init__(provider, f"Service unavailable: {detail}", retriable=True)


class InvalidResponseError(AIProviderError):
    """AI returned unparseable response."""
    def __init__(self, provider: str, detail: str = ""):
        super().__init__(provider, f"Invalid response: {detail}", retriable=False)


# =============================================================================
# AI PROVIDER ABSTRACTION
# =============================================================================


class AIProvider(ABC):
    """
    Abstract base class for AI analysis providers.
    
    Implements the Strategy pattern for swappable AI backends.
    Each provider must implement audio analysis and return standardized results.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier for logging."""
        pass
    
    @abstractmethod
    async def analyze_audio(
        self, 
        audio_data: bytes, 
        mime_type: str
    ) -> AnalysisResult:
        """
        Analyze audio for deception indicators.
        
        Args:
            audio_data: Raw audio bytes
            mime_type: MIME type (e.g., 'audio/wav', 'audio/m4a')
            
        Returns:
            AnalysisResult with verdict, confidence, and reasoning
            
        Raises:
            AIProviderError: On any failure
        """
        pass
    
    def _build_analysis_prompt(self) -> str:
        """Shared prompt for consistent analysis across providers."""
        return """You are an expert deception analyst. Analyze this audio recording for signs of deception.

ANALYZE FOR:
1. Vocal stress patterns (pitch variations, tremors)
2. Speech hesitations and filler words
3. Response latency and pacing changes
4. Micro-expressions in voice (clearing throat, swallowing)
5. Inconsistencies in narrative flow
6. Overcompensation (too much detail, excessive qualifiers)

RESPONSE FORMAT (JSON only, no markdown):
{
    "verdict": "BLUFF" or "NO BLUFF",
    "confidence": <float between 0.0 and 1.0>,
    "reason": "<concise explanation of key indicators>"
}

Be decisive. Provide exactly one verdict based on the balance of evidence."""

    def _parse_verdict_response(self, text: str) -> tuple[Verdict, float, str]:
        """
        Parse AI response text into structured verdict.
        
        Handles both clean JSON and markdown-wrapped responses.
        """
        import json
        import re
        
        # Strip markdown code blocks if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            # Remove ```json and ``` wrappers
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            # Attempt regex extraction as fallback
            verdict_match = re.search(r'"verdict"\s*:\s*"(BLUFF|NO BLUFF)"', text, re.I)
            conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', text)
            reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', text)
            
            if verdict_match and conf_match:
                verdict_str = verdict_match.group(1).upper()
                verdict = Verdict.BLUFF if verdict_str == "BLUFF" else Verdict.NO_BLUFF
                confidence = min(1.0, max(0.0, float(conf_match.group(1))))
                reason = reason_match.group(1) if reason_match else "Analysis complete"
                return verdict, confidence, reason
            
            raise InvalidResponseError(self.name, f"JSON parse failed: {e}")
        
        # Normalize verdict string
        verdict_str = str(data.get("verdict", "")).upper().strip()
        if verdict_str == "BLUFF":
            verdict = Verdict.BLUFF
        elif verdict_str in ("NO BLUFF", "NO_BLUFF", "NOBLUFF"):
            verdict = Verdict.NO_BLUFF
        else:
            raise InvalidResponseError(self.name, f"Invalid verdict: {verdict_str}")
        
        confidence = float(data.get("confidence", 0.5))
        confidence = min(1.0, max(0.0, confidence))
        
        reason = str(data.get("reason", "No explanation provided"))
        
        return verdict, confidence, reason


# =============================================================================
# GEMINI SERVICE
# =============================================================================


class GeminiService(AIProvider):
    """
    Google Gemini 1.5 Flash implementation.
    
    Primary provider - fast and cost-effective for audio analysis.
    Uses the generativelanguage API with inline audio data.
    """
    
    API_BASE = "https://generativelanguage.googleapis.com/v1beta"
    MODEL = "models/gemini-3-flash-preview"
    TIMEOUT = 30.0
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def name(self) -> str:
        return "Gemini"
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.TIMEOUT)
        return self._client
    
    async def analyze_audio(
        self, 
        audio_data: bytes, 
        mime_type: str
    ) -> AnalysisResult:
        """Send audio to Gemini for analysis."""
        
        logger.info(f"[Gemini] Analyzing {len(audio_data)} bytes, type={mime_type}")
        
        # Normalize MIME type for Gemini
        gemini_mime = self._normalize_mime_type(mime_type)
        
        # Encode audio as base64
        audio_b64 = base64.standard_b64encode(audio_data).decode("utf-8")
        
        # Build request payload
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": gemini_mime,
                                "data": audio_b64
                            }
                        },
                        {
                            "text": self._build_analysis_prompt()
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 500,
            }
        }
        
        url = f"{self.API_BASE}/{self.MODEL}:generateContent?key={self.api_key}"
        
        try:
            client = await self._get_client()
            response = await client.post(url, json=payload)
            
            # Handle error responses
            if response.status_code == 429:
                logger.warning("[Gemini] Rate limited")
                raise RateLimitError("Gemini")
            
            if response.status_code >= 500:
                logger.error(f"[Gemini] Server error: {response.status_code}")
                raise ServiceUnavailableError("Gemini", f"HTTP {response.status_code}")
            
            if response.status_code != 200:
                detail = response.text[:200]
                logger.error(f"[Gemini] API error: {response.status_code} - {detail}")
                raise AIProviderError("Gemini", f"HTTP {response.status_code}: {detail}")
            
            # Parse response
            data = response.json()
            
            # Extract text from Gemini response structure
            candidates = data.get("candidates", [])
            if not candidates:
                raise InvalidResponseError("Gemini", "No candidates in response")
            
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            if not parts:
                raise InvalidResponseError("Gemini", "No parts in response")
            
            text = parts[0].get("text", "")
            if not text:
                raise InvalidResponseError("Gemini", "Empty text response")
            
            logger.debug(f"[Gemini] Raw response: {text[:200]}")
            
            verdict, confidence, reason = self._parse_verdict_response(text)
            
            logger.info(f"[Gemini] Verdict: {verdict.value}, Confidence: {confidence:.2f}")
            
            return AnalysisResult(
                verdict=verdict,
                confidence=confidence,
                reason=reason,
                provider="Gemini"
            )
            
        except httpx.TimeoutException:
            logger.error("[Gemini] Request timeout")
            raise ServiceUnavailableError("Gemini", "Timeout")
        
        except httpx.RequestError as e:
            logger.error(f"[Gemini] Network error: {e}")
            raise ServiceUnavailableError("Gemini", str(e))
    
    def _normalize_mime_type(self, mime_type: str) -> str:
        """Map common MIME types to Gemini-supported formats."""
        mime_map = {
            "audio/m4a": "audio/mp4",
            "audio/x-m4a": "audio/mp4",
            "audio/mpeg": "audio/mp3",
            "audio/wave": "audio/wav",
            "audio/x-wav": "audio/wav",
        }
        return mime_map.get(mime_type.lower(), mime_type.lower())


# =============================================================================
# OPENAI SERVICE
# =============================================================================


class OpenAIService(AIProvider):
    """
    OpenAI GPT-4o-mini implementation with Whisper transcription.
    
    Fallback provider - uses Whisper for speech-to-text,
    then GPT-4o-mini for deception analysis on the transcript.
    """
    
    API_BASE = "https://api.openai.com/v1"
    WHISPER_MODEL = "whisper-1"
    CHAT_MODEL = "gpt-4o-mini"
    TIMEOUT = 60.0  # Longer timeout for two-stage processing
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def name(self) -> str:
        return "OpenAI"
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.TIMEOUT)
        return self._client
    
    async def analyze_audio(
        self, 
        audio_data: bytes, 
        mime_type: str
    ) -> AnalysisResult:
        """
        Two-stage analysis:
        1. Transcribe with Whisper
        2. Analyze transcript with GPT-4o-mini
        """
        
        logger.info(f"[OpenAI] Starting analysis, {len(audio_data)} bytes")
        
        # Stage 1: Transcription
        transcript = await self._transcribe(audio_data, mime_type)
        logger.info(f"[OpenAI] Transcribed: {len(transcript)} chars")
        
        # Stage 2: Deception analysis
        result = await self._analyze_transcript(transcript)
        
        return result
    
    async def _transcribe(self, audio_data: bytes, mime_type: str) -> str:
        """Transcribe audio using Whisper API."""
        
        # Determine file extension from MIME type
        ext_map = {
            "audio/wav": "wav",
            "audio/wave": "wav",
            "audio/x-wav": "wav",
            "audio/mp3": "mp3",
            "audio/mpeg": "mp3",
            "audio/m4a": "m4a",
            "audio/mp4": "m4a",
            "audio/x-m4a": "m4a",
            "audio/ogg": "ogg",
            "audio/webm": "webm",
        }
        ext = ext_map.get(mime_type.lower(), "wav")
        
        # Create multipart form data
        files = {
            "file": (f"audio.{ext}", audio_data, mime_type),
            "model": (None, self.WHISPER_MODEL),
            "response_format": (None, "text"),
        }
        
        url = f"{self.API_BASE}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            client = await self._get_client()
            response = await client.post(url, headers=headers, files=files)
            
            if response.status_code == 429:
                logger.warning("[OpenAI/Whisper] Rate limited")
                raise RateLimitError("OpenAI")
            
            if response.status_code >= 500:
                logger.error(f"[OpenAI/Whisper] Server error: {response.status_code}")
                raise ServiceUnavailableError("OpenAI", f"Whisper HTTP {response.status_code}")
            
            if response.status_code != 200:
                detail = response.text[:200]
                logger.error(f"[OpenAI/Whisper] Error: {response.status_code} - {detail}")
                raise AIProviderError("OpenAI", f"Whisper error: {detail}")
            
            return response.text.strip()
            
        except httpx.TimeoutException:
            logger.error("[OpenAI/Whisper] Timeout")
            raise ServiceUnavailableError("OpenAI", "Whisper timeout")
        
        except httpx.RequestError as e:
            logger.error(f"[OpenAI/Whisper] Network error: {e}")
            raise ServiceUnavailableError("OpenAI", str(e))
    
    async def _analyze_transcript(self, transcript: str) -> AnalysisResult:
        """Analyze transcript for deception using GPT-4o-mini."""
        
        # Build analysis prompt with transcript
        prompt = f"""{self._build_analysis_prompt()}

TRANSCRIPT TO ANALYZE:
\"\"\"
{transcript}
\"\"\"

Analyze the speech patterns, hesitations, and content for signs of deception."""
        
        payload = {
            "model": self.CHAT_MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert deception analyst. Respond only with valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.3,
            "max_tokens": 500,
        }
        
        url = f"{self.API_BASE}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            client = await self._get_client()
            response = await client.post(url, headers=headers, json=payload)
            
            if response.status_code == 429:
                logger.warning("[OpenAI/Chat] Rate limited")
                raise RateLimitError("OpenAI")
            
            if response.status_code >= 500:
                logger.error(f"[OpenAI/Chat] Server error: {response.status_code}")
                raise ServiceUnavailableError("OpenAI", f"Chat HTTP {response.status_code}")
            
            if response.status_code != 200:
                detail = response.text[:200]
                logger.error(f"[OpenAI/Chat] Error: {response.status_code} - {detail}")
                raise AIProviderError("OpenAI", f"Chat error: {detail}")
            
            data = response.json()
            
            choices = data.get("choices", [])
            if not choices:
                raise InvalidResponseError("OpenAI", "No choices in response")
            
            text = choices[0].get("message", {}).get("content", "")
            if not text:
                raise InvalidResponseError("OpenAI", "Empty response content")
            
            logger.debug(f"[OpenAI] Raw response: {text[:200]}")
            
            verdict, confidence, reason = self._parse_verdict_response(text)
            
            logger.info(f"[OpenAI] Verdict: {verdict.value}, Confidence: {confidence:.2f}")
            
            return AnalysisResult(
                verdict=verdict,
                confidence=confidence,
                reason=reason,
                provider="OpenAI"
            )
            
        except httpx.TimeoutException:
            logger.error("[OpenAI/Chat] Timeout")
            raise ServiceUnavailableError("OpenAI", "Chat timeout")
        
        except httpx.RequestError as e:
            logger.error(f"[OpenAI/Chat] Network error: {e}")
            raise ServiceUnavailableError("OpenAI", str(e))


# =============================================================================
# ORCHESTRATOR - FAILOVER CHAIN
# =============================================================================


class AnalysisOrchestrator:
    """
    Orchestrates AI provider failover chain.
    
    Pattern: Primary -> Fallback -> Error
    Gemini (fast, cheap) -> OpenAI (reliable backup) -> ServiceUnavailable
    """
    
    def __init__(self):
        self.providers: list[AIProvider] = []
        
        # Initialize providers if keys available
        if GOOGLE_API_KEY:
            self.providers.append(GeminiService(GOOGLE_API_KEY))
            logger.info("Gemini provider initialized")
        else:
            logger.warning("GOOGLE_API_KEY not set - Gemini unavailable")
        
        if OPENAI_API_KEY:
            self.providers.append(OpenAIService(OPENAI_API_KEY))
            logger.info("OpenAI provider initialized")
        else:
            logger.warning("OPENAI_API_KEY not set - OpenAI unavailable")
        
        if not self.providers:
            logger.critical("No AI providers available!")
    
    async def analyze(
        self, 
        audio_data: bytes, 
        mime_type: str
    ) -> AnalysisResult:
        """
        Execute failover chain until one provider succeeds.
        
        Raises:
            HTTPException(503): All providers failed
        """
        
        if not self.providers:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "error": "No AI providers configured",
                    "code": "NO_PROVIDERS"
                }
            )
        
        errors: list[str] = []
        
        for provider in self.providers:
            try:
                logger.info(f"Attempting analysis with {provider.name}")
                result = await provider.analyze_audio(audio_data, mime_type)
                logger.info(f"Success with {provider.name}")
                return result
                
            except AIProviderError as e:
                error_msg = f"{provider.name}: {e.message}"
                errors.append(error_msg)
                logger.warning(f"Provider failed: {error_msg}")
                
                # Continue to next provider if retriable
                if e.retriable:
                    continue
                else:
                    # Non-retriable error, still try next provider
                    continue
        
        # All providers failed
        logger.error(f"All providers failed: {errors}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "All AI providers failed",
                "code": "SERVICE_UNAVAILABLE",
                "providers_tried": errors
            }
        )


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="NoBluff API",
    description="High-stakes lie detection API with AI failover",
    version="1.0.0",
)

# CORS middleware for iOS client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Initialize orchestrator
orchestrator = AnalysisOrchestrator()


@app.get("/health")
async def health_check():
    """Health check endpoint for Render."""
    return {
        "status": "healthy",
        "providers": [p.name for p in orchestrator.providers]
    }


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={
        503: {"model": ErrorResponse, "description": "All providers failed"}
    }
)
async def analyze_audio(
    file: UploadFile = File(..., description="Audio file to analyze"),
    mime_type: str = Form(..., description="MIME type of the audio file"),
):
    """
    Analyze audio for deception indicators.
    
    Uses Gemini as primary provider with OpenAI fallback.
    Returns verdict (BLUFF/NO BLUFF), confidence score, and reasoning.
    """
    
    logger.info(f"Received analysis request: {file.filename}, type={mime_type}")
    
    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "No file provided", "code": "MISSING_FILE"}
        )
    
    # Read audio data
    try:
        audio_data = await file.read()
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Failed to read audio file", "code": "READ_ERROR"}
        )
    
    if len(audio_data) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Empty audio file", "code": "EMPTY_FILE"}
        )
    
    # Size limit: 25MB
    if len(audio_data) > 25 * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"error": "File too large (max 25MB)", "code": "FILE_TOO_LARGE"}
        )
    
    logger.info(f"Processing {len(audio_data)} bytes of audio")
    
    # Execute analysis with failover
    result = await orchestrator.analyze(audio_data, mime_type)
    
    return AnalysisResponse(
        verdict=result.verdict.value,
        confidence=result.confidence,
        reason=result.reason,
    )


# =============================================================================
# GUNICORN ENTRY POINT
# =============================================================================

# For local development:
#   uvicorn main:app --reload --port 8000
#
# For Render.com (gunicorn):
#   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
