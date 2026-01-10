"""
NoBluff Backend - State-of-the-Art Lie Detection
Optimized for Gemini 3.0 Multimodal Audio Reasoning.
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
from typing import Optional, Tuple, Dict, Any
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =============================================================================
# CONFIGURATION & LOGGING
# =============================================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("nobluff")

http_client: Optional[httpx.AsyncClient] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=90.0)
    logger.info("🚀 NOBLUFF AI CORE ONLINE")
    yield
    if http_client: await http_client.aclose()

# =============================================================================
# DOMAIN MODELS
# =============================================================================

class Verdict(str, Enum):
    """Internal verdict enum - values match iOS VerdictType raw values."""
    BLUFF = "bluff"
    NO_BLUFF = "no_bluff"
    INCONCLUSIVE = "inconclusive"

@dataclass
class AnalysisResult:
    verdict: Verdict
    confidence: float
    analysis: str  # Changed from 'reason' to match iOS field name
    prosody_score: float
    linguistic_score: float
    provider: str

class AnalysisResponse(BaseModel):
    """API response - field names must match iOS Verdict.swift CodingKeys."""
    verdict: str  # "bluff", "no_bluff", or "inconclusive" (lowercase)
    confidence: float
    analysis: str  # iOS maps this to 'reason' via CodingKeys

# =============================================================================
# GEMINI 3.0 MULTIMODAL SERVICE (Primary)
# =============================================================================

class GeminiService:
    """
    Expert-level Gemini implementation utilizing native audio tokens 
    to detect micro-tremors and cognitive dissonance.
    """
    API_BASE = "https://generativelanguage.googleapis.com/v1beta"
    
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def analyze_audio(self, audio_data: bytes, mime_type: str, model_name: str = None) -> AnalysisResult:
        # Use provided model or fall back to env default
        model = model_name or GEMINI_MODEL_NAME
        model_id = model if model.startswith("models/") else f"models/{model}"
        
        # Entertainment-focused prompt (avoid overclaiming accuracy)
        system_instruction = (
            "You are analyzing audio for an entertainment lie detection app. "
            "Listen for vocal patterns that might suggest stress or hesitation. "
            "This is for fun - not forensic use. Be decisive but fair."
        )

        prompt = """
        Analyze this audio clip for signs the speaker might be bluffing.
        
        Look for:
        - Vocal hesitations or unusual pauses
        - Changes in pitch or speaking pace
        - Hedging language or excessive qualifiers
        
        If speech sounds natural and confident: no_bluff
        If speech sounds stressed or evasive: bluff
        If audio is unclear or too short: inconclusive
        
        Respond with ONLY this JSON (no markdown):
        {
          "verdict": "bluff" or "no_bluff" or "inconclusive",
          "confidence": 0.0 to 1.0,
          "analysis": "Brief explanation (max 20 words)"
        }
        """

        payload = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": "audio/mp4" if "m4a" in mime_type else mime_type,
                                     "data": base64.standard_b64encode(audio_data).decode("utf-8")}},
                    {"text": prompt}
                ]
            }],
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {
                "temperature": 0.2,
                "response_mime_type": "application/json"
            }
        }

        try:
            url = f"{self.API_BASE}/{model_id}:generateContent?key={self.api_key}"
            response = await http_client.post(url, json=payload)
            
            if response.status_code == 429:
                raise Exception("Rate limit exceeded")
            
            if response.status_code != 200:
                raise Exception(f"Gemini API Error: {response.text}")

            res_json = response.json()
            raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
            data = json.loads(raw_text)
            
            # Normalize verdict to lowercase
            verdict_str = str(data.get("verdict", "inconclusive")).lower().strip()
            if verdict_str in ("bluff",):
                verdict = Verdict.BLUFF
            elif verdict_str in ("no_bluff", "nobluff", "no bluff", "no-bluff"):
                verdict = Verdict.NO_BLUFF
            else:
                verdict = Verdict.INCONCLUSIVE

            return AnalysisResult(
                verdict=verdict,
                confidence=min(1.0, max(0.0, float(data.get("confidence", 0.5)))),
                analysis=data.get("analysis", data.get("reason", "Analysis complete.")),
                prosody_score=data.get("prosody_score", 0.0),
                linguistic_score=data.get("linguistic_score", 0.0),
                provider="Gemini"
            )
        except Exception as e:
            logger.error(f"Gemini Processing Failure: {e}")
            raise e

# =============================================================================
# OPENAI BACKUP SERVICE
# =============================================================================

class OpenAIService:
    """Fallback using Whisper and GPT-4o-mini."""
    API_BASE = "https://api.openai.com/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def analyze_audio(self, audio_data: bytes, mime_type: str) -> AnalysisResult:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # 1. Transcription (Whisper)
        ext_map = {"audio/wav": "wav", "audio/m4a": "m4a", "audio/mp4": "m4a", "audio/mpeg": "mp3"}
        ext = ext_map.get(mime_type, "wav")
        files = {"file": (f"audio.{ext}", audio_data, mime_type), "model": (None, "whisper-1")}
        
        trans_res = await http_client.post(f"{self.API_BASE}/audio/transcriptions", headers=headers, files=files)
        
        if trans_res.status_code != 200:
            raise Exception(f"Whisper error: {trans_res.text[:200]}")
            
        transcript = trans_res.json().get("text", "")
        logger.info(f"[OpenAI] Transcript: {transcript[:100]}...")

        # 2. Analysis (GPT-4o-mini)
        prompt = f"""Analyze this transcript for signs the speaker might be bluffing.
        
Transcript: "{transcript}"

Look for hedging, excessive qualifiers, or evasive language.
Respond with ONLY JSON (no markdown):
{{"verdict": "bluff" or "no_bluff", "confidence": 0.0-1.0, "analysis": "brief explanation"}}"""

        chat_payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You analyze speech for an entertainment app. Respond only with valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 200
        }
        
        chat_res = await http_client.post(
            f"{self.API_BASE}/chat/completions",
            headers={**headers, "Content-Type": "application/json"},
            json=chat_payload
        )
        
        if chat_res.status_code != 200:
            raise Exception(f"Chat error: {chat_res.text[:200]}")
        
        raw_content = chat_res.json()["choices"][0]["message"]["content"]
        
        # Clean markdown if present
        cleaned = raw_content.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        
        data = json.loads(cleaned)
        
        # Normalize verdict
        verdict_str = str(data.get("verdict", "no_bluff")).lower().strip()
        if verdict_str in ("bluff",):
            verdict = Verdict.BLUFF
        elif verdict_str in ("no_bluff", "nobluff", "no bluff"):
            verdict = Verdict.NO_BLUFF
        else:
            verdict = Verdict.INCONCLUSIVE

        return AnalysisResult(
            verdict=verdict,
            confidence=min(1.0, max(0.0, float(data.get("confidence", 0.7)))),
            analysis=data.get("analysis", data.get("reason", "Textual analysis only.")),
            prosody_score=0.5,
            linguistic_score=0.8,
            provider="OpenAI"
        )

# =============================================================================
# ORCHESTRATOR & ENDPOINTS
# =============================================================================

class NoBluffOrchestrator:
    def __init__(self):
        self.gemini = GeminiService(GOOGLE_API_KEY) if GOOGLE_API_KEY else None
        self.openai = OpenAIService(OPENAI_API_KEY) if OPENAI_API_KEY else None
        
        # DEFINING THE HIERARCHY
        # 1. gemini-3-flash-preview: Highest reasoning (if available)
        # 2. gemini-2.5-flash: The 'Fast & Intelligent' price-performance king
        # 3. gemini-2.0-flash: Stable fallback
        self.model_rotation = [
            "gemini-3-flash-preview",
            "gemini-2.5-flash",
            "gemini-2.0-flash"
        ]

    async def run_analysis(self, audio: bytes, mime: str) -> AnalysisResult:
        if not self.gemini:
            return await self._run_openai_fallback(audio, mime)
        
        for model in self.model_rotation:
            try:
                logger.info(f"🔮 Analyzing via {model}...")
                result = await self.gemini.analyze_audio(audio, mime, model_name=model)
                logger.info(f"✅ {model} succeeded")
                return result
            except Exception as e:
                error_str = str(e).lower()
                # If we hit a 429 (Too Many Requests) or quota error, rotate to next model
                if "429" in str(e) or "quota" in error_str or "rate" in error_str:
                    logger.warning(f"⚠️ {model} Quota/Rate Limited. Rotating to next model...")
                    continue
                # Also rotate on model not found errors
                if "404" in str(e) or "not found" in error_str:
                    logger.warning(f"⚠️ {model} not available. Rotating to next model...")
                    continue
                
                logger.error(f"❌ {model} Critical Error: {e}")
                break  # Stop rotating on auth/logic errors
        
        return await self._run_openai_fallback(audio, mime)

    async def _run_openai_fallback(self, audio: bytes, mime: str) -> AnalysisResult:
        if self.openai:
            try:
                logger.warning("🚀 All Gemini engines offline. Engaging OpenAI Core...")
                return await self.openai.analyze_audio(audio, mime)
            except Exception as e:
                logger.error(f"💀 OpenAI fallback also failed: {e}")
        
        # The 'Psychological Safety' result: keeps the user in the app's world
        return AnalysisResult(
            verdict=Verdict.INCONCLUSIVE,
            confidence=0.0,
            analysis="System desynchronization. Please try again.",
            prosody_score=0,
            linguistic_score=0,
            provider="FailSafe"
        )

orchestrator = NoBluffOrchestrator()
app = FastAPI(title="NoBluff AI", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health_check():
    """Health check endpoint for Render.com."""
    providers = []
    if GOOGLE_API_KEY:
        providers.append("Gemini")
    if OPENAI_API_KEY:
        providers.append("OpenAI")
    return {
        "status": "healthy",
        "providers": providers,
        "gemini_rotation": orchestrator.model_rotation,
        "openai_fallback": "gpt-4o-mini + whisper-1"
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...), mime_type: str = Form(...)):
    # 1. Read and validate
    content = await file.read()
    if len(content) < 1000:
        return AnalysisResponse(
            verdict="inconclusive",
            confidence=0.0,
            analysis="Audio clip too short for analysis."
        )

    # 2. Run Orchestrator
    result = await orchestrator.run_analysis(content, mime_type)
    
    # 3. Log
    logger.info(f"Verdict: {result.verdict.value} | Conf: {result.confidence} | Reason: {result.analysis[:50]}")

    # 4. Return iOS-compatible response
    return AnalysisResponse(
        verdict=result.verdict.value,  # Now lowercase: "bluff", "no_bluff", "inconclusive"
        confidence=result.confidence,
        analysis=result.analysis  # iOS maps this to 'reason' via CodingKeys
    )
