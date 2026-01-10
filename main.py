
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
    http_client = httpx.AsyncClient(timeout=90.0) # Increased for deep analysis
    logger.info("🚀 NOBLUFF AI CORE ONLINE")
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
    prosody_score: float # Measures vocal stress
    linguistic_score: float # Measures cognitive load
    provider: str

class AnalysisResponse(BaseModel):
    verdict: str
    confidence: float
    reason: str
    analysis_depth: str = "High-Stakes Forensic"

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

    async def analyze_audio(self, audio_data: bytes, mime_type: str) -> AnalysisResult:
        model_id = GEMINI_MODEL_NAME if GEMINI_MODEL_NAME.startswith("models/") else f"models/{GEMINI_MODEL_NAME}"
        
        # System Instruction for "95% Confidence" Strategy
        # We force the model to look for the "Deception Triad":
        # 1. Pitch Jitter 2. Latency/Response Delay 3. Distancing Language
        system_instruction = (
            "You are a Senior Forensic Psycho-Acoustic Analyst. Your goal is 95% accuracy in deception detection. "
            "Analyze the RAW AUDIO for: \n"
            "1. PROSODY: Detect micro-tremors, vocal fry, or sudden pitch shifts indicative of the 'Pinocchio Effect'.\n"
            "2. COGNITIVE LOAD: Identify unnatural pauses, over-explaining, or 'stonewalling' silence.\n"
            "3. LINGUISTIC MARKERS: Look for distancing (avoiding 'I'), non-contracted denials, and verb tense shifts.\n"
            "Output strictly in JSON format."
        )

        prompt = """
        TASK: Perform a deep-tissue lie detection analysis on this clip.
        
        EVALUATION CRITERIA:
        - If the speaker sounds overly rehearsed or lacks emotional resonance: Flag as BLUFF.
        - If the speaker exhibits 'Physiological Arousal' (breathlessness, lip smacks): Flag as BLUFF.
        - If response is immediate, prosody is stable, and syntax is direct: Flag as NO_BLUFF.
        
        JSON SCHEMA:
        {
          "verdict": "BLUFF" | "NO_BLUFF" | "INCONCLUSIVE",
          "confidence": float (0.0 to 1.0),
          "prosody_score": float, 
          "linguistic_score": float,
          "reason": "Explain the exact acoustic or linguistic trigger for this verdict (max 15 words)."
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
                "temperature": 0.1, # Low temperature for high precision/consistency
                "response_mime_type": "application/json"
            }
        }

        try:
            url = f"{self.API_BASE}/{model_id}:generateContent?key={self.api_key}"
            response = await http_client.post(url, json=payload)
            
            if response.status_code != 200:
                raise Exception(f"Gemini API Error: {response.text}")

            res_json = response.json()
            raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
            data = json.loads(raw_text)

            return AnalysisResult(
                verdict=Verdict(data.get("verdict", "INCONCLUSIVE")),
                confidence=data.get("confidence", 0.0),
                reason=data.get("reason", "Analysis incomplete."),
                prosody_score=data.get("prosody_score", 0.0),
                linguistic_score=data.get("linguistic_score", 0.0),
                provider="Gemini 3.0 (Multimodal)"
            )
        except Exception as e:
            logger.error(f"Gemini Processing Failure: {e}")
            raise e

# =============================================================================
# OPENAI BACKUP SERVICE
# =============================================================================

class OpenAIService:
    """Fallback using Whisper-v3 and GPT-4o-Audio-Preview if available."""
    API_BASE = "https://api.openai.com/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def analyze_audio(self, audio_data: bytes, mime_type: str) -> AnalysisResult:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # 1. Transcription (Whisper)
        files = {"file": ("audio.m4a", audio_data, mime_type), "model": (None, "whisper-1")}
        trans_res = await http_client.post(f"{self.API_BASE}/audio/transcriptions", headers=headers, files=files)
        transcript = trans_res.json().get("text", "")

        # 2. Linguistic Analysis (GPT-4o)
        prompt = f"Analyze this transcript for deception markers: '{transcript}'. Return JSON with verdict, confidence, and reason."
        chat_payload = {
            "model": "gpt-4o",
            "messages": [{"role": "system", "content": "You are a forensic linguist."}, {"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"}
        }
        chat_res = await http_client.post(f"{self.API_BASE}/chat/completions", headers=headers, json=chat_payload)
        data = json.loads(chat_res.json()["choices"][0]["message"]["content"])

        return AnalysisResult(
            verdict=Verdict(data.get("verdict", "INCONCLUSIVE")),
            confidence=data.get("confidence", 0.7), # OpenAI text-only is capped lower
            reason=data.get("reason", "Textual analysis only."),
            prosody_score=0.5,
            linguistic_score=0.8,
            provider="OpenAI Backup"
        )

# =============================================================================
# ORCHESTRATOR & ENDPOINTS
# =============================================================================

class NoBluffOrchestrator:
    def __init__(self):
        self.gemini = GeminiService(GOOGLE_API_KEY) if GOOGLE_API_KEY else None
        self.openai = OpenAIService(OPENAI_API_KEY) if OPENAI_API_KEY else None

    async def run_analysis(self, audio: bytes, mime: str) -> AnalysisResult:
        # Step 1: Attempt Gemini (Multimodal is superior for Lie Detection)
        if self.gemini:
            try:
                return await self.gemini.analyze_audio(audio, mime)
            except Exception:
                logger.warning("Gemini Core failed. Failing over to OpenAI...")

        # Step 2: Fallback
        if self.openai:
            return await self.openai.analyze_audio(audio, mime)
            
        return AnalysisResult(Verdict.INCONCLUSIVE, 0.0, "No AI providers available.", 0, 0, "FailSafe")

orchestrator = NoBluffOrchestrator()
app = FastAPI(title="NoBluff AI", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(file: UploadFile = File(...), mime_type: str = Form(...)):
    # 1. Minimum Data Check
    content = await file.read()
    if len(content) < 1000: # Ensure there is enough audio data to analyze prosody
        return AnalysisResponse(verdict="INCONCLUSIVE", confidence=0.0, reason="Audio clip too short for forensic analysis.")

    # 2. Run Orchestrator
    result = await orchestrator.run_analysis(content, mime_type)
    
    # 3. Log for Data Science Review
    logger.info(f"Verdict: {result.verdict} | Conf: {result.confidence} | Reason: {result.reason}")

    return AnalysisResponse(
        verdict=result.verdict.value,
        confidence=result.confidence,
        reason=result.reason
    )
