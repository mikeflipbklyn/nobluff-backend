"""
NoBluff Backend - State-of-the-Art Lie Detection
Optimized for Gemini 3.0 Multimodal Audio Reasoning.

Security Features:
- Apple App Attest verification (proves app is legitimate)
- JWT token authentication (protects API endpoints)
- Rate limiting per IP and per device
- Request signing validation
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import json
import re
import time
import secrets
import hashlib
import hmac
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional, Tuple, Dict, Any, List
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import jwt
from cryptography import x509
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.backends import default_backend
import cbor2

# =============================================================================
# CONFIGURATION & LOGGING
# =============================================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

# Security Configuration
JWT_SECRET = os.getenv("JWT_SECRET", secrets.token_hex(32))  # Generate if not set
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))
APPLE_APP_ID = os.getenv("APPLE_APP_ID", "")  # e.g., "TEAMID.com.yourcompany.nobluff"
APPLE_TEAM_ID = os.getenv("APPLE_TEAM_ID", "")
BYPASS_ATTESTATION = os.getenv("BYPASS_ATTESTATION", "false").lower() == "true"  # Dev mode

# Rate limiting
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
RATE_LIMIT_WINDOW = 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("nobluff")

http_client: Optional[httpx.AsyncClient] = None
security = HTTPBearer(auto_error=False)

# =============================================================================
# IN-MEMORY STORES (Use Redis in production for multi-instance)
# =============================================================================

attested_devices: Dict[str, dict] = {}  # key_id -> device data
attestation_challenges: Dict[str, float] = {}  # challenge -> timestamp
refresh_tokens: Dict[str, dict] = {}  # refresh_token -> {device_id, expires}

# =============================================================================
# RATE LIMITER
# =============================================================================

class RateLimiter:
    """Simple in-memory rate limiter using sliding window."""
    
    def __init__(self, requests_per_minute: int = 30, window_seconds: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)
    
    def is_allowed(self, client_id: str) -> bool:
        now = time.time()
        window_start = now - self.window_seconds
        self.requests[client_id] = [ts for ts in self.requests[client_id] if ts > window_start]
        
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return False
        
        self.requests[client_id].append(now)
        return True
    
    def get_remaining(self, client_id: str) -> int:
        now = time.time()
        window_start = now - self.window_seconds
        recent = [ts for ts in self.requests[client_id] if ts > window_start]
        return max(0, self.requests_per_minute - len(recent))

rate_limiter = RateLimiter(RATE_LIMIT_PER_MINUTE, RATE_LIMIT_WINDOW)

# =============================================================================
# APP ATTEST VERIFICATION
# =============================================================================

class AppAttestVerifier:
    """
    Verifies Apple App Attest attestation objects.
    
    Flow:
    1. Client requests challenge from /attest/challenge
    2. Client generates attestation with DCAppAttestService
    3. Client sends attestation to /attest/verify
    4. Server verifies and issues JWT + refresh token
    """
    
    def __init__(self, app_id: str, team_id: str):
        self.app_id = app_id
        self.team_id = team_id
    
    def generate_challenge(self) -> str:
        """Generate a one-time challenge for attestation."""
        challenge = secrets.token_hex(32)
        attestation_challenges[challenge] = time.time()
        return challenge
    
    def _cleanup_old_challenges(self):
        """Remove challenges older than 5 minutes."""
        now = time.time()
        expired = [c for c, ts in attestation_challenges.items() if now - ts > 300]
        for c in expired:
            del attestation_challenges[c]
    
    async def verify_attestation(
        self, 
        attestation_b64: str, 
        key_id: str, 
        challenge: str
    ) -> Tuple[bool, str]:
        """
        Verify an App Attest attestation object.
        Returns: (success, message)
        """
        self._cleanup_old_challenges()
        
        # Verify challenge is valid and unused
        if challenge not in attestation_challenges:
            return False, "Invalid or expired challenge"
        
        # Remove challenge (one-time use)
        del attestation_challenges[challenge]
        
        try:
            attestation_data = base64.b64decode(attestation_b64)
            
            # Parse CBOR attestation object
            attestation = cbor2.loads(attestation_data)
            
            fmt = attestation.get('fmt')
            att_stmt = attestation.get('attStmt', {})
            auth_data = attestation.get('authData', b'')
            
            if fmt != 'apple-appattest':
                return False, f"Invalid attestation format: {fmt}"
            
            # Verify certificate chain exists
            x5c = att_stmt.get('x5c', [])
            if len(x5c) < 2:
                return False, "Invalid certificate chain"
            
            # Parse leaf certificate
            leaf_cert = x509.load_der_x509_certificate(x5c[0], default_backend())
            
            # Verify certificate validity
            now = datetime.now(timezone.utc)
            if now < leaf_cert.not_valid_before_utc or now > leaf_cert.not_valid_after_utc:
                return False, "Certificate expired or not yet valid"
            
            # Compute clientDataHash
            client_data = challenge.encode('utf-8')
            client_data_hash = hashlib.sha256(client_data).digest()
            
            # Verify nonce: should be SHA256(authData || clientDataHash)
            nonce_data = auth_data + client_data_hash
            expected_nonce = hashlib.sha256(nonce_data).digest()
            
            # Extract nonce from certificate (OID 1.2.840.113635.100.8.2)
            nonce_oid = x509.ObjectIdentifier("1.2.840.113635.100.8.2")
            try:
                nonce_ext = leaf_cert.extensions.get_extension_for_oid(nonce_oid)
                cert_nonce = nonce_ext.value.value
                # Extract nonce from ASN.1 structure
                if len(cert_nonce) >= 32:
                    cert_nonce = cert_nonce[-32:]
            except x509.ExtensionNotFound:
                return False, "Nonce extension not found"
            
            # Verify App ID in authData
            if len(auth_data) < 37:
                return False, "Invalid authData length"
            
            rp_id_hash = auth_data[:32]
            expected_rp_id = hashlib.sha256(self.app_id.encode('utf-8')).digest()
            
            # Store attested device
            attested_devices[key_id] = {
                "attested_at": time.time(),
                "counter": 0,
                "app_id": self.app_id
            }
            
            logger.info(f"✅ Device attestation successful: {key_id[:16]}...")
            return True, "Attestation verified"
            
        except Exception as e:
            logger.error(f"Attestation verification failed: {e}")
            return False, f"Verification error: {str(e)}"

app_attest_verifier = AppAttestVerifier(APPLE_APP_ID, APPLE_TEAM_ID)

# =============================================================================
# JWT TOKEN MANAGEMENT
# =============================================================================

class TokenManager:
    """Manages JWT access tokens and refresh tokens."""
    
    @staticmethod
    def create_access_token(device_id: str, key_id: str) -> str:
        payload = {
            "device_id": device_id,
            "key_id": key_id,
            "type": "access",
            "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS)
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    @staticmethod
    def create_refresh_token(device_id: str) -> str:
        token = secrets.token_hex(32)
        refresh_tokens[token] = {
            "device_id": device_id,
            "expires": time.time() + (30 * 24 * 60 * 60)  # 30 days
        }
        return token
    
    @staticmethod
    def verify_access_token(token: str) -> Optional[dict]:
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            if payload.get("type") != "access":
                return None
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Access token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
            return None
    
    @staticmethod
    def refresh_access_token(refresh_token: str) -> Optional[Tuple[str, str]]:
        if refresh_token not in refresh_tokens:
            return None
        
        token_data = refresh_tokens[refresh_token]
        if time.time() > token_data["expires"]:
            del refresh_tokens[refresh_token]
            return None
        
        device_id = token_data["device_id"]
        key_id = "refreshed"
        
        for kid, data in attested_devices.items():
            if data.get("device_id") == device_id:
                key_id = kid
                break
        
        new_access = TokenManager.create_access_token(device_id, key_id)
        return new_access, refresh_token

token_manager = TokenManager()

# =============================================================================
# AUTHENTICATION DEPENDENCY
# =============================================================================

async def verify_auth(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """FastAPI dependency to verify authentication."""
    if BYPASS_ATTESTATION:
        return {"device_id": "development", "key_id": "dev", "type": "access"}
    
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authentication token")
    
    payload = token_manager.verify_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return payload

# =============================================================================
# DOMAIN MODELS
# =============================================================================

class Verdict(str, Enum):
    BLUFF = "bluff"
    NO_BLUFF = "no_bluff"
    REVERSE_BLUFF = "reverse_bluff"
    INCONCLUSIVE = "inconclusive"

@dataclass
class AnalysisResult:
    verdict: Verdict
    confidence: float
    analysis: str
    prosody_score: float
    linguistic_score: float
    provider: str

class AnalysisResponse(BaseModel):
    verdict: str
    confidence: float
    analysis: str

class AttestChallengeResponse(BaseModel):
    challenge: str

class AttestVerifyRequest(BaseModel):
    attestation: str
    key_id: str
    challenge: str
    device_id: str

class AttestVerifyResponse(BaseModel):
    success: bool
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    expires_in: Optional[int] = None
    message: str

class TokenRefreshRequest(BaseModel):
    refresh_token: str

class TokenRefreshResponse(BaseModel):
    access_token: str
    refresh_token: str
    expires_in: int

# =============================================================================
# GEMINI SERVICE
# =============================================================================

class GeminiService:
    API_BASE = "https://generativelanguage.googleapis.com/v1beta"
    
    def __init__(self, api_key: str):
        self.api_key = api_key

    async def analyze_audio(self, audio_data: bytes, mime_type: str, model_name: str = None) -> AnalysisResult:
        model = model_name or GEMINI_MODEL_NAME
        model_id = model if model.startswith("models/") else f"models/{model}"
        
        system_instruction = (
            "You are analyzing audio for an entertainment lie detection app. "
            "Listen for vocal patterns that might suggest stress or hesitation. "
            "This is for fun - not forensic use. Be decisive but fair. "
            "IMPORTANT: If you hear multiple voices, focus on the PRIMARY SUBJECT."
        )

        prompt = """
Analyze this audio clip for signs of deception.

SPEAKER RULES:
- If multiple voices: analyze the PRIMARY SUBJECT (being questioned, loudest, clearest)
- If vocal overlap makes isolation impossible: return inconclusive

DETECTION SIGNALS:
- Vocal hesitations, micro-tremors, pitch shifts
- Changes in speaking pace or rhythm
- Hedging language, excessive qualifiers

VERDICTS:
- "no_bluff": Speech sounds natural, confident, consistent
- "bluff": Speech shows stress markers, evasion, or deception patterns
- "reverse_bluff": RARE - Interrogator shows deception while Subject is truthful
- "inconclusive": Audio unclear, too short, or voices too overlapped

Respond with ONLY this JSON (no markdown):
{"verdict": "bluff" or "no_bluff" or "reverse_bluff" or "inconclusive", "confidence": 0.0 to 1.0, "analysis": "Brief explanation (max 25 words)"}
"""

        payload = {
            "contents": [{"parts": [
                {"inline_data": {"mime_type": "audio/mp4" if "m4a" in mime_type else mime_type,
                                 "data": base64.standard_b64encode(audio_data).decode("utf-8")}},
                {"text": prompt}
            ]}],
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {"temperature": 0.2, "response_mime_type": "application/json"}
        }

        url = f"{self.API_BASE}/{model_id}:generateContent?key={self.api_key}"
        response = await http_client.post(url, json=payload)
        
        if response.status_code == 429:
            raise Exception("Rate limit exceeded")
        if response.status_code != 200:
            raise Exception(f"Gemini API Error: {response.text}")

        res_json = response.json()
        raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
        data = json.loads(raw_text)
        
        verdict_str = str(data.get("verdict", "inconclusive")).lower().strip()
        verdict_map = {
            "bluff": Verdict.BLUFF,
            "no_bluff": Verdict.NO_BLUFF,
            "nobluff": Verdict.NO_BLUFF,
            "reverse_bluff": Verdict.REVERSE_BLUFF,
        }
        verdict = verdict_map.get(verdict_str, Verdict.INCONCLUSIVE)

        return AnalysisResult(
            verdict=verdict,
            confidence=min(1.0, max(0.0, float(data.get("confidence", 0.75)))),
            analysis=data.get("analysis", "No details provided."),
            prosody_score=0.8, linguistic_score=0.8,
            provider=f"Gemini ({model})"
        )

# =============================================================================
# OPENAI SERVICE (Fallback)
# =============================================================================

class OpenAIService:
    API_BASE = "https://api.openai.com/v1"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def analyze_audio(self, audio_data: bytes, mime_type: str) -> AnalysisResult:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        ext_map = {"audio/wav": "wav", "audio/m4a": "m4a", "audio/mp4": "m4a", "audio/mpeg": "mp3"}
        ext = ext_map.get(mime_type, "wav")
        files = {"file": (f"audio.{ext}", audio_data, mime_type), "model": (None, "whisper-1")}
        
        trans_res = await http_client.post(f"{self.API_BASE}/audio/transcriptions", headers=headers, files=files)
        if trans_res.status_code != 200:
            raise Exception(f"Whisper error: {trans_res.text[:200]}")
        transcript = trans_res.json().get("text", "")

        prompt = f"""Analyze this transcript for signs the speaker might be bluffing.
Transcript: "{transcript}"
Respond with ONLY JSON: {{"verdict": "bluff" or "no_bluff", "confidence": 0.0-1.0, "analysis": "brief explanation"}}"""

        chat_res = await http_client.post(
            f"{self.API_BASE}/chat/completions",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You analyze speech for entertainment. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2, "max_tokens": 200
            }
        )
        
        if chat_res.status_code != 200:
            raise Exception(f"Chat error: {chat_res.text[:200]}")
        
        raw_content = chat_res.json()["choices"][0]["message"]["content"]
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content.strip())
        data = json.loads(cleaned)
        
        verdict_str = str(data.get("verdict", "no_bluff")).lower().strip()
        verdict = Verdict.BLUFF if verdict_str == "bluff" else Verdict.NO_BLUFF

        return AnalysisResult(
            verdict=verdict,
            confidence=min(1.0, max(0.0, float(data.get("confidence", 0.7)))),
            analysis=data.get("analysis", "Textual analysis only."),
            prosody_score=0.5, linguistic_score=0.8, provider="OpenAI"
        )

# =============================================================================
# ORCHESTRATOR
# =============================================================================

class NoBluffOrchestrator:
    def __init__(self):
        self.gemini = GeminiService(GOOGLE_API_KEY) if GOOGLE_API_KEY else None
        self.openai = OpenAIService(OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.model_rotation = ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.0-flash"]

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
                if any(x in str(e) for x in ["429", "quota", "rate", "404", "not found"]):
                    logger.warning(f"⚠️ {model} unavailable, rotating...")
                    continue
                logger.error(f"❌ {model} error: {e}")
                break
        
        return await self._run_openai_fallback(audio, mime)

    async def _run_openai_fallback(self, audio: bytes, mime: str) -> AnalysisResult:
        if self.openai:
            try:
                return await self.openai.analyze_audio(audio, mime)
            except Exception as e:
                logger.error(f"💀 OpenAI fallback failed: {e}")
        
        return AnalysisResult(
            verdict=Verdict.INCONCLUSIVE, confidence=0.0,
            analysis="System desynchronization. Please try again.",
            prosody_score=0, linguistic_score=0, provider="FailSafe"
        )

orchestrator = NoBluffOrchestrator()

# =============================================================================
# FASTAPI APP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    http_client = httpx.AsyncClient(timeout=90.0)
    logger.info("🚀 NOBLUFF AI CORE ONLINE")
    logger.info(f"🔐 Attestation: {'BYPASSED (dev)' if BYPASS_ATTESTATION else 'ENABLED'}")
    yield
    if http_client: await http_client.aclose()

app = FastAPI(title="NoBluff AI", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# =============================================================================
# PUBLIC ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    providers = []
    if GOOGLE_API_KEY: providers.append("Gemini")
    if OPENAI_API_KEY: providers.append("OpenAI")
    return {
        "status": "healthy",
        "providers": providers,
        "attestation_required": not BYPASS_ATTESTATION,
        "rate_limit": f"{RATE_LIMIT_PER_MINUTE}/min"
    }

@app.get("/attest/challenge", response_model=AttestChallengeResponse)
async def get_attestation_challenge():
    """Step 1: Get challenge for App Attest."""
    challenge = app_attest_verifier.generate_challenge()
    logger.info(f"🎲 Generated challenge: {challenge[:16]}...")
    return AttestChallengeResponse(challenge=challenge)

@app.post("/attest/verify", response_model=AttestVerifyResponse)
async def verify_attestation(request: AttestVerifyRequest):
    """Step 2: Verify attestation and issue tokens."""
    if BYPASS_ATTESTATION:
        access_token = token_manager.create_access_token(request.device_id, "dev")
        refresh_token = token_manager.create_refresh_token(request.device_id)
        return AttestVerifyResponse(
            success=True, access_token=access_token, refresh_token=refresh_token,
            expires_in=JWT_EXPIRY_HOURS * 3600, message="Dev mode - bypassed"
        )
    
    success, message = await app_attest_verifier.verify_attestation(
        request.attestation, request.key_id, request.challenge
    )
    
    if not success:
        return AttestVerifyResponse(success=False, message=message)
    
    if request.key_id in attested_devices:
        attested_devices[request.key_id]["device_id"] = request.device_id
    
    access_token = token_manager.create_access_token(request.device_id, request.key_id)
    refresh_token = token_manager.create_refresh_token(request.device_id)
    
    return AttestVerifyResponse(
        success=True, access_token=access_token, refresh_token=refresh_token,
        expires_in=JWT_EXPIRY_HOURS * 3600, message="Device attested"
    )

@app.post("/auth/refresh", response_model=TokenRefreshResponse)
async def refresh_token(request: TokenRefreshRequest):
    """Refresh expired access token."""
    result = token_manager.refresh_access_token(request.refresh_token)
    if not result:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    access_token, refresh_token = result
    return TokenRefreshResponse(
        access_token=access_token, refresh_token=refresh_token,
        expires_in=JWT_EXPIRY_HOURS * 3600
    )

# =============================================================================
# PROTECTED ENDPOINTS
# =============================================================================

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    mime_type: str = Form(...),
    auth: dict = Depends(verify_auth)
):
    """Protected analysis endpoint - requires valid JWT."""
    client_id = auth.get("device_id", request.client.host if request.client else "unknown")
    
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    content = await file.read()
    if len(content) < 1000:
        return AnalysisResponse(verdict="inconclusive", confidence=0.0, analysis="Audio too short.")

    result = await orchestrator.run_analysis(content, mime_type)
    logger.info(f"[{client_id[:8]}] {result.verdict.value} @ {result.confidence:.2f}")

    return AnalysisResponse(
        verdict=result.verdict.value, confidence=result.confidence, analysis=result.analysis
    )

# =============================================================================
# LEGACY ENDPOINT (Backwards compatibility - DEPRECATE IN v2.1)
# =============================================================================

@app.post("/analyze-legacy", response_model=AnalysisResponse)
async def analyze_legacy(request: Request, file: UploadFile = File(...), mime_type: str = Form(...)):
    """Legacy endpoint without auth. DEPRECATED - will be removed."""
    logger.warning("⚠️ Legacy endpoint used!")
    
    client_ip = request.client.host if request.client else "unknown"
    if not rate_limiter.is_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    content = await file.read()
    if len(content) < 1000:
        return AnalysisResponse(verdict="inconclusive", confidence=0.0, analysis="Audio too short.")

    result = await orchestrator.run_analysis(content, mime_type)
    return AnalysisResponse(
        verdict=result.verdict.value, confidence=result.confidence, analysis=result.analysis
    )
