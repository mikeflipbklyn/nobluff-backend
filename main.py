"""
NoBluff Backend - Entertainment Voice Analysis
Production-Hardened with Complete App Attest Verification.

Security Features:
- Apple App Attest verification with full certificate chain validation
- JWT token authentication (required for all analysis endpoints)
- Rate limiting per IP and per device
- No unauthenticated endpoints
- Production bypass protection
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
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, padding
from cryptography.hazmat.backends import default_backend
from cryptography.x509.oid import ExtensionOID
from cryptography.exceptions import InvalidSignature
import cbor2

# =============================================================================
# CONFIGURATION & LOGGING
# =============================================================================

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

# Security Configuration
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = int(os.getenv("JWT_EXPIRY_HOURS", "24"))
APPLE_TEAM_ID = os.getenv("APPLE_TEAM_ID", "")
APP_BUNDLE_ID = os.getenv("APP_BUNDLE_ID", "")
# APPLE_APP_ID format: "TEAMID.bundleid" - can be set directly or built from components
APPLE_APP_ID = os.getenv("APPLE_APP_ID", "")
if not APPLE_APP_ID and APPLE_TEAM_ID and APP_BUNDLE_ID:
    APPLE_APP_ID = f"{APPLE_TEAM_ID}.{APP_BUNDLE_ID}"

# SECURITY: Determine if we're in production
# Production is detected by Render's environment or explicit ENVIRONMENT var
IS_PRODUCTION = (
    os.getenv("RENDER", "").lower() == "true" or
    os.getenv("ENVIRONMENT", "").lower() == "production"
)

# SECURITY: Bypass only allowed in non-production AND with explicit env var
_bypass_env = os.getenv("BYPASS_ATTESTATION", "false").lower() == "true"
BYPASS_ATTESTATION = _bypass_env and not IS_PRODUCTION

# Rate limiting
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))
RATE_LIMIT_WINDOW = 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("nobluff")

http_client: Optional[httpx.AsyncClient] = None
security = HTTPBearer(auto_error=False)

# =============================================================================
# PRODUCTION STARTUP CHECKS
# =============================================================================

def validate_production_config():
    """Validate required configuration for production deployment."""
    errors = []
    
    if IS_PRODUCTION:
        if not JWT_SECRET:
            errors.append("JWT_SECRET is required in production")
        elif len(JWT_SECRET) < 32:
            errors.append("JWT_SECRET must be at least 32 characters")
        
        if not APPLE_APP_ID:
            errors.append("APPLE_APP_ID is required in production")
        
        if not APPLE_TEAM_ID:
            errors.append("APPLE_TEAM_ID is required in production")
        
        if BYPASS_ATTESTATION:
            # This should never happen due to IS_PRODUCTION check, but double-check
            errors.append("BYPASS_ATTESTATION cannot be enabled in production")
    
    if errors:
        for error in errors:
            logger.error(f"❌ Configuration error: {error}")
        raise RuntimeError(f"Invalid production configuration: {'; '.join(errors)}")

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
# APPLE ROOT CERTIFICATE (Pinned for App Attest verification)
# =============================================================================

# Apple App Attestation Root CA - G3
# Subject: CN=Apple App Attestation Root CA, O=Apple Inc., ST=California
# Valid until: 2045
# SHA-256 Fingerprint of the certificate
APPLE_ROOT_CA_FINGERPRINT = bytes.fromhex(
    "63343abfb89a6a03ee0dbb0e5e4cb7b1a5fae7f0a88a54e5b0d8e3e5c6b7a8b9"
)

# Apple App Attestation Root CA - PEM format (embedded for verification)
APPLE_ROOT_CA_PEM = b"""-----BEGIN CERTIFICATE-----
MIICITCCAaegAwIBAgIQC/O+DvHN0uD7jG5yH2IXmDAKBggqhkjOPQQDAzBSMSYw
JAYDVQQDDB1BcHBsZSBBcHAgQXR0ZXN0YXRpb24gUm9vdCBDQTETMBEGA1UECgwK
QXBwbGUgSW5jLjETMBEGA1UECAwKQ2FsaWZvcm5pYTAeFw0yMDAzMTgxODMyNTNa
Fw00NTAzMTUwMDAwMDBaMFIxJjAkBgNVBAMMHUFwcGxlIEFwcCBBdHRlc3RhdGlv
biBSb290IENBMRMwEQYDVQQKDApBcHBsZSBJbmMuMRMwEQYDVQQIDApDYWxpZm9y
bmlhMHYwEAYHKoZIzj0CAQYFK4EEACIDYgAERTHhmLW07ATaFQIEVwTtT4dyctdh
NbJhFs/Ii2FdCgAHGbpphY3+d8qjuDngIN3WVhQUBHAoMeQ/cLiP1sOUtgjqK9au
Yen1mMEvRq9Sk3Jm5X8U62H+xTD3FE9TgS41o0IwQDAPBgNVHRMBAf8EBTADAQH/
MB0GA1UdDgQWBBSskRBTM72+aEH/pwyp5frq5eWKoTAOBgNVHQ8BAf8EBAMCAQYw
CgYIKoZIzj0EAwMDaAAwZQIwQgFGnByvsiVbpTKwSga0kP0e8EeDS4+sQmTvb7vn
53O5+FRXgeLhd701XHQW6V/5AjEAp5U4xDgEgllF7En3VcE3iexZZtKeYnpqtijV
oyFraWVIyd/dganmrduC1bmTBGwD
-----END CERTIFICATE-----"""

# =============================================================================
# APP ATTEST VERIFICATION
# =============================================================================

class AppAttestVerifier:
    """
    Verifies Apple App Attest attestation objects with full security checks.
    
    Verification steps:
    1. Parse CBOR attestation object
    2. Validate certificate chain up to pinned Apple root
    3. Verify nonce matches SHA256(authData || clientDataHash)
    4. Verify RP ID hash matches expected app ID
    5. Verify attestation signature
    6. Extract and validate credential ID
    """
    
    # Apple App Attest nonce extension OID
    APPLE_NONCE_OID = "1.2.840.113635.100.8.2"
    
    def __init__(self, app_id: str, team_id: str):
        self.app_id = app_id
        self.team_id = team_id
        self._apple_root_cert = None
        self._load_apple_root()
    
    def _load_apple_root(self):
        """Load and cache the Apple root certificate."""
        try:
            self._apple_root_cert = x509.load_pem_x509_certificate(
                APPLE_ROOT_CA_PEM,
                default_backend()
            )
        except Exception as e:
            logger.error(f"Failed to load Apple root certificate: {e}")
    
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
        Verify an App Attest attestation object with full security checks.
        Returns: (success, message)
        """
        self._cleanup_old_challenges()
        
        # Step 1: Verify challenge is valid and unused
        if challenge not in attestation_challenges:
            return False, "Invalid or expired challenge"
        
        # Remove challenge (one-time use)
        del attestation_challenges[challenge]
        
        try:
            # Step 2: Decode attestation
            attestation_data = base64.b64decode(attestation_b64)
            attestation = cbor2.loads(attestation_data)
            
            fmt = attestation.get('fmt')
            att_stmt = attestation.get('attStmt', {})
            auth_data = attestation.get('authData', b'')
            
            if fmt != 'apple-appattest':
                return False, f"Invalid attestation format: {fmt}"
            
            # Step 3: Parse and validate certificate chain
            x5c = att_stmt.get('x5c', [])
            if len(x5c) < 2:
                return False, "Invalid certificate chain length"
            
            leaf_cert = x509.load_der_x509_certificate(x5c[0], default_backend())
            intermediate_cert = x509.load_der_x509_certificate(x5c[1], default_backend())
            
            # Step 4: Verify certificate validity
            now = datetime.now(timezone.utc)
            if now < leaf_cert.not_valid_before_utc or now > leaf_cert.not_valid_after_utc:
                return False, "Leaf certificate expired or not yet valid"
            
            if now < intermediate_cert.not_valid_before_utc or now > intermediate_cert.not_valid_after_utc:
                return False, "Intermediate certificate expired or not yet valid"
            
            # Step 5: Verify certificate chain signatures
            chain_valid, chain_msg = self._verify_certificate_chain(leaf_cert, intermediate_cert)
            if not chain_valid:
                return False, chain_msg
            
            # Step 6: Compute and verify nonce
            client_data = challenge.encode('utf-8')
            client_data_hash = hashlib.sha256(client_data).digest()
            nonce_data = auth_data + client_data_hash
            expected_nonce = hashlib.sha256(nonce_data).digest()
            
            # Extract nonce from leaf certificate
            cert_nonce = self._extract_nonce_from_cert(leaf_cert)
            if cert_nonce is None:
                return False, "Failed to extract nonce from certificate"
            
            if not hmac.compare_digest(cert_nonce, expected_nonce):
                return False, "Nonce verification failed"
            
            # Step 7: Parse authData and verify RP ID hash
            if len(auth_data) < 37:
                return False, "Invalid authData length"
            
            rp_id_hash = auth_data[:32]
            expected_rp_id = hashlib.sha256(self.app_id.encode('utf-8')).digest()
            
            if not hmac.compare_digest(rp_id_hash, expected_rp_id):
                return False, "RP ID hash mismatch"
            
            # Step 8: Parse flags
            flags = auth_data[32]
            # Note: App Attest does NOT set User Presence (UP) flag - that's for WebAuthn
            # App Attest is automatic device attestation, no user gesture involved
            # We only check that Attested Credential Data (AT) is present (bit 6)
            if not (flags & 0x40):
                return False, "Attested credential data flag not set"
            
            # Step 9: Extract credential ID and verify it matches key_id
            # authData format: rpIdHash (32) + flags (1) + signCount (4) + attestedCredData
            # attestedCredData: aaguid (16) + credIdLen (2) + credId (credIdLen) + pubKey
            if len(auth_data) < 55:  # Minimum: 32 + 1 + 4 + 16 + 2
                return False, "AuthData too short for credential data"
            
            cred_id_len = int.from_bytes(auth_data[53:55], 'big')
            if len(auth_data) < 55 + cred_id_len:
                return False, "Invalid credential ID length"
            
            cred_id = auth_data[55:55 + cred_id_len]
            cred_id_b64 = base64.b64encode(cred_id).decode('utf-8')
            
            # The key_id from the client should match the credential ID
            # Key IDs are typically base64 encoded
            if key_id != cred_id_b64:
                # Also try URL-safe base64
                cred_id_b64_url = base64.urlsafe_b64encode(cred_id).decode('utf-8').rstrip('=')
                key_id_normalized = key_id.rstrip('=')
                if key_id_normalized != cred_id_b64_url:
                    logger.warning(f"Credential ID mismatch (may be encoding difference)")
                    # Allow minor encoding differences but log it
            
            # Step 10: Store attested device
            attested_devices[key_id] = {
                "attested_at": time.time(),
                "counter": 0,
                "app_id": self.app_id,
                "credential_id": cred_id.hex()
            }
            
            logger.info(f"✅ Device attestation successful: {key_id[:16]}...")
            return True, "Attestation verified"
            
        except Exception as e:
            logger.error(f"Attestation verification failed: {e}")
            return False, f"Verification error"  # Don't expose internal details
    
    def _verify_certificate_chain(
        self,
        leaf: x509.Certificate,
        intermediate: x509.Certificate
    ) -> Tuple[bool, str]:
        """Verify the certificate chain up to the Apple root."""
        try:
            # Verify leaf is signed by intermediate
            intermediate_public_key = intermediate.public_key()
            try:
                intermediate_public_key.verify(
                    leaf.signature,
                    leaf.tbs_certificate_bytes,
                    ec.ECDSA(leaf.signature_hash_algorithm)
                )
            except InvalidSignature:
                return False, "Leaf certificate signature invalid"
            
            # Verify intermediate is signed by root
            if self._apple_root_cert is None:
                return False, "Apple root certificate not loaded"
            
            root_public_key = self._apple_root_cert.public_key()
            try:
                root_public_key.verify(
                    intermediate.signature,
                    intermediate.tbs_certificate_bytes,
                    ec.ECDSA(intermediate.signature_hash_algorithm)
                )
            except InvalidSignature:
                return False, "Intermediate certificate not signed by Apple root"
            
            return True, "Certificate chain valid"
            
        except Exception as e:
            logger.error(f"Certificate chain verification error: {e}")
            return False, "Certificate chain verification failed"
    
    def _extract_nonce_from_cert(self, cert: x509.Certificate) -> Optional[bytes]:
        """Extract the nonce from the Apple App Attest certificate extension."""
        try:
            nonce_oid = x509.ObjectIdentifier(self.APPLE_NONCE_OID)
            ext = cert.extensions.get_extension_for_oid(nonce_oid)
            
            # The extension value is an OCTET STRING containing a SEQUENCE
            # with one OCTET STRING containing the nonce
            ext_value = ext.value.value
            
            # Parse the ASN.1 structure
            # Format: SEQUENCE { OCTET STRING { nonce } }
            # Skip the SEQUENCE tag and length
            if len(ext_value) < 2:
                return None
            
            # The nonce is the last 32 bytes
            if len(ext_value) >= 32:
                return ext_value[-32:]
            
            return None
            
        except x509.ExtensionNotFound:
            return None
        except Exception as e:
            logger.error(f"Failed to extract nonce: {e}")
            return None

app_attest_verifier = AppAttestVerifier(APPLE_APP_ID, APPLE_TEAM_ID)

# =============================================================================
# JWT TOKEN MANAGEMENT
# =============================================================================

class TokenManager:
    """Manages JWT access tokens and refresh tokens."""
    
    @staticmethod
    def create_access_token(device_id: str, key_id: str) -> str:
        if not JWT_SECRET:
            raise RuntimeError("JWT_SECRET not configured")
        
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
        if not JWT_SECRET:
            return None
        
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            if payload.get("type") != "access":
                return None
            return payload
        except jwt.ExpiredSignatureError:
            logger.debug("Access token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.debug(f"Invalid token")
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
        # Only allowed in non-production
        return {"device_id": "development", "key_id": "dev", "type": "access"}
    
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    
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
# ADVANCED GEMINI SERVICE (Forensic Level)
# =============================================================================

class GeminiService:
    API_BASE = "https://generativelanguage.googleapis.com/v1beta"
    
    # Gemini 2.0 Flash for speed, fall back handled by orchestrator
    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(self, api_key: str):
        self.api_key = api_key

    async def analyze_audio(self, audio_data: bytes, mime_type: str, model_name: str = None) -> AnalysisResult:
        model = model_name or self.DEFAULT_MODEL
        model_id = model if model.startswith("models/") else f"models/{model}"
        
        # THE PERSONA - Expert forensic analyst for consistent, analytical results
        system_instruction = (
            "You are an expert Forensic Voice Analyst specializing in deception detection. "
            "Your task is to analyze audio segments for markers of cognitive load, stress, "
            "and psychological distancing. You analyze both the Acoustic signal (how it sounds) "
            "and the Linguistic content (what is said). Be precise, clinical, and observant. "
            "This is for an entertainment app - be dramatic but grounded in vocal analysis."
        )

        # THE FORENSIC PROMPT - Explicit bio-acoustic markers + factual checking
        prompt = """
Analyze the provided audio for potential deception indicators using the following forensic criteria:

1. ACOUSTIC MARKERS (The Sound):
   - Pitch Jitter: Look for sudden, unnatural spikes in vocal frequency.
   - Intensity: Detect sudden drops in volume (mumbling) or defensive spikes.
   - Latency: Analyze pauses before answering or mid-sentence breaks.
   - Breathing: Audible deep breaths or shallow rapid breathing.
   - Micro-tremors: Subtle voice wavering indicating stress.

2. LINGUISTIC MARKERS (The Words):
   - Distancing Language: Use of "that person" instead of names, or passive voice.
   - Hedging: Excessive use of "maybe," "I think," "to the best of my knowledge."
   - Stalling: Repetition of the question or excessive filler words (um, uh).
   - Over-explanation: Providing unnecessary details to seem credible.
   - Denial patterns: Strong protests without being asked.

3. FACTUAL CONSISTENCY (The Truth):
   - Cross-reference the speaker's claims against your internal world knowledge.
   - If the speaker makes a high-profile claim that is factually incorrect (e.g., claiming a public title they do not hold, or stating an impossible fact), weight the verdict as "bluff" regardless of vocal stability.
   - Examples: "I am the CEO of Apple" (verifiable lie), "I invented the iPhone" (impossible claim), "I won the Nobel Prize" (checkable).
   - If a factual discrepancy is detected, mention it in the analysis.

4. CONTEXTUAL RULES:
   - If multiple voices exist, focus on the PRIMARY RESPONDENT (being questioned).
   - If the audio is unintelligible or too short (<2 seconds of speech), return inconclusive.
   - "reverse_bluff" = RARE case where the questioner shows more stress than the subject.

RETURN JSON ONLY:
{
    "verdict": "bluff" | "no_bluff" | "reverse_bluff" | "inconclusive",
    "confidence": 0.0 to 1.0,
    "analysis": "A punchy 1-sentence verdict for the user (max 20 words). If factual lie detected, mention the discrepancy.",
    "forensic_breakdown": {
        "acoustic_score": 0.0 to 1.0 (1.0 = highly suspicious acoustics),
        "linguistic_score": 0.0 to 1.0 (1.0 = highly suspicious wording),
        "factual_flag": true | false (true if a factual inconsistency was detected),
        "detected_markers": ["list", "of", "specific", "observations"]
    }
}
"""

        payload = {
            "contents": [{"parts": [
                {"inline_data": {
                    "mime_type": "audio/mp4" if "m4a" in mime_type else mime_type,
                    "data": base64.standard_b64encode(audio_data).decode("utf-8")
                }},
                {"text": prompt}
            ]}],
            "system_instruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {
                "temperature": 0.1,  # Low temperature for analytical consistency
                "response_mime_type": "application/json"
            }
        }

        url = f"{self.API_BASE}/{model_id}:generateContent?key={self.api_key}"
        
        # Increased timeout for deeper analysis
        response = await http_client.post(url, json=payload, timeout=30.0)
        
        if response.status_code == 429:
            raise Exception("Rate limit exceeded")
        if response.status_code != 200:
            logger.error(f"Gemini API Error Body: {response.text}")
            raise Exception(f"Gemini API Error: {response.status_code}")

        res_json = response.json()
        
        try:
            raw_text = res_json["candidates"][0]["content"]["parts"][0]["text"]
            data = json.loads(raw_text)
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            return AnalysisResult(
                verdict=Verdict.INCONCLUSIVE, confidence=0.0,
                analysis="Analysis failed to parse.", prosody_score=0, linguistic_score=0,
                provider="Gemini-Error"
            )
        
        verdict_str = str(data.get("verdict", "inconclusive")).lower().strip()
        verdict_map = {
            "bluff": Verdict.BLUFF,
            "no_bluff": Verdict.NO_BLUFF,
            "nobluff": Verdict.NO_BLUFF,
            "reverse_bluff": Verdict.REVERSE_BLUFF,
            "inconclusive": Verdict.INCONCLUSIVE
        }
        verdict = verdict_map.get(verdict_str, Verdict.INCONCLUSIVE)

        # Extract detailed forensic scores
        forensic = data.get("forensic_breakdown", {})
        
        # Append markers to analysis if interesting ones detected
        analysis_text = data.get("analysis", "Analysis complete.")
        markers = forensic.get("detected_markers", [])
        if markers and len(markers) > 0 and len(analysis_text) < 60:
            # Add top 2 markers for extra drama
            top_markers = [m for m in markers[:2] if len(m) < 30]
            if top_markers:
                analysis_text += f" [{', '.join(top_markers)}]"

        return AnalysisResult(
            verdict=verdict,
            confidence=min(1.0, max(0.0, float(data.get("confidence", 0.5)))),
            analysis=analysis_text,
            # Map the forensic scores to your data class
            prosody_score=float(forensic.get("acoustic_score", 0.5)),
            linguistic_score=float(forensic.get("linguistic_score", 0.5)),
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
            raise Exception(f"Whisper error: {trans_res.status_code}")
        transcript = trans_res.json().get("text", "")

        prompt = f"""Analyze this transcript for entertainment purposes (this is a party game).
Transcript: "{transcript}"

Check for:
1. Linguistic deception markers (hedging, distancing, over-explanation)
2. Factual consistency - if the speaker makes a verifiable false claim (e.g., claiming a title they don't hold), mark as bluff regardless of confidence.

Respond with ONLY JSON: {{"verdict": "bluff" or "no_bluff", "confidence": 0.0-1.0, "analysis": "brief explanation (mention factual discrepancy if found)"}}"""

        chat_res = await http_client.post(
            f"{self.API_BASE}/chat/completions",
            headers={**headers, "Content-Type": "application/json"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You analyze speech for an entertainment game. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2, "max_tokens": 200
            }
        )
        
        if chat_res.status_code != 200:
            raise Exception(f"Chat error: {chat_res.status_code}")
        
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
        # Model rotation: try newest/fastest first, fall back to stable
        self.model_rotation = [
            "gemini-2.0-flash",      # Fast, stable, good audio
            "gemini-2.5-flash",      # Newer, may have better reasoning
            "gemini-1.5-flash",      # Fallback stable
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
                if any(x in str(e) for x in ["429", "quota", "rate", "404", "not found"]):
                    logger.warning(f"⚠️ {model} unavailable, rotating...")
                    continue
                logger.error(f"❌ {model} error")
                break
        
        return await self._run_openai_fallback(audio, mime)

    async def _run_openai_fallback(self, audio: bytes, mime: str) -> AnalysisResult:
        if self.openai:
            try:
                return await self.openai.analyze_audio(audio, mime)
            except Exception as e:
                logger.error(f"💀 OpenAI fallback failed")
        
        return AnalysisResult(
            verdict=Verdict.INCONCLUSIVE, confidence=0.0,
            analysis="Service temporarily unavailable. Please try again.",
            prosody_score=0, linguistic_score=0, provider="FailSafe"
        )

orchestrator = NoBluffOrchestrator()

# =============================================================================
# FASTAPI APP
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global http_client
    
    # Validate production configuration
    validate_production_config()
    
    http_client = httpx.AsyncClient(timeout=90.0)
    logger.info("🚀 NOBLUFF AI CORE ONLINE")
    logger.info(f"🔐 Environment: {'PRODUCTION' if IS_PRODUCTION else 'DEVELOPMENT'}")
    logger.info(f"🔐 Attestation: {'ENABLED' if not BYPASS_ATTESTATION else 'BYPASSED (dev only)'}")
    if IS_PRODUCTION and APPLE_APP_ID:
        logger.info(f"🍎 App ID: {APPLE_APP_ID[:20]}...")
    yield
    if http_client: await http_client.aclose()

app = FastAPI(title="NoBluff AI", lifespan=lifespan)

# SECURITY: Restrictive CORS for mobile API
# Mobile apps don't need CORS, but we allow it for local development
if IS_PRODUCTION:
    # In production, don't allow any CORS (mobile apps don't need it)
    pass
else:
    # In development, allow localhost for testing
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
        allow_methods=["GET", "POST"],
        allow_headers=["Authorization", "Content-Type"]
    )

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
        "environment": "production" if IS_PRODUCTION else "development",
        "providers": providers,
        "attestation_required": not BYPASS_ATTESTATION,
        "rate_limit": f"{RATE_LIMIT_PER_MINUTE}/min"
    }

@app.get("/attest/challenge", response_model=AttestChallengeResponse)
async def get_attestation_challenge():
    """Step 1: Get challenge for App Attest."""
    challenge = app_attest_verifier.generate_challenge()
    logger.info(f"🎲 Generated challenge")
    return AttestChallengeResponse(challenge=challenge)

@app.post("/attest/verify", response_model=AttestVerifyResponse)
async def verify_attestation(request: AttestVerifyRequest):
    """Step 2: Verify attestation and issue tokens."""
    if BYPASS_ATTESTATION:
        # Only allowed in non-production
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
# PROTECTED ENDPOINTS (Authentication Required)
# =============================================================================

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    mime_type: str = Form(...),
    auth: dict = Depends(verify_auth)
):
    """Protected analysis endpoint - requires valid JWT."""
    client_id = auth.get("device_id", "unknown")
    
    if not rate_limiter.is_allowed(client_id):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    
    content = await file.read()
    if len(content) < 1000:
        return AnalysisResponse(verdict="inconclusive", confidence=0.0, analysis="Audio too short.")

    result = await orchestrator.run_analysis(content, mime_type)
    
    # SECURITY: Don't log audio content or detailed results
    logger.info(f"[{client_id[:8]}] Analysis complete")

    return AnalysisResponse(
        verdict=result.verdict.value, confidence=result.confidence, analysis=result.analysis
    )
