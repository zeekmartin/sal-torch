"""Offline Ed25519 license verification."""
from __future__ import annotations
import base64, json, logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Embedded Ed25519 public key (hex). Set with tools/generate_license.py --embed-key.
# Kept as a single 64-char literal so --embed-key can replace it cleanly.
_PLACEHOLDER = "0000000000000000000000000000000000000000000000000000000000000000"
_PUBLIC_KEY_HEX = "0000000000000000000000000000000000000000000000000000000000000000"
_SEPARATOR = b"\n---\n"

class LicenseError(Exception): pass

@dataclass
class LicenseInfo:
    tier: str
    organization: Optional[str]
    issued: Optional[str]
    expires: Optional[str]
    features: list[str]
    license_id: Optional[str] = None
    contact_email: Optional[str] = None
    notes: Optional[str] = None

    @property
    def is_expired(self):
        if not self.expires: return False
        try: return datetime.now() > datetime.fromisoformat(self.expires)
        except ValueError: return False

    @property
    def is_valid(self): return not self.is_expired

    def to_dict(self):
        return {"tier": self.tier, "organization": self.organization,
                "license_id": self.license_id, "contact_email": self.contact_email,
                "issued": self.issued, "expires": self.expires,
                "features": self.features, "notes": self.notes,
                "is_valid": self.is_valid}


def _verify_signature(payload_bytes: bytes, signature_b64: bytes):
    """Verify the Ed25519 signature over ``payload_bytes``. Raises LicenseError
    if the package has no embedded key, PyNaCl is missing, the encoding is bad,
    or the signature does not match."""
    if _PUBLIC_KEY_HEX == _PLACEHOLDER:
        raise LicenseError("No public key embedded in this build; cannot verify license.")
    try:
        from nacl.exceptions import BadSignatureError
        from nacl.signing import VerifyKey
    except ImportError as e:
        raise LicenseError("License verification needs PyNaCl. "
                           "Install with: pip install sal-torch[crypto]") from e
    try:
        signature = base64.b64decode(signature_b64, validate=True)
    except Exception as e:
        raise LicenseError("Malformed license signature encoding.") from e
    try:
        VerifyKey(bytes.fromhex(_PUBLIC_KEY_HEX)).verify(payload_bytes, signature)
    except BadSignatureError as e:
        raise LicenseError("License signature verification failed "
                           "(tampered file or wrong key).") from e


def verify_license(path: str) -> LicenseInfo:
    p = Path(path)
    if not p.exists(): raise LicenseError(f"Not found: {p}")
    raw = p.read_bytes()
    parts = raw.split(_SEPARATOR, 1)
    if len(parts) != 2: raise LicenseError("Invalid format")
    payload_bytes, signature_b64 = parts[0], parts[1].strip()

    _verify_signature(payload_bytes, signature_b64)

    payload = json.loads(payload_bytes.decode("utf-8"))
    info = LicenseInfo(tier=payload.get("tier", "community"),
                       organization=payload.get("organization"),
                       issued=payload.get("issued"), expires=payload.get("expires"),
                       features=payload.get("features", []),
                       license_id=payload.get("license_id"),
                       contact_email=payload.get("contact_email"),
                       notes=payload.get("notes"))
    if info.is_expired:
        logger.warning(f"License expired: {info.expires}")
    return info
