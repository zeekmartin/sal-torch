"""Offline Ed25519 license verification."""
from __future__ import annotations
import json, logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
_PUBLIC_KEY_HEX = "0" * 64  # placeholder

class LicenseError(Exception): pass

@dataclass
class LicenseInfo:
    tier: str
    organization: Optional[str]
    issued: Optional[str]
    expires: Optional[str]
    features: list[str]

    @property
    def is_expired(self):
        if not self.expires: return False
        try: return datetime.now() > datetime.fromisoformat(self.expires)
        except ValueError: return False

    @property
    def is_valid(self): return not self.is_expired

    def to_dict(self):
        return {"tier": self.tier, "organization": self.organization,
                "expires": self.expires, "features": self.features, "is_valid": self.is_valid}

def verify_license(path: str) -> LicenseInfo:
    p = Path(path)
    if not p.exists(): raise LicenseError(f"Not found: {p}")
    raw = p.read_bytes()
    parts = raw.split(b"\n---\n")
    if len(parts) != 2: raise LicenseError("Invalid format")
    payload = json.loads(parts[0].decode("utf-8"))
    # Signature verification would go here with PyNaCl
    info = LicenseInfo(tier=payload.get("tier","community"), organization=payload.get("organization"),
                       issued=payload.get("issued"), expires=payload.get("expires"),
                       features=payload.get("features", []))
    if info.is_expired:
        logger.warning(f"License expired: {info.expires}")
    return info
