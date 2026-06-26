"""License signing/verification tests (CPU, no network).

These exercise the shipped verification path in ``sal.license`` directly with
PyNaCl — mirroring the file format produced by tools/generate_license.py — so
they stay self-contained and do not depend on the (gitignored) tooling script.
"""
import base64
import json
import logging

import pytest

nacl_signing = pytest.importorskip("nacl.signing")
from nacl.signing import SigningKey

import sal
import sal.license as lic

_SEPARATOR = b"\n---\n"


def _make_license(path, signing_key, *, tier="professional", org="Acme Corp",
                  email="cto@acme.com", expires="2099-01-01", license_id="SAL-PRO-2099-0001",
                  features=("sal", "fi", "scanner", "plasticity", "compare", "report"),
                  notes="", tamper=False):
    payload = {
        "tier": tier, "organization": org, "contact_email": email,
        "issued": "2026-01-01", "expires": expires, "license_id": license_id,
        "features": list(features), "notes": notes,
    }
    payload_bytes = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    signature = signing_key.sign(payload_bytes).signature
    if tamper:
        payload = dict(payload, tier="enterprise")  # change content, keep old signature
        payload_bytes = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(payload_bytes + _SEPARATOR + base64.b64encode(signature))
    return path


@pytest.fixture
def keypair():
    sk = SigningKey.generate()
    return sk, sk.verify_key.encode().hex()


@pytest.fixture
def embed_key(keypair, monkeypatch):
    """Embed the test public key into the package for the duration of a test."""
    _, pub_hex = keypair
    monkeypatch.setattr(lic, "_PUBLIC_KEY_HEX", pub_hex)
    return keypair


def test_full_cycle(embed_key, tmp_path):
    sk, _ = embed_key
    path = _make_license(tmp_path / "valid.lic", sk)
    info = lic.verify_license(str(path))
    assert info.tier == "professional"
    assert info.organization == "Acme Corp"
    assert info.contact_email == "cto@acme.com"
    assert info.license_id == "SAL-PRO-2099-0001"
    assert "plasticity" in info.features
    assert info.is_valid


def test_expired_license_warns(embed_key, tmp_path, caplog):
    sk, _ = embed_key
    path = _make_license(tmp_path / "expired.lic", sk, expires="2020-01-01")
    with caplog.at_level(logging.WARNING, logger="sal.license"):
        info = lic.verify_license(str(path))
    assert info.is_expired
    assert any("expired" in r.message.lower() for r in caplog.records)


def test_tampered_license_rejected(embed_key, tmp_path):
    sk, _ = embed_key
    path = _make_license(tmp_path / "tampered.lic", sk, tamper=True)
    with pytest.raises(lic.LicenseError, match="signature"):
        lic.verify_license(str(path))


def test_wrong_key_rejected(embed_key, tmp_path):
    sk, _ = embed_key
    other = SigningKey.generate()  # not the embedded key
    path = _make_license(tmp_path / "wrongkey.lic", other)
    with pytest.raises(lic.LicenseError, match="signature"):
        lic.verify_license(str(path))


def test_no_embedded_key_rejected(keypair, tmp_path, monkeypatch):
    sk, _ = keypair
    monkeypatch.setattr(lic, "_PUBLIC_KEY_HEX", "0" * 64)  # placeholder build
    path = _make_license(tmp_path / "nokey.lic", sk)
    with pytest.raises(lic.LicenseError, match="public key"):
        lic.verify_license(str(path))


def test_set_license_updates_info(embed_key, tmp_path, monkeypatch):
    sk, _ = embed_key
    path = _make_license(tmp_path / "set.lic", sk)
    monkeypatch.setattr(sal, "_LICENSE_INFO", None)  # restored after test
    sal.set_license(str(path))
    out = sal.license_info()
    assert out["tier"] == "professional"
    assert out["organization"] == "Acme Corp"
    assert out["is_valid"]
