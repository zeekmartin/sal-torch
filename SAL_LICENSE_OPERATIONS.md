# sal-torch License Management — Internal Operations Guide

**Cognitive Engineering — INTERNAL / CONFIDENTIAL**
**Last updated: June 2026**

---

## 1. Overview

sal-torch uses offline Ed25519 license keys. No server, no phone-home, no telemetry. A license is a signed JSON file (`.lic`) that the customer places on their machine. The package verifies the signature at `import sal` using an embedded public key.

This document covers everything needed to create, deliver, renew, and manage licenses.

---

## 2. Architecture

```
Cognitive Engineering (private)          Customer (their infra)
┌─────────────────────────┐              ┌──────────────────────┐
│ Ed25519 PRIVATE key     │              │ sal-torch package    │
│ (sal_private.pem)       │──signs──→    │ contains PUBLIC key  │
│                         │              │ (sal_public.pem)     │
│ tools/generate_license  │              │                      │
│ → produces .lic file    │──delivers──→ │ customer.lic         │
│                         │              │ verified at import   │
└─────────────────────────┘              └──────────────────────┘
```

The private key NEVER leaves Cognitive Engineering infrastructure.
The public key is embedded in the shipped package (sal/_keys/sal_public.pem).

---

## 3. Initial Setup (One-Time)

### 3.1 Generate the Master Keypair

Run this ONCE. Store the private key securely. Never regenerate unless compromised.

```bash
cd H:\sal-torch
python tools/generate_license.py --generate-keypair
```

This creates:
- `keys/sal_private.pem` — PRIVATE. Never commit. Never share. Back up securely.
- `keys/sal_public.pem` — PUBLIC. Copy to `sal/_keys/sal_public.pem` in the package.

### 3.2 Secure the Private Key

The private key is the crown jewel. If compromised, anyone can forge licenses.

Storage requirements:
- Primary: encrypted USB drive or hardware security key, stored physically at Cognitive Engineering
- Backup: second encrypted copy in a different physical location
- NEVER on GitHub, NEVER in cloud storage unencrypted, NEVER on a shared drive
- NEVER in the sal-torch repository (add `keys/` and `*.pem` to .gitignore)

### 3.3 Embed the Public Key

After generating the keypair:

```bash
cp keys/sal_public.pem sal/_keys/sal_public.pem
```

Update `sal/license.py` — replace the placeholder `_PUBLIC_KEY_HEX` with the actual hex-encoded public key:

```python
_PUBLIC_KEY_HEX = "<actual 64-character hex string from sal_public.pem>"
```

Commit and release. This is the only time the public key changes (unless the private key is compromised and you need to rotate).

---

## 4. License File Format

A `.lic` file contains two parts separated by `\n---\n`:

```
<JSON payload (UTF-8)>
---
<Ed25519 signature (base64)>
```

### 4.1 Payload Fields

```json
{
    "tier": "professional",
    "organization": "Acme Corp",
    "contact_email": "cto@acme.com",
    "issued": "2026-07-01",
    "expires": "2027-07-01",
    "license_id": "SAL-PRO-2026-0001",
    "features": ["sal", "fi", "scanner", "plasticity", "compare", "report"],
    "max_models": null,
    "notes": "Annual license — 1 year from issue date"
}
```

| Field          | Required | Description                                         |
|----------------|----------|-----------------------------------------------------|
| tier           | Yes      | "professional" or "enterprise"                      |
| organization   | Yes      | Legal entity name                                   |
| contact_email  | Yes      | Primary contact for renewals                        |
| issued         | Yes      | ISO date (YYYY-MM-DD)                               |
| expires        | Yes      | ISO date — license invalid after this date          |
| license_id     | Yes      | Unique ID: SAL-{TIER}-{YEAR}-{SEQ}                  |
| features       | Yes      | List of enabled features (all for now)              |
| max_models     | No       | null = unlimited. Future: limit for lower tiers     |
| notes          | No       | Internal notes (visible to customer in license_info)|

### 4.2 License ID Convention

```
SAL-PRO-2026-0001    → Professional, 2026, first customer
SAL-ENT-2026-0001    → Enterprise, 2026, first enterprise customer
SAL-PRO-2026-0002    → Professional, 2026, second customer
SAL-TRIAL-2026-0001  → Trial license (30 days, if needed later)
```

---

## 5. License Tiers

### 5.1 Community (no license file needed)

- Price: Free
- Usage: Research, evaluation, education, prototyping
- Restriction: No commercial production deployment (enforced by BSL terms, not technically)
- Duration: Unlimited
- Support: Community only (GitHub Issues)
- License file: None — `sal.license_info()` returns `{"tier": "community"}`

### 5.2 Professional — $10,000-$15,000/year

- Price: $12,000/year (standard), negotiable $10K-$15K based on size
- Usage: Commercial production deployment
- Features: All current features
- Duration: 12 months from issue date
- Support: Email, 48h SLA response
- Updates: All versions released during the license period
- License file: `.lic` with tier="professional"

Target customer: startups and mid-size companies fine-tuning models for production (fintech, pharma, insurance).

### 5.3 Enterprise — $30,000-$50,000/year

- Price: $35,000/year (standard), negotiable $30K-$50K
- Usage: Commercial production deployment + regulatory compliance
- Features: All current features + compliance reports (PDF export)
- Duration: 12 months from issue date
- Support: Priority email, 24h SLA response, quarterly review call
- Updates: All versions + pre-releases
- License file: `.lic` with tier="enterprise"

Target customer: regulated industries (banks, pharma, defense, aerospace), AI labs training large models.

### 5.4 Audit (separate, optional add-on)

- Price: $5,000-$10,000 per model (one-time)
- Deliverable: Structural analysis report (FI + PlasticityMap + recommendations)
- Not a license — it's a consulting engagement
- Available to Professional and Enterprise customers
- Often serves as the entry point: audit first, license follows

---

## 6. Creating a License

### 6.1 Command

```bash
cd H:\sal-torch

python tools/generate_license.py \
    --private-key keys/sal_private.pem \
    --tier professional \
    --org "Acme Corp" \
    --email "cto@acme.com" \
    --duration 365 \
    --output licenses/SAL-PRO-2026-0001.lic
```

Parameters:
- `--private-key`: path to the private key
- `--tier`: professional | enterprise
- `--org`: organization name (must match contract)
- `--email`: contact email
- `--duration`: days from today (365 = 1 year)
- `--license-id`: auto-generated if omitted (SAL-{TIER}-{YEAR}-{SEQ})
- `--output`: output path for the .lic file
- `--notes`: optional notes

### 6.2 Verification

Always verify a license after creating it:

```bash
python tools/generate_license.py --verify licenses/SAL-PRO-2026-0001.lic
```

Expected output:
```
License SAL-PRO-2026-0001 verified ✓
  Tier:         professional
  Organization: Acme Corp
  Issued:       2026-07-01
  Expires:      2027-07-01
  Features:     sal, fi, scanner, plasticity, compare, report
  Status:       VALID (361 days remaining)
```

### 6.3 Test with the Package

```python
import sal
sal.set_license("path/to/SAL-PRO-2026-0001.lic")
print(sal.license_info())
# → {"tier": "professional", "organization": "Acme Corp", ...}
```

---

## 7. Delivering a License

### 7.1 Delivery Process

1. **Contract signed** — customer signs license agreement (separate document)
2. **Payment received** — wire transfer or invoice payment confirmed
3. **Generate license** — using `tools/generate_license.py`
4. **Verify license** — always verify before sending
5. **Send to customer** — encrypted email to the contact_email
6. **Confirm installation** — customer confirms `sal.license_info()` works
7. **Log in registry** — record in the license registry (see §9)

### 7.2 Delivery Email Template

```
Subject: sal-torch License — [ORGANIZATION] — [LICENSE_ID]

Dear [CONTACT],

Attached is your sal-torch [TIER] license file.

To activate:
  Option A: Set the environment variable
    export SAL_LICENSE_FILE=/path/to/[LICENSE_ID].lic

  Option B: Set in code
    import sal
    sal.set_license("/path/to/[LICENSE_ID].lic")

  Option C: Verify
    python -c "import sal; sal.set_license('[LICENSE_ID].lic'); print(sal.license_info())"

License details:
  ID:       [LICENSE_ID]
  Tier:     [TIER]
  Issued:   [ISSUED_DATE]
  Expires:  [EXPIRY_DATE]

This license is tied to [ORGANIZATION] and is non-transferable.
Please store the file securely — it contains your organization's
license credentials.

Support: contact@cognitive-engineering.dev
Docs: https://docs.cognitive-engineering.dev/sal/licensing

Best regards,
David Martin Venti
Cognitive Engineering
```

### 7.3 File Security

- Send the `.lic` file as an encrypted email attachment (PGP or S/MIME) if possible
- Alternative: share via a secure file transfer (not plain email attachment)
- Never post license files in GitHub Issues, Slack, or any public channel
- Each customer gets a unique license — never reuse license files

---

## 8. Renewal Process

### 8.1 Timeline

- **60 days before expiry**: send renewal reminder email
- **30 days before expiry**: second reminder with renewal invoice
- **7 days before expiry**: final reminder
- **Expiry day**: license expires, package logs a warning at import
- **Grace period**: none — expired means expired. Generate a new license upon payment.

### 8.2 Renewal Command

Same as creation, with a new expiry date and incremented license ID:

```bash
python tools/generate_license.py \
    --private-key keys/sal_private.pem \
    --tier professional \
    --org "Acme Corp" \
    --email "cto@acme.com" \
    --duration 365 \
    --license-id SAL-PRO-2027-0001 \
    --output licenses/SAL-PRO-2027-0001.lic
```

### 8.3 Upgrade (Professional → Enterprise)

Generate a new Enterprise license for the remaining duration:

```bash
python tools/generate_license.py \
    --private-key keys/sal_private.pem \
    --tier enterprise \
    --org "Acme Corp" \
    --email "cto@acme.com" \
    --expires 2027-07-01 \
    --license-id SAL-ENT-2026-0001 \
    --output licenses/SAL-ENT-2026-0001.lic
```

The customer replaces their `.lic` file. The old Professional license still works until it expires but the Enterprise one supersedes it.

---

## 9. License Registry

Maintain a private spreadsheet or database (NOT in the repo):

```
| License ID        | Org        | Tier         | Contact          | Issued     | Expires    | Amount    | Status  |
|-------------------|------------|--------------|------------------|------------|------------|-----------|---------|
| SAL-PRO-2026-0001 | Acme Corp  | Professional | cto@acme.com     | 2026-07-01 | 2027-07-01 | CHF 12000 | ACTIVE  |
| SAL-ENT-2026-0001 | SwissBank  | Enterprise   | ai@swissbank.ch  | 2026-08-15 | 2027-08-15 | CHF 35000 | ACTIVE  |
| SAL-PRO-2026-0002 | PharmaCo   | Professional | ml@pharmaco.com  | 2026-09-01 | 2027-09-01 | CHF 10000 | PENDING |
```

Store in:
- Primary: encrypted local spreadsheet (H:\cognitive-engineering\licenses\registry.xlsx)
- Backup: encrypted cloud backup
- NEVER in the sal-torch repo

---

## 10. Revocation

### 10.1 Limitation

Offline Ed25519 licenses CANNOT be revoked remotely. Once delivered, the customer can use the license until it expires. This is by design — no phone-home means no revocation.

### 10.2 Mitigation

- Short durations (12 months max) limit exposure
- If a customer violates terms, do not renew
- In extreme cases (fraud, redistribution), pursue legal remedies under BSL
- Future option: if enough customers justify it, add optional online validation as an opt-in (never mandatory — segment B won't accept it)

### 10.3 Key Rotation

If the private key is compromised:
1. Generate a new keypair immediately
2. Release a new version of sal-torch with the new public key
3. Re-issue all active customer licenses with the new private key
4. Notify all active customers to update both the package and their license file
5. The old public key no longer validates — old forged licenses won't work on new versions

---

## 11. Implementation Checklist

### Before first customer:

- [ ] Generate master Ed25519 keypair
- [ ] Store private key securely (encrypted USB, backed up)
- [ ] Embed public key in sal/_keys/sal_public.pem
- [ ] Update _PUBLIC_KEY_HEX in sal/license.py
- [ ] Create tools/generate_license.py (keypair generation + license creation + verification)
- [ ] Test full cycle: generate → verify → set_license → license_info
- [ ] Create license registry spreadsheet
- [ ] Prepare delivery email template
- [ ] Set up renewal calendar reminders
- [ ] Release updated sal-torch version with real public key

### For each new customer:

- [ ] Contract signed
- [ ] Payment received
- [ ] Generate license with tools/generate_license.py
- [ ] Verify license
- [ ] Deliver via secure channel
- [ ] Confirm customer installation works
- [ ] Add to registry
- [ ] Set renewal reminder (expiry - 60 days)

---

## 12. tools/generate_license.py Specification

This tool needs to be created. Give Claude Code this spec:

```
Create tools/generate_license.py with the following commands:

1. --generate-keypair
   Generate Ed25519 keypair, save to keys/sal_private.pem and keys/sal_public.pem.
   Print the public key hex for embedding in sal/license.py.

2. License creation (default mode):
   --private-key PATH    (required)
   --tier TIER           professional|enterprise
   --org NAME            organization name
   --email EMAIL         contact email
   --duration DAYS       days from today (default: 365)
   --expires DATE        explicit expiry (alternative to --duration)
   --license-id ID       auto-generated if omitted
   --features LIST       comma-separated (default: all)
   --notes TEXT           optional notes
   --output PATH         output .lic file path

3. --verify PATH
   Verify a .lic file against the embedded public key.
   Print license details and validity status.

Dependencies: PyNaCl (nacl.signing)
This tool is NEVER shipped with the package. It stays in tools/ and is .gitignored.
```

---

## Appendix: BSL 1.1 Summary for License Discussions

When talking to customers about licensing:

- **Community**: "Free for research, evaluation, prototyping. You can publish papers with it, run experiments, build POCs. The only restriction is commercial production deployment."

- **Professional**: "Same code, same features. The license authorizes production commercial use. You get 12 months of updates and email support."

- **Enterprise**: "Everything in Professional, plus compliance reports you can attach to regulatory filings, priority support, and quarterly review calls."

- **BSL conversion**: "On June 25, 2030, the entire codebase converts to Apache 2.0 automatically. Any version released before that date becomes fully open source on the conversion date."

The BSL is not a trick — it's a standard license used by MariaDB, CockroachDB, Sentry, and others. It protects the business while keeping the code transparent and auditable, which is exactly what regulated industries need.
