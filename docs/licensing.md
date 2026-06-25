# Licensing

`sal-torch` is released under the Business Source License 1.1 (BSL 1.1). All
features are present in every install — there is no feature gating, no
phone-home, and no telemetry. The tier you need depends on **how you use it**.

## Tiers

| Tier | Use | Price |
|---|---|---|
| **Community** | Research, evaluation, and other non-commercial use. Full features. | Free |
| **Professional** | Commercial / production use. | ~$10–15K / year |
| **Enterprise** | Production use plus compliance reports, robustness certification, and priority support. | Contact us |

The enforcement is deliberately light: the real terms are legal (BSL 1.1). A
license file exists mainly for compliance reporting and organizational tracking.

## Setting a license

Professional and Enterprise users receive a license file. Point the library at
it in either of two ways.

Environment variable (picked up automatically at import):

```bash
export SAL_LICENSE_FILE=/path/to/your.lic
```

Or in code, once at startup:

```python
import sal
sal.set_license("/path/to/your.lic")

print(sal.license_info())
# {'tier': 'professional', 'organization': 'Acme Corp', 'expires': '2027-06-25', ...}
```

With no license file, `sal.license_info()` reports Community mode:

```python
import sal
sal.license_info()
# {'tier': 'community', 'organization': None, 'features': [...],
#  'note': 'Community — full features, non-commercial use only.'}
```

Never commit license files to version control — `*.lic` is in the project
`.gitignore` for this reason.

## Getting a license

For a Professional or Enterprise license, or to ask which tier fits your use,
contact **[contact@cognitive-engineering.dev](mailto:contact@cognitive-engineering.dev)**.

Built by [Cognitive Engineering](https://cognitive-engineering.dev) in Switzerland.
