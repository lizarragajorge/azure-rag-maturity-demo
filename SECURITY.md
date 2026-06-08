# Security policy

## Scope

This repository is a **demonstration** of the post-Build-2026 RAG pattern on
Azure AI Search + Azure OpenAI. It is intended as a teaching reference, not as
a production-hardened template. Do not deploy it to handle real customer data
without first applying the hardening notes below.

## Reporting a vulnerability

If you find a security issue:

- For issues in **this repo's code or infrastructure**, please open a
  [GitHub Security Advisory](https://github.com/jlizarraga/azure-rag-maturity-demo/security/advisories/new)
  rather than a public issue. We aim to acknowledge reports within 5 business
  days.
- For issues in the underlying **Azure services** (AI Search, Azure OpenAI,
  Container Apps), report through the
  [Microsoft Security Response Center](https://msrc.microsoft.com/).

Do **not** include reproduction details, screenshots, or sample payloads in a
public issue. The advisory channel is private until we coordinate disclosure.

## What this demo is — and isn't — defended against

| Threat | Mitigation in this repo | Notes |
|---|---|---|
| Hard-coded credentials | None used. All Azure calls go through `DefaultAzureCredential` → user-assigned managed identity in the deployed Container App. | `disableLocalAuth: true` on both AI Search and Azure OpenAI. |
| AOAI cost DoS (someone burning your token budget) | Per-session cap of 10 agentic queries (Tab 3 only — the other tabs don't call AOAI). `max_completion_tokens=600` bounds the worst-case cost per call. Cheap model (`gpt-5-mini` GlobalStandard). | Session cap is client-side and bypassable with incognito tabs or scripts. If you see abuse, flip the `ingressAllowedIpRanges` Bicep param to restrict to known IPs, or enable Container Apps Easy Auth. |
| Cross-site request forgery | Streamlit's XSRF protection is enabled (default). | Don't pass `--server.enableXsrfProtection=false`. |
| Direct prompt injection in user questions | System prompt scopes the LLM to "answer ONLY from the JSON grounding". The user question is delimited with explicit `<user_question>` XML tags so "ignore previous instructions" attacks have a harder time. | Not a perfect defense — for production, layer Azure AI Content Safety prompt-shields on top. |
| Indirect prompt injection from the corpus | Corpus is hand-curated synthetic markdown. | For real RAG apps that index user-submitted or third-party content, add content moderation on ingest and on retrieval. |
| Harmful content generation | Azure OpenAI content filters are on by default for Standard / GlobalStandard deployments. | Don't lower the defaults. |
| Image vulnerabilities | Base image is `python:3.12-slim`; container runs as a non-root user. | Pin the base image SHA and rebuild periodically for long-lived deployments. |
| Container Registry credential leak | `adminUserEnabled: false`. App pulls images via UAMI + AcrPull role. | No registry passwords exist to leak. |
| Search service credential leak | `disableLocalAuth: true`. App queries via UAMI + Search Index Data Reader. | No admin keys to leak. |
| Inbound to Search / AOAI | Both have `publicNetworkAccess: Enabled` for demo simplicity. | For production: switch to Private Endpoints and remove the public network rule. |
| Secrets in the public repo | `.gitignore` excludes `.env`, `.azure/`, `.streamlit/secrets.toml`. GitHub secret scanning is enabled automatically on public repos. | Never put real endpoints, principal IDs, or tokens in committed files. `.env.example` contains placeholders only. |

## Hardening checklist if you fork this for production use

- [ ] Remove the per-session client-side cap and put Azure API Management
      (with token-limit policy) in front of the Container App
- [ ] Switch the Container App ingress to `external: false` + put it behind
      Application Gateway / Front Door with WAF
- [ ] Enable Microsoft Entra Easy Auth on the Container App
      (`identityProviders.azureActiveDirectory`)
- [ ] Move AI Search and Azure OpenAI to Private Endpoints; set
      `publicNetworkAccess: Disabled` on both
- [ ] Enable Microsoft Defender for Cloud on the subscription
- [ ] Turn on diagnostic settings (Search + AOAI + Container App) → Log Analytics
- [ ] Add Azure AI Content Safety **prompt shields** for direct + indirect
      prompt injection
- [ ] Replace `gpt-5-mini` with the smallest model that meets your quality bar
      and set a Cognitive Services capacity reservation (cost predictability)
- [ ] Pin the Docker base image to a specific SHA and rebuild on a cadence
- [ ] Add Dependabot or Renovate for `requirements.txt`
