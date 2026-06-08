---
id: doc-006
title: "NERC CIP-013-2 Supply Chain Risk Management — Procurement Controls"
document_type: "Regulatory Compliance"
keywords: ["NERC", "CIP-013", "supply chain", "vendor", "BES Cyber System", "procurement", "SBOM"]
page: 1
source: "NGU-REG-CIP013-007 Rev. 5"
last_updated: "2026-03-30"
---

# NERC CIP-013-2 — Supply Chain Procurement Controls

CIP-013-2 requires a documented supply-chain cyber security risk management
plan for **high- and medium-impact BES Cyber Systems**. This document specifies
the procurement controls that procurement, engineering, and IT/OT security
apply on every covered acquisition.

## Six required controls (per CIP-013-2 R1.2)

1. **Notification of vendor security events** — every contract for covered
   hardware, software, or services must obligate the vendor to notify within 24
   hours of an incident that could affect the procured product.
2. **Coordination of incident response** — defined points of contact and a
   tabletop exercise within 12 months of contract execution.
3. **Verification of software integrity and authenticity** — checksum or
   signature verification required before installation; SBOM (Software Bill of
   Materials) in **CycloneDX or SPDX** format delivered with each release.
4. **Coordination of vendor-initiated remote access controls** — vendors who
   require remote access must use the company's privileged-access workstation;
   no direct VPN credentials are issued.
5. **Disclosure of known vulnerabilities** — vendor must disclose any
   unpatched CVE with CVSS ≥ 7.0 affecting the product at time of delivery.
6. **Termination of remote access** — within 4 hours of contract termination
   or employee turnover at the vendor.

## Procurement checklist

Procurement may not issue a purchase order for any covered product without:

- A completed Vendor Cyber Risk Questionnaire (NGU-FORM-VCR-04)
- Evidence of SOC 2 Type II or ISO 27001 certification, or a documented
  exception approved by the CISO
- Contract language incorporating the six controls above
- An entry in the BES Cyber Asset register identifying the receiving location

## Re-assessment cadence

Every covered vendor is re-assessed at contract renewal or every **36 months**,
whichever is sooner. Tier-1 vendors (those whose product directly controls or
monitors a high-impact BES Cyber System) are re-assessed annually.

## Audit evidence

Procurement records, completed questionnaires, SBOMs, and incident-coordination
test evidence are retained for the **current audit cycle plus 6 years** in the
GRC platform and provided to the NERC CIP audit team during the standard
14-month spot-check window.
