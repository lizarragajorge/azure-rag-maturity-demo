---
id: doc-014
title: "OT/IT Cybersecurity Incident Response — SCADA Environment"
document_type: "Security Procedure"
keywords: ["cybersecurity", "incident response", "SCADA", "OT", "IT", "CIP-008", "ransomware", "isolation"]
page: 1
source: "NGU-SEC-IR-002 Rev. 7"
last_updated: "2026-05-03"
---

# OT/IT Cybersecurity Incident Response — SCADA Environment

This procedure governs the response to cybersecurity events in the
Operational Technology (OT) environment, including the Energy Management
System (EMS), the SCADA front-end processors, the Distribution Management
System (DMS), and the gas SCADA. Aligned with **NERC CIP-008-6** and
**TSA Pipeline Security Directive 2021-02C**.

## Classification

The Security Operations Center (SOC) classifies each event within **15
minutes** of detection:

- **Reportable Cyber Security Incident** (CIP-008): an incident that has
  compromised, or attempted to compromise, the BES Cyber System or its
  Electronic Security Perimeter.
- **Cyber Security Incident** (not reportable): unusual but contained,
  affecting only non-BES assets or thwarted at the perimeter.
- **Operational anomaly**: unexplained behavior of OT systems with no evidence
  of malicious cause; investigated under the engineering ticket queue, not the
  incident-response process.

## Reporting timelines

| Recipient                                | Reportable Cyber Security Incident |
|------------------------------------------|------------------------------------|
| E-ISAC                                   | Within **1 hour** of determination |
| CISA                                     | Within **24 hours**                |
| TSA (gas pipeline)                       | Within **24 hours**                |
| Reliability Coordinator (SPP, WECC)      | Within **1 hour**                  |
| State PUC                                | Per state-specific filing rules    |

## Containment — OT-specific cautions

The **first rule of OT incident response** is do not lose visibility or
control of the physical process. Standard IT containment actions (isolate the
asset from the network) can blind operators to the state of the grid.
Therefore:

1. The shift dispatch supervisor — not the SOC analyst — has final authority
   over any containment action that affects an EMS or SCADA component.
2. Before isolating a SCADA host, confirm that an alternate path of visibility
   (redundant front-end, ICCP from neighbor, manual field reporting) is
   established.
3. Field crews are dispatched to **manually monitor** any breakers or valves
   that lose telemetry, until the situation is resolved.

## Evidence preservation

The SOC preserves evidence per CIP-008 R3:

- Full packet capture of the affected segment for the last **72 hours**
  (rolling buffer always available).
- Memory dump of affected hosts before reimaging.
- Log retention of **6 years** for IR-related logs.

## Tabletop exercises

A full-scope OT tabletop is conducted **annually** with the Reliability
Coordinator and a no-notice partial exercise is conducted **semi-annually**.
After-action reports and remediation tracking are retained for the current
audit cycle plus 3 years.
