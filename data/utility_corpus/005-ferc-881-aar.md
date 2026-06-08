---
id: doc-005
title: "FERC Order 881 Compliance — Ambient-Adjusted Ratings (AAR) Implementation"
document_type: "Regulatory Compliance"
keywords: ["FERC", "Order 881", "AAR", "ambient adjusted ratings", "transmission", "dynamic line ratings", "SLR"]
page: 1
source: "NGU-REG-FERC881-002 Rev. 2"
last_updated: "2026-05-20"
---

# FERC Order 881 — AAR Implementation Compliance Note

FERC Order No. 881 requires transmission providers to use **Ambient-Adjusted
Ratings (AAR)** for near-term transmission line ratings, replacing the legacy
practice of seasonal static line ratings (Seasonal Static Ratings, or SLR), no
later than **July 12, 2025**. This note summarizes the utility's implementation
posture and operational changes in effect on the transmission footprint.

## What AAR changes

Every transmission line is now rated at **hourly** intervals based on the
forecasted ambient temperature at representative weather stations along the
line. The rating is recomputed at minimum once per hour for the next 240 hours
and submitted to the SPP (Southwest Power Pool) and WECC reliability
coordinator real-time tools.

## Inputs

- Temperature forecast: HRRR (3-km, hourly) for hours 0–48, then NBM (12-km,
  hourly) for hours 49–240.
- Conductor parameters: from the conductor type, diameter, emissivity, and
  absorptivity database (NGU-DBA-COND-118).
- Wind assumption: **2 ft/s perpendicular**, conservative.
- Solar assumption: clear-sky direct normal irradiance for the date and latitude.

## What AAR is *not*

AAR is not the same as **Dynamic Line Ratings (DLR)**, which use real-time
measurement of conductor sag, temperature, or tension. FERC has proposed DLR in
a follow-on rulemaking; the utility is participating in the SPP DLR pilot on
the south-corridor 230 kV line but has not yet committed to a system-wide
deployment.

## Operational consequences

- Line ratings change throughout the day; dispatchers must reference the
  current-hour rating rather than a printed seasonal value.
- Congestion patterns shift: lines previously rated 1,000 A in summer may now
  rate at 920 A during a 35°C afternoon and 1,150 A during a 5°C night.
- Outage scheduling tools (POW system) consume AAR forecasts when evaluating
  the reliability impact of a planned outage.

## Audit trail

All hourly ratings, the input forecasts, and the conductor parameter snapshots
are retained for **five years** per Order 881 §35.28(g) and made available to
FERC enforcement staff on request.
