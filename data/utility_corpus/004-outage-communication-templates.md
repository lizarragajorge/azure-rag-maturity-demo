---
id: doc-004
title: "Storm Outage Customer Communication — Standard Templates"
document_type: "Communications Playbook"
keywords: ["outage", "storm", "communication", "ETR", "estimated time of restoration", "social media", "press release"]
page: 1
source: "NGU-COMM-OUT-009 Rev. 6"
last_updated: "2026-02-19"
---

# Storm Outage Customer Communication — Standard Templates

When a named storm or a multi-county outage event exceeds **2,500 customers
affected**, the Emergency Operations Center activates this communications
playbook. All external messaging is approved by the on-call corporate
communications officer before publication.

## The four-message cadence

| Phase             | Trigger                                  | Audience              | Lead time |
|-------------------|------------------------------------------|-----------------------|-----------|
| Pre-event alert   | Storm forecast 24–48h out                | Customer self-service | T-24h     |
| Event activation  | First sustained outage > 250 customers   | All affected accounts | < 15 min  |
| ETR update        | Every 4 hours, or when ETR changes ±25%  | All affected accounts | Rolling   |
| Restoration close | Last customer restored                   | All affected accounts | < 30 min  |

## Estimated Time of Restoration (ETR) discipline

The Outage Management System (OMS) generates a system-calculated ETR within 30
minutes of event activation. Dispatch may **override** the system ETR upward
when:

- Field damage assessment reveals broken poles, slip-on conductor, or transmission
  damage that the system did not infer from breaker trips alone.
- Mutual-assistance crews have not yet arrived at staging.
- Hazardous terrain (active wildfire zone, flooding, ice-covered ridgeline)
  prevents safe access.

Do **not** revise an ETR downward by more than two hours without confirmation
from the field general foreman. Customers tolerate a late restoration far better
than a missed restoration promise.

## Plain-language template (SMS, 160 chars)

`Utility Outage Alert: We are aware of an outage at your service address. Crews are en route. Current ETR: {{etr_local}}. Updates: utility.example/outages or text STATUS.`

## Plain-language template (voice, 25 seconds)

> This is your electric utility with an outage update for the service at
> {{premise_short_address}}. Our crews have identified the cause and current
> estimated restoration is {{etr_local}}. For real-time status, visit
> utility.example/outages or reply STATUS to this number. We appreciate your
> patience.

## What we never say

- "Power will be back on shortly" — too vague, sets unmeasurable expectations.
- Estimates of crew counts or equipment damage without dispatch verification.
- Comparisons to other utilities' restoration performance.
- Speculation about root cause until the engineering review is complete.
