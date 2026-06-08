---
id: doc-008
title: "AMI Smart Meter Installation — Field Procedure"
document_type: "Operations Procedure"
keywords: ["AMI", "smart meter", "installation", "meter", "field procedure", "Itron", "Landis+Gyr"]
page: 1
source: "NGU-OPS-AMI-022 Rev. 3"
last_updated: "2026-02-04"
---

# AMI Smart Meter Installation — Field Procedure

This procedure applies to the installation of single-phase Form 2S residential
AMI meters (Itron Centron II or Landis+Gyr E360) on existing socket-based
service points rated up to 320A continuous.

## Pre-installation

1. Verify the customer account is in active status in CIS and an AMI install
   work order is open.
2. Confirm the meter serial number on the truck inventory matches the work
   order. Mismatches require dispatch reassignment before installation.
3. Knock and announce yourself; if no one is home, leave the standard
   door-hanger and proceed only if the meter is accessible without entering
   fenced or locked areas.
4. Photograph the existing meter, the socket interior, and the customer-side
   service entrance from a distance of approximately 4 ft.

## Power-off / power-on

The procedure causes a service interruption of approximately **30–90 seconds**.

1. Apply blank insulating gloves rated 1000V and a Class 0 face shield.
2. Open the meter ring seal and document the seal number in the WMS.
3. Withdraw the existing meter, observing for arcing or hot socket jaws. If a
   socket jaw shows discoloration or pitting, **stop work** and open a
   damaged-socket work order; do not install the new meter into a damaged
   socket.
4. Insert the new meter, ensuring all four jaws are seated.
5. Wait for the new meter to complete its self-test (approximately 10 seconds)
   and confirm the kWh display advances under load.

## Network registration

The meter registers automatically to the nearest collector within **15
minutes** under normal conditions. Verify registration on the field tablet
before leaving the premise. If registration fails:

- Confirm the collector serving the premise is online in NMS.
- Trigger a manual ping from the tablet.
- If still unregistered after 30 minutes, leave the meter installed, attach a
  white "needs network verification" tag to the work order, and dispatch a
  network-engineering follow-up.

## Customer communication

Leave a door-hanger that includes:

- The date and approximate time of the install.
- The new meter serial number.
- A reminder to reset digital clocks, microwave timers, and similar devices.
- The 1-555-0142 customer line for billing or display questions.
