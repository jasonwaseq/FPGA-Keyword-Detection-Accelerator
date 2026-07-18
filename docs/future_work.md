# Future Improvements

Ordered roughly by value per effort.

## Protocol / control
* **WRITE_REG / READ_REG commands** — the smoothing/pooling configuration
  registers already exist with parameter defaults; exposing them makes
  threshold tuning a host-side loop instead of a re-synthesis.
* **Sequence numbers on events** plus a host-side gap detector, closing the
  last theoretical hole in event accounting (today: keyword counter vs.
  received events).

## Model capacity
* **Second conv layer / more channels.** LC headroom is modest (87 %) but EBR
  (11 free) and SPRAM (4 × 32 KB untouched) are not: weights for a much
  larger model fit; the constraint is control logic. A microcoded layer
  sequencer (one engine, layer descriptors in ROM) would trade the
  per-layer FSM duplication for a small program store and unlock deeper
  networks at roughly constant LC.
* **Depthwise-separable conv** — natural fit for the existing lane
  structure and the standard next step for KWS accuracy per MAC.
* **PARALLEL=8** — RTL supports it; needs sliced kernel-ROM init images
  (build-system change) and the P>4 bench harness accessors.

## Throughput
* **921600 baud** (FTDI supports it): 8× feature bandwidth for richer
  features or shorter hop; RX oversampling margin at 12 MHz (13 clocks/bit)
  is workable but a 24/36 MHz PLL clock would restore comfort.
* **Inter-layer pipelining** — only worthwhile after the link stops being
  the bottleneck; costs double-buffered activation RAMs (2 EBR).

## Verification
* Gate-level (post-PnR) simulation of the boot + one-window scenario.
* Coverage collection (`verilator --coverage`) wired into CI with a ratchet.
* Constrained-random command fuzzer against a protocol reference state
  machine (today's command coverage is directed).

## Host
* ALSA capture is compile-time optional; add a WASAPI backend to replace
  the legacy WinMM path, and automatic FTDI port discovery via libusb IDs.
* On-line calibration mode: stream labelled clips, compute detection ROC,
  recommend threshold/vote settings.

## Hardware bring-up extras
* Power measurement (INA219 on the iCEBreaker 5 V rail) for a real mW
  figure in performance.md.
* A logic-analyzer profile (sigrok) for the IRQ pin + UART, documented as a
  debug recipe.
