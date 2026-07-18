# Build Instructions

## Toolchain

Everything is open source:

| Tool | Purpose |
|---|---|
| [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) | Yosys, nextpnr-ice40, IceStorm (icepack/icetime/iceprog), Verilator, GTKWave |
| GCC (or MinGW/MSYS2 UCRT64 on Windows) | host application + Verilator C++ benches |
| GNU make | build orchestration |
| Python 3 | weight generation / export tooling |

### Linux / CI

```sh
# with oss-cad-suite on PATH:
make lint      # Verilator -Wall, zero warnings expected
make sim       # full regression (13 benches)
make bit       # synthesis -> PnR -> bitstream (build/kws.bin)
make time      # icetime report
make host      # host app -> host/build/kws_host
make prog      # flash the iCEBreaker
```

### Windows (MSYS2 + OSS CAD Suite)

The OSS CAD Suite Windows build ships `verilator_bin.exe` but no C++
compiler; use MSYS2 UCRT64 gcc/g++ for the benches and host app:

```powershell
$env:PATH = "C:\msys64\ucrt64\bin;C:\msys64\usr\bin;<oss-cad-suite>\bin;<oss-cad-suite>\lib;$env:PATH"
$env:VERILATOR      = "verilator_bin.exe"
$env:VERILATOR_ROOT = "<oss-cad-suite>/share/verilator"
make lint sim bit host
```

Notes:
* `<oss-cad-suite>\lib` must be on PATH (tool DLLs live there).
* The sim Makefile compiles verilated models with a single g++ invocation
  instead of the generated sub-make — identical flow on both platforms.

## Build outputs

| Path | Content |
|---|---|
| `build/kws.json` | synthesized netlist (+ `build/yosys.log`) |
| `build/kws.asc`, `build/kws.bin` | placed/routed design, bitstream |
| `build/nextpnr.log`, `build/nextpnr_report.json` | utilization + timing |
| `build/icetime.rpt` | independent static timing |
| `sim/out/` | bench binaries, optional VCDs |
| `host/build/kws_host` | host streaming application |

## Regenerating weights

```sh
make weights-bringup    # UNTRAINED bring-up set - replaces the shipped
                        # trained artifacts; retrain via docs/training.md
# full retraining/export pipeline: docs/training.md
make sim bit host       # ROMs are baked at synthesis; the reference model
                        # and regression pick up the new header automatically
```

## Running the demo

1. `make prog` (iCEBreaker on USB; bitstream persists in flash).
2. Find the UART: the iCEBreaker enumerates two FTDI channels — the design
   is on the **second** (`COMn+1` / `/dev/ttyUSB1`).
3. `host/build/kws_host --port COM7 --input mic` (or `--input synth` for a
   hardware-free soak test, `--input wav:file.wav` for recorded audio).
4. Watch for `>>> KEYWORD ...` lines; `--stats 10` polls FPGA statistics.
   The green LED heartbeats while streaming and goes solid on detection.

The shipped weights are the deterministic bring-up set — they exercise the
full pipeline bit-exactly but are not trained for accuracy. Train + export
(docs/training.md) for real keyword spotting.
