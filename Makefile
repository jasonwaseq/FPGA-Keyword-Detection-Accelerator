# -----------------------------------------------------------------------------
# Project : iCE40 KWS Accelerator
# File    : Makefile (repository root)
# Purpose : Single entry point for every flow.
#
#   make lint       - Verilator -Wall lint of the full design
#   make sim        - build + run the complete Verilator regression
#   make synth      - Yosys synthesis (build/kws.json)
#   make pnr        - nextpnr-ice40 place & route (build/kws.asc)
#   make bit        - icepack bitstream (build/kws.bin)
#   make time       - icetime timing analysis (build/icetime.rpt)
#   make prog       - flash the iCEBreaker (iceprog)
#   make host       - build the host application (host/build/kws_host)
#   make weights    - regenerate the deterministic bring-up weight set
#   make all        - lint + sim + bit + host
#
# Windows/MSYS2 note: point VERILATOR at verilator_bin.exe and set
# VERILATOR_ROOT to <oss-cad-suite>/share/verilator; everything else is
# identical to the Linux flow. See docs/build.md.
# -----------------------------------------------------------------------------

VERILATOR ?= verilator
PYTHON    ?= python3

RTL_PKG  := rtl/kws_pkg.sv
RTL_SRCS := $(RTL_PKG) $(filter-out $(RTL_PKG),$(wildcard rtl/*.sv))

BUILD := build

.PHONY: all lint sim synth pnr bit time prog host weights clean

all: lint sim bit host

# --- static verification --------------------------------------------------------
lint:
	$(VERILATOR) --lint-only -Wall --timing -sv --top-module kws_top $(RTL_SRCS)
	@echo "==== lint clean ===="

# --- simulation regression -------------------------------------------------------
sim:
	$(MAKE) -C sim run

# --- FPGA flow --------------------------------------------------------------------
$(BUILD):
	mkdir -p $(BUILD)

synth: $(BUILD)/kws.json
$(BUILD)/kws.json: $(RTL_SRCS) scripts/synth.ys weights/conv_weights.mem | $(BUILD)
	yosys -q -l $(BUILD)/yosys.log scripts/synth.ys

pnr: $(BUILD)/kws.asc
$(BUILD)/kws.asc: $(BUILD)/kws.json constraints/icebreaker.pcf
	nextpnr-ice40 --up5k --package sg48 --json $(BUILD)/kws.json \
	  --pcf constraints/icebreaker.pcf --asc $(BUILD)/kws.asc \
	  --freq 12 --report $(BUILD)/nextpnr_report.json --seed 7 \
	  2> $(BUILD)/nextpnr.log
	@grep "Max frequency" $(BUILD)/nextpnr.log | tail -n 1

bit: $(BUILD)/kws.bin
$(BUILD)/kws.bin: $(BUILD)/kws.asc
	icepack $(BUILD)/kws.asc $(BUILD)/kws.bin

time: $(BUILD)/icetime.rpt
$(BUILD)/icetime.rpt: $(BUILD)/kws.asc
	icetime -d up5k -mtr $(BUILD)/icetime.rpt $(BUILD)/kws.asc

prog: $(BUILD)/kws.bin
	iceprog $(BUILD)/kws.bin

# --- host + model -----------------------------------------------------------------
host:
	$(MAKE) -C host

weights:
	$(PYTHON) model/gen_weights.py --out weights

clean:
	rm -rf $(BUILD)
	$(MAKE) -C sim clean
	$(MAKE) -C host clean
