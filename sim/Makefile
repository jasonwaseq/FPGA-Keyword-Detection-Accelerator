# -----------------------------------------------------------------------------
# Project : iCE40 KWS Accelerator
# File    : sim/Makefile
# Purpose : Verilator regression build/run. Each bench is verilated with its
#           own top module and compiled together with the shared C reference
#           model (host/src/ref_model.c, protocol.c) as the golden oracle.
#
#   make            - build and run the full regression
#   make build      - build all benches
#   make run-tb_X   - run one bench (add SIMARGS=+trace for VCD in sim/out/)
#
# Portable across Linux CI (verilator in PATH) and Windows/MSYS2 with the
# OSS CAD Suite (set VERILATOR=verilator_bin.exe and VERILATOR_ROOT).
# The verilated model is compiled manually (single g++ invocation) instead of
# via the generated sub-make: identical everywhere, no recursive-make quirks.
# -----------------------------------------------------------------------------

TOP_DIR   := ..
OUT       := out

VERILATOR      ?= verilator
VERILATOR_ROOT ?= $(shell $(VERILATOR) --getenv VERILATOR_ROOT 2>/dev/null)

CC  ?= gcc
CXX ?= g++

ifeq ($(OS),Windows_NT)
EXE := .exe
else
EXE :=
endif

# Package first: Verilator resolves imports in file order.
RTL_SRCS := $(TOP_DIR)/rtl/kws_pkg.sv \
            $(filter-out $(TOP_DIR)/rtl/kws_pkg.sv,$(wildcard $(TOP_DIR)/rtl/*.sv))

VLT_FLAGS := --cc --assert --trace -O2 --x-assign 0 --x-initial 0 \
             -Wno-UNOPTFLAT

VLT_RUNTIME := $(VERILATOR_ROOT)/include/verilated.cpp \
               $(VERILATOR_ROOT)/include/verilated_vcd_c.cpp \
               $(VERILATOR_ROOT)/include/verilated_threads.cpp

CXXFLAGS := -std=c++17 -O2 \
            -I$(VERILATOR_ROOT)/include -I$(VERILATOR_ROOT)/include/vltstd \
            -I$(TOP_DIR)/tb/common -I$(TOP_DIR)/host/include -I$(TOP_DIR)/weights
CFLAGS   := -std=c11 -O2 -Wall \
            -I$(TOP_DIR)/host/include -I$(TOP_DIR)/weights

# Reference model objects (C, shared with the host application)
REF_OBJS := $(OUT)/ref_model.o $(OUT)/protocol.o

$(OUT)/%.o: $(TOP_DIR)/host/src/%.c | $(OUT)
	$(CC) $(CFLAGS) -c -o $@ $<

$(OUT):
	mkdir -p $(OUT)

# --- bench table --------------------------------------------------------------
# name : top module : extra verilator -G flags : extra CXXFLAGS : cpp basename
TBS := tb_crc tb_uart_rx tb_uart_tx tb_fifo tb_decoder tb_encoder \
       tb_conv1d_p1 tb_conv1d_p2 tb_conv1d_p4 \
       tb_pooling tb_classifier tb_smoothing tb_kws_core

top_tb_crc          := crc16
top_tb_uart_rx      := uart_rx
top_tb_uart_tx      := uart_tx
top_tb_fifo         := uart_fifo
top_tb_decoder      := packet_decoder
top_tb_encoder      := packet_encoder
top_tb_conv1d_p1    := conv1d_engine
top_tb_conv1d_p2    := conv1d_engine
top_tb_conv1d_p4    := conv1d_engine
top_tb_pooling      := pooling_engine
top_tb_classifier   := classifier
top_tb_smoothing    := temporal_smoothing
top_tb_kws_core     := kws_core

gflags_tb_decoder   := -GTIMEOUT_CYCLES=2000
gflags_tb_conv1d_p1 := -GPARALLEL=1
gflags_tb_conv1d_p2 := -GPARALLEL=2
gflags_tb_conv1d_p4 := -GPARALLEL=4
gflags_tb_kws_core  := -GBAUD_RATE=750000

cxx_tb_conv1d_p1    := -DPARAM_PARALLEL=1
cxx_tb_conv1d_p2    := -DPARAM_PARALLEL=2
cxx_tb_conv1d_p4    := -DPARAM_PARALLEL=4

cpp_tb_conv1d_p1    := tb_conv1d
cpp_tb_conv1d_p2    := tb_conv1d
cpp_tb_conv1d_p4    := tb_conv1d

# default cpp basename = bench name
cppof = $(if $(cpp_$(1)),$(cpp_$(1)),$(1))

define TB_template
$(OUT)/$(1)$(EXE): $$(RTL_SRCS) $$(TOP_DIR)/tb/$$(call cppof,$(1)).cpp \
                   $$(TOP_DIR)/tb/common/tb_util.h $$(REF_OBJS) | $(OUT)
	$$(VERILATOR) $$(VLT_FLAGS) --top-module $$(top_$(1)) $$(gflags_$(1)) \
	  --Mdir obj_$(1) --prefix V$$(top_$(1)) $$(RTL_SRCS)
	$$(CXX) $$(CXXFLAGS) $$(cxx_$(1)) -Iobj_$(1) \
	  $$(TOP_DIR)/tb/$$(call cppof,$(1)).cpp obj_$(1)/*.cpp \
	  $$(VLT_RUNTIME) $$(REF_OBJS) -o $$@

.PHONY: run-$(1)
run-$(1): $(OUT)/$(1)$(EXE)
	cd $$(TOP_DIR) && sim/$(OUT)/$(1)$(EXE) $$(SIMARGS)
endef

$(foreach tb,$(TBS),$(eval $(call TB_template,$(tb))))

.PHONY: all build run clean
all: run

build: $(foreach tb,$(TBS),$(OUT)/$(tb)$(EXE))

run: $(foreach tb,$(TBS),run-$(tb))
	@echo "==== regression complete: all benches passed ===="

clean:
	rm -rf $(OUT) obj_*
