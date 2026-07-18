# -----------------------------------------------------------------------------
# Project : iCE40 KWS Accelerator
# File    : host/Makefile
# Purpose : Build the host streaming application (kws_host) and nothing else.
#           Invoked standalone or via the top-level Makefile ('make host').
#
#   make            - release build
#   make DEBUG=1    - -O0 -g build
#   make ALSA=1     - enable ALSA microphone backend on Linux
# -----------------------------------------------------------------------------

CC      ?= gcc
BUILD   := build
BIN     := $(BUILD)/kws_host

SRCS := src/main.c src/protocol.c src/ref_model.c src/mfcc.c src/audio.c \
        src/serial.c src/config.c src/log.c src/stats.c
OBJS := $(SRCS:src/%.c=$(BUILD)/%.o)

CFLAGS  ?= -O2
CFLAGS  += -std=c11 -Wall -Wextra -Iinclude -I../weights
# Expose POSIX/BSD declarations (clock_gettime, usleep, cfmakeraw) under
# strict -std=c11 on glibc; no effect on Windows builds.
CFLAGS  += -D_DEFAULT_SOURCE -D_POSIX_C_SOURCE=200809L
LDFLAGS :=
LDLIBS  := -lm

ifeq ($(DEBUG),1)
CFLAGS := $(filter-out -O2,$(CFLAGS)) -O0 -g
endif

ifeq ($(OS),Windows_NT)
BIN    := $(BIN).exe
LDLIBS += -lwinmm
else
ifeq ($(ALSA),1)
CFLAGS += -DKWS_USE_ALSA
LDLIBS += -lasound
endif
endif

.PHONY: all clean hwtest

all: $(BIN)

# Hardware acceptance test (POSIX only; runs the tb_kws_core scenario
# against the real board - see src/hwtest.c)
HWTEST := $(BUILD)/kws_hwtest
hwtest: $(HWTEST)
$(HWTEST): src/hwtest.c src/protocol.c src/ref_model.c src/serial.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

# Training featurizer: WAV clips -> INT8 feature records through the exact
# deployment MFCC front end (see src/featurize.c and docs/training.md)
FEATURIZE := $(BUILD)/kws_featurize
featurize: $(FEATURIZE)
$(FEATURIZE): src/featurize.c src/mfcc.c | $(BUILD)
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(BIN): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/%.o: src/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
