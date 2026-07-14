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

.PHONY: all clean

all: $(BIN)

$(BIN): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

$(BUILD)/%.o: src/%.c | $(BUILD)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILD):
	mkdir -p $(BUILD)

clean:
	rm -rf $(BUILD)
