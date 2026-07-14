/* -----------------------------------------------------------------------------
 * Project : iCE40 KWS Accelerator
 * File    : kws_stats.h
 * Purpose : Host-side session statistics: link health, event bookkeeping,
 *           end-to-end latency (audio frame sent -> event received) and
 *           agreement with the software reference model.
 * ---------------------------------------------------------------------------*/
#ifndef KWS_STATS_H
#define KWS_STATS_H

#include <stdint.h>
#include "kws_config.h"

#ifdef __cplusplus
extern "C" {
#endif

#define KWS_LATRING 1024   /* frame send-time ring (power of two) */

typedef struct {
    /* link */
    uint32_t frames_sent;
    uint32_t cmds_sent;
    uint32_t acks_rx;
    uint32_t errors_rx;      /* RSP_ERROR packets */
    uint32_t parse_crc_err;  /* corrupted FPGA->host packets */
    /* events */
    uint32_t events_rx;
    uint32_t ref_events;     /* software reference detections */
    uint32_t agree;          /* FPGA event matched by reference */
    /* end-to-end latency (ms) */
    double   lat_sum_ms;
    double   lat_min_ms;
    double   lat_max_ms;
    uint32_t lat_n;
    /* frame send timestamps for latency attribution */
    long long send_us[KWS_LATRING];
} kws_stats_t;

void kws_stats_init(kws_stats_t *s);
void kws_stats_frame_sent(kws_stats_t *s, uint32_t frame_num, long long now_us);

/* Record an FPGA event attributed to frame_num; returns latency in ms
 * (or -1.0 if the frame left the ring). */
double kws_stats_event(kws_stats_t *s, uint32_t frame_num, long long now_us);

void kws_stats_print(const kws_stats_t *s);

/* Pretty-print an RSP_STATS payload (16 u32 LE) from the FPGA. */
void kws_stats_print_fpga(const uint8_t *payload, int len, int clk_mhz);

#ifdef __cplusplus
}
#endif
#endif /* KWS_STATS_H */
