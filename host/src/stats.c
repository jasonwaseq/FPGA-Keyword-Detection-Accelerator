/* -----------------------------------------------------------------------------
 * Project : iCE40 KWS Accelerator
 * File    : stats.c
 * Purpose : Host statistics implementation (see kws_stats.h).
 * ---------------------------------------------------------------------------*/
#include "kws_stats.h"
#include "kws_protocol.h"
#include "kws_log.h"

#include <string.h>

void kws_stats_init(kws_stats_t *s)
{
    memset(s, 0, sizeof(*s));
    s->lat_min_ms = 1e30;
}

void kws_stats_frame_sent(kws_stats_t *s, uint32_t frame_num, long long now_us)
{
    s->frames_sent++;
    s->send_us[frame_num & (KWS_LATRING - 1)] = now_us;
}

double kws_stats_event(kws_stats_t *s, uint32_t frame_num, long long now_us)
{
    s->events_rx++;
    if (s->frames_sent == 0
        || frame_num + KWS_LATRING < s->frames_sent) return -1.0;

    long long sent = s->send_us[frame_num & (KWS_LATRING - 1)];
    if (sent == 0) return -1.0;

    double ms = (double)(now_us - sent) / 1000.0;
    s->lat_sum_ms += ms;
    if (ms < s->lat_min_ms) s->lat_min_ms = ms;
    if (ms > s->lat_max_ms) s->lat_max_ms = ms;
    s->lat_n++;
    return ms;
}

void kws_stats_print(const kws_stats_t *s)
{
    KWS_INFO("---- host session statistics ----");
    KWS_INFO("frames sent      : %u", s->frames_sent);
    KWS_INFO("commands sent    : %u (acks %u, errors %u)",
             s->cmds_sent, s->acks_rx, s->errors_rx);
    KWS_INFO("rx parse CRC err : %u", s->parse_crc_err);
    KWS_INFO("keyword events   : %u (reference predicted %u, agree %u)",
             s->events_rx, s->ref_events, s->agree);
    if (s->lat_n) {
        KWS_INFO("event latency ms : min %.1f  avg %.1f  max %.1f  (n=%u)",
                 s->lat_min_ms, s->lat_sum_ms / s->lat_n, s->lat_max_ms,
                 s->lat_n);
    }
}

void kws_stats_print_fpga(const uint8_t *payload, int len, int clk_mhz)
{
    static const char *names[KWS_STAT_NUM] = {
        "frames_rx", "frames_dropped", "pkts_rx", "crc_errors",
        "framing_errors", "fifo_overflows", "windows_sched", "inferences",
        "keywords", "lat_last_cyc", "lat_max_cyc", "lat_acc_cyc",
        "conv_busy_cyc", "uptime_cyc", "windows_drop", "tx_pkts"
    };

    if (len < 4 * KWS_STAT_NUM) {
        KWS_WARN("short RSP_STATS payload (%d bytes)", len);
        return;
    }

    uint32_t v[KWS_STAT_NUM];
    for (int i = 0; i < KWS_STAT_NUM; i++) {
        v[i] = (uint32_t)payload[4 * i]
             | (uint32_t)payload[4 * i + 1] << 8
             | (uint32_t)payload[4 * i + 2] << 16
             | (uint32_t)payload[4 * i + 3] << 24;
    }

    KWS_INFO("---- FPGA statistics ----");
    for (int i = 0; i < KWS_STAT_NUM; i++) {
        KWS_INFO("  %-15s : %u", names[i], v[i]);
    }
    if (clk_mhz > 0) {
        if (v[KWS_STAT_INFERENCES]) {
            double avg = (double)v[KWS_STAT_LAT_ACC] / v[KWS_STAT_INFERENCES];
            KWS_INFO("  inference latency: last %.2f ms, max %.2f ms, avg %.2f ms",
                     v[KWS_STAT_LAT_LAST] / (clk_mhz * 1e3),
                     v[KWS_STAT_LAT_MAX]  / (clk_mhz * 1e3),
                     avg / (clk_mhz * 1e3));
        }
        if (v[KWS_STAT_UPTIME_CYC]) {
            KWS_INFO("  conv utilization : %.2f%% (mod-2^32 window)",
                     100.0 * v[KWS_STAT_CONV_BUSY] / v[KWS_STAT_UPTIME_CYC]);
        }
    }
}
