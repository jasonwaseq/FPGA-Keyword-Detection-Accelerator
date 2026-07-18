/* -----------------------------------------------------------------------------
 * Project : iCE40 KWS Accelerator
 * File    : hwtest.c
 * Purpose : Hardware acceptance test - the full-system testbench scenario
 *           (tb/tb_kws_core.cpp) executed against the real board over its
 *           UART. Uses the same protocol library and bit-accurate reference
 *           model as the simulation benches, so a pass means the silicon
 *           behaves exactly like the verified RTL:
 *
 *             1. PING/ACK with frame echo
 *             2. RESET (clean statistics baseline)
 *             3. READ_VERSION geometry contract vs compiled weights
 *             4. START + 90 feature frames (keyword bursts, 2 CRC-corrupted
 *                packets) - every EVT_KEYWORD must match the reference
 *                event stream in order, class, confidence, votes and frame
 *             5. statistics cross-check against ground truth
 *             6. STOP semantics (frames dropped, not committed)
 *             7. RESET clears statistics
 *             8. protocol error responses (unknown type / bad len / bad ver)
 *             9. second session: START behaves like a fresh boot
 *
 *           Exit code 0 = hardware validated.
 *
 * Usage: kws_hwtest [port]        (default /dev/ttyUSB1)
 * ---------------------------------------------------------------------------*/
#include "kws_protocol.h"
#include "kws_ref.h"
#include "kws_serial.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int g_errors = 0;
#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                     \
            printf(__VA_ARGS__);                                            \
            printf("\n");                                                   \
            g_errors++;                                                     \
        }                                                                   \
    } while (0)

/* Self-test stimulus: emitted alongside the weights and proven at emit time
 * to fire >= 1 keyword event under both streaming patterns used here. */
static int8_t g_self[KWS_SELFTEST_FRAMES][KWS_NUM_MFCC];

static int load_selftest(void)
{
    const char *paths[] = { "weights/selftest_frames.mem",
                            "../weights/selftest_frames.mem" };
    FILE *f = 0;
    for (unsigned i = 0; i < 2 && !f; i++) f = fopen(paths[i], "r");
    if (!f) {
        printf("FAIL cannot open weights/selftest_frames.mem\n");
        return -1;
    }
    char line[128];
    int n = 0;
    while (n < KWS_SELFTEST_FRAMES * KWS_NUM_MFCC && fgets(line, sizeof(line), f)) {
        if (line[0] == '/' || line[0] == '\n' || line[0] == '\r') continue;
        ((int8_t *)g_self)[n++] = (int8_t)strtol(line, 0, 16);
    }
    fclose(f);
    if (n != KWS_SELFTEST_FRAMES * KWS_NUM_MFCC) {
        printf("FAIL selftest_frames.mem short: %d bytes\n", n);
        return -1;
    }
    return 0;
}

static kws_serial_t *g_ser;
static kws_parser_t  g_parser;

/* --- reference mirror --------------------------------------------------------*/
#define MAX_EVT 32
static struct {
    int8_t   win[KWS_WINDOW_LEN][KWS_NUM_MFCC];   /* ring by commit index */
    uint32_t commit_fn[KWS_WINDOW_LEN];
    unsigned commits;
    kws_smooth_t smooth;
    unsigned inferences;
    struct { uint8_t cls, conf, votes; uint32_t fn; } evt[MAX_EVT];
    unsigned evt_head, evt_tail;
} g_ref;

static void ref_reset(void)
{
    memset(&g_ref, 0, sizeof(g_ref));
    kws_smooth_init(&g_ref.smooth, 0);
}

static void ref_commit(const int8_t *feat, uint32_t fn)
{
    memcpy(g_ref.win[g_ref.commits % KWS_WINDOW_LEN], feat, KWS_NUM_MFCC);
    g_ref.commit_fn[g_ref.commits % KWS_WINDOW_LEN] = fn;
    g_ref.commits++;

    if (g_ref.commits < KWS_WINDOW_LEN
        || (g_ref.commits - KWS_WINDOW_LEN) % 8 != 0) return;

    int8_t win[KWS_WINDOW_LEN][KWS_NUM_MFCC];
    for (unsigned t = 0; t < KWS_WINDOW_LEN; t++) {
        unsigned c = g_ref.commits - KWS_WINDOW_LEN + t;
        memcpy(win[t], g_ref.win[c % KWS_WINDOW_LEN], KWS_NUM_MFCC);
    }
    int8_t logits[KWS_NUM_CLASSES];
    int winner;
    kws_ref_infer(win, logits, &winner, KWS_POOL_MAX);
    g_ref.inferences++;

    kws_ref_event_t rev;
    if (kws_smooth_step(&g_ref.smooth, logits, winner, &rev)) {
        unsigned i = g_ref.evt_tail++ % MAX_EVT;
        g_ref.evt[i].cls   = rev.class_id;
        g_ref.evt[i].conf  = rev.confidence;
        g_ref.evt[i].votes = rev.votes;
        g_ref.evt[i].fn    = fn;
    }
}

static unsigned g_evt_matched = 0;

static void handle_event(const kws_packet_t *p)
{
    kws_event_t e;
    CHECK(kws_event_decode(p, &e) == 0, "malformed EVT_KEYWORD");
    CHECK(g_ref.evt_head < g_ref.evt_tail,
          "unexpected event (class %u frame %u)", e.class_id, p->frame_num);
    if (g_ref.evt_head >= g_ref.evt_tail) return;

    unsigned i = g_ref.evt_head++ % MAX_EVT;
    CHECK(e.class_id == g_ref.evt[i].cls,
          "event class %u != ref %u", e.class_id, g_ref.evt[i].cls);
    CHECK(e.confidence == g_ref.evt[i].conf,
          "event conf %u != ref %u", e.confidence, g_ref.evt[i].conf);
    CHECK(e.votes == g_ref.evt[i].votes,
          "event votes %u != ref %u", e.votes, g_ref.evt[i].votes);
    CHECK(p->frame_num == g_ref.evt[i].fn,
          "event frame %u != ref %u", p->frame_num, g_ref.evt[i].fn);
    CHECK(e.latency_cycles > 0 && e.latency_cycles < 100000,
          "implausible latency %u cycles", e.latency_cycles);
    printf("  event: class=%u conf=%u votes=%u frame=%u fpga_latency=%.2fms"
           "  [matches reference]\n",
           e.class_id, e.confidence, e.votes, p->frame_num,
           e.latency_cycles / 12000.0);
    g_evt_matched++;
}

/* --- link helpers ------------------------------------------------------------*/
static long long now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static void pump(void)
{
    uint8_t buf[256];
    int n = kws_serial_read(g_ser, buf, (int)sizeof(buf));
    for (int i = 0; i < n; i++) {
        if (kws_parser_feed(&g_parser, buf[i])) {
            if (g_parser.pkt.type == KWS_PKT_EVT_KEYWORD)
                handle_event(&g_parser.pkt);
            else
                CHECK(0, "unsolicited packet type 0x%02x", g_parser.pkt.type);
        }
    }
}

static int wait_pkt(uint8_t type, kws_packet_t *out, int timeout_ms)
{
    long long deadline = now_ms() + timeout_ms;
    while (now_ms() < deadline) {
        uint8_t buf[256];
        int n = kws_serial_read(g_ser, buf, (int)sizeof(buf));
        for (int i = 0; i < n; i++) {
            if (kws_parser_feed(&g_parser, buf[i])) {
                if (g_parser.pkt.type == KWS_PKT_EVT_KEYWORD) {
                    handle_event(&g_parser.pkt);
                } else if (g_parser.pkt.type == type) {
                    if (out) *out = g_parser.pkt;
                    return 0;
                } else {
                    CHECK(0, "got 0x%02x while waiting for 0x%02x",
                          g_parser.pkt.type, type);
                }
            }
        }
        if (n <= 0) usleep(1000);
    }
    CHECK(0, "timeout waiting for packet type 0x%02x", type);
    return -1;
}

static uint32_t g_cmd_fn = 0x4000;

static int send_raw(const uint8_t *buf, size_t n)
{
    return kws_serial_write(g_ser, buf, (int)n) == (int)n ? 0 : -1;
}

static void cmd_ack(uint8_t cmd)
{
    uint8_t buf[KWS_PROTO_MAX_PKT];
    uint32_t fn = g_cmd_fn++;
    size_t n = kws_pkt_build(buf, cmd, 0, 0, (uint32_t)now_ms(), fn);
    CHECK(send_raw(buf, n) == 0, "serial write failed");
    kws_packet_t ack;
    if (wait_pkt(KWS_PKT_RSP_ACK, &ack, 1000) == 0) {
        CHECK(ack.len == 1 && ack.payload[0] == cmd,
              "ACK echo wrong for 0x%02x", cmd);
        CHECK(ack.frame_num == fn, "ACK frame echo %u != %u", ack.frame_num, fn);
    }
}

static void read_stats(uint32_t v[KWS_STAT_NUM])
{
    uint8_t buf[KWS_PROTO_MAX_PKT];
    size_t n = kws_pkt_build(buf, KWS_PKT_CMD_READ_STATS, 0, 0,
                             (uint32_t)now_ms(), g_cmd_fn++);
    CHECK(send_raw(buf, n) == 0, "serial write failed");
    kws_packet_t p;
    memset(v, 0, 4 * KWS_STAT_NUM);
    if (wait_pkt(KWS_PKT_RSP_STATS, &p, 1000) == 0) {
        CHECK(p.len == 4 * KWS_STAT_NUM, "stats payload %u bytes", p.len);
        for (int i = 0; i < KWS_STAT_NUM; i++)
            v[i] = (uint32_t)p.payload[4*i]       | (uint32_t)p.payload[4*i+1] << 8
                 | (uint32_t)p.payload[4*i+2] << 16 | (uint32_t)p.payload[4*i+3] << 24;
    }
}

static void send_feature(const int8_t *feat, uint32_t fn, int corrupt,
                         int streaming)
{
    uint8_t buf[KWS_PROTO_MAX_PKT];
    size_t n = kws_pkt_build(buf, KWS_PKT_DATA_FEATURE, (const uint8_t *)feat,
                             KWS_NUM_MFCC, (uint32_t)now_ms(), fn);
    if (corrupt) buf[KWS_PROTO_HDR_BYTES + 7] ^= 0x10;
    CHECK(send_raw(buf, n) == 0, "serial write failed");
    if (!corrupt && streaming) ref_commit(feat, fn);
    pump();
}

/* ---------------------------------------------------------------------------*/
int main(int argc, char **argv)
{
    const char *port = (argc > 1) ? argv[1] : "/dev/ttyUSB1";
    char err[128];

    printf("== iCE40 KWS hardware acceptance test ==\n");
    printf("port: %s\n", port);
    if (load_selftest() != 0) return 1;

    g_ser = kws_serial_open(port, 115200, err, sizeof(err));
    if (!g_ser) { printf("FAIL serial: %s\n", err); return 1; }
    kws_parser_init(&g_parser);
    ref_reset();
    usleep(200000);   /* let the line settle after open */

    /* 1-2: liveness + clean baseline */
    cmd_ack(KWS_PKT_CMD_PING);
    printf("PING/ACK: ok\n");
    cmd_ack(KWS_PKT_CMD_RESET);
    printf("RESET: ok\n");

    /* 3: geometry contract */
    {
        uint8_t buf[KWS_PROTO_MAX_PKT];
        size_t n = kws_pkt_build(buf, KWS_PKT_CMD_READ_VERSION, 0, 0, 0,
                                 g_cmd_fn++);
        CHECK(send_raw(buf, n) == 0, "serial write failed");
        kws_packet_t v;
        if (wait_pkt(KWS_PKT_RSP_VERSION, &v, 1000) == 0) {
            printf("FPGA v%u.%u.%u proto %u | mfcc=%u win=%u stride=%u k=%u "
                   "ch=%u cls=%u par=%u clk=%uMHz\n",
                   v.payload[0], v.payload[1], v.payload[2], v.payload[3],
                   v.payload[4], v.payload[5], v.payload[6], v.payload[7],
                   v.payload[8], v.payload[9], v.payload[10], v.payload[11]);
            CHECK(v.len == 12, "version payload %u bytes", v.len);
            CHECK(v.payload[3] == KWS_PROTO_VERSION, "protocol mismatch");
            CHECK(v.payload[4] == KWS_NUM_MFCC,      "NUM_MFCC mismatch");
            CHECK(v.payload[5] == KWS_WINDOW_LEN,    "WINDOW_LEN mismatch");
            CHECK(v.payload[6] == 8,                 "STRIDE mismatch");
            CHECK(v.payload[7] == KWS_CONV_K,        "CONV_K mismatch");
            CHECK(v.payload[8] == KWS_CONV_OUT_CH,   "CONV_OUT_CH mismatch");
            CHECK(v.payload[9] == KWS_NUM_CLASSES,   "NUM_CLASSES mismatch");
        }
    }

    /* 4: session 1 - stream the self-test frames with fault injection */
    cmd_ack(KWS_PKT_CMD_START_STREAM);
    ref_reset();
    unsigned corrupted = 0;
    for (uint32_t fn = 0; fn < KWS_SELFTEST_FRAMES; fn++) {
        int corrupt = (fn == 12 || fn == 55);
        if (corrupt) corrupted++;
        send_feature(g_self[fn], fn, corrupt, 1);
        usleep(10000);   /* 100 fps, as in deployment */
    }
    /* drain in-flight events */
    for (int i = 0; i < 50; i++) { pump(); usleep(10000); }

    CHECK(g_evt_matched > 0, "stimulus produced no keyword events");
    CHECK(g_ref.evt_head == g_ref.evt_tail,
          "%u reference events missing from hardware",
          g_ref.evt_tail - g_ref.evt_head);
    printf("session 1: %u keyword events, all match the reference model\n",
           g_evt_matched);

    /* 5: statistics ground truth */
    {
        uint32_t v[KWS_STAT_NUM];
        read_stats(v);
        CHECK(v[KWS_STAT_FRAMES_RX] == KWS_SELFTEST_FRAMES - corrupted,
              "frames_rx %u != %u", v[KWS_STAT_FRAMES_RX], KWS_SELFTEST_FRAMES - corrupted);
        CHECK(v[KWS_STAT_CRC_ERRORS] == corrupted,
              "crc_errors %u != %u", v[KWS_STAT_CRC_ERRORS], corrupted);
        CHECK(v[KWS_STAT_INFERENCES] == g_ref.inferences,
              "inferences %u != ref %u", v[KWS_STAT_INFERENCES],
              g_ref.inferences);
        CHECK(v[KWS_STAT_KEYWORDS] == g_evt_matched,
              "keywords %u != %u", v[KWS_STAT_KEYWORDS], g_evt_matched);
        CHECK(v[KWS_STAT_WINDOWS_DROP] == 0, "windows dropped: %u",
              v[KWS_STAT_WINDOWS_DROP]);
        CHECK(v[KWS_STAT_FRAMES_DROPPED] == 0, "frames dropped: %u",
              v[KWS_STAT_FRAMES_DROPPED]);
        CHECK(v[KWS_STAT_FIFO_OVERFLOWS] == 0, "fifo overflows: %u",
              v[KWS_STAT_FIFO_OVERFLOWS]);
        printf("stats: frames=%u crc_err=%u inferences=%u keywords=%u "
               "lat_max=%.2fms conv_busy=%.1f%%\n",
               v[KWS_STAT_FRAMES_RX], v[KWS_STAT_CRC_ERRORS],
               v[KWS_STAT_INFERENCES], v[KWS_STAT_KEYWORDS],
               v[KWS_STAT_LAT_MAX] / 12000.0,
               v[KWS_STAT_UPTIME_CYC]
                   ? 100.0 * v[KWS_STAT_CONV_BUSY] / v[KWS_STAT_UPTIME_CYC]
                   : 0.0);
    }

    /* 6: STOP drops features */
    cmd_ack(KWS_PKT_CMD_STOP_STREAM);
    {
        int8_t feat[KWS_NUM_MFCC] = {0};
        send_feature(feat, 1000, 0, 0);
        usleep(50000);
        uint32_t v[KWS_STAT_NUM];
        read_stats(v);
        CHECK(v[KWS_STAT_FRAMES_DROPPED] == 1,
              "frames_dropped %u != 1 after STOP", v[KWS_STAT_FRAMES_DROPPED]);
        printf("STOP semantics: ok\n");
    }

    /* 7: RESET clears statistics */
    cmd_ack(KWS_PKT_CMD_RESET);
    {
        uint32_t v[KWS_STAT_NUM];
        read_stats(v);
        CHECK(v[KWS_STAT_FRAMES_RX] == 0 && v[KWS_STAT_KEYWORDS] == 0
              && v[KWS_STAT_INFERENCES] == 0, "RESET did not clear stats");
        printf("RESET clears statistics: ok\n");
    }

    /* 8: protocol error responses */
    {
        uint8_t buf[KWS_PROTO_MAX_PKT];
        kws_packet_t p;

        size_t n = kws_pkt_build(buf, 0x55, 0, 0, 0, g_cmd_fn++);
        send_raw(buf, n);
        if (wait_pkt(KWS_PKT_RSP_ERROR, &p, 1000) == 0)
            CHECK(p.payload[0] == KWS_ERR_UNKNOWN_TYPE && p.payload[1] == 0x55,
                  "unknown-type error %02x %02x", p.payload[0], p.payload[1]);

        uint8_t short_feat[39] = {0};
        n = kws_pkt_build(buf, KWS_PKT_DATA_FEATURE, short_feat, 39, 0,
                          g_cmd_fn++);
        send_raw(buf, n);
        if (wait_pkt(KWS_PKT_RSP_ERROR, &p, 1000) == 0)
            CHECK(p.payload[0] == KWS_ERR_BAD_LENGTH, "bad-length error missing");

        n = kws_pkt_build(buf, KWS_PKT_CMD_PING, 0, 0, 0, g_cmd_fn++);
        buf[1] = 0x02;
        uint16_t crc = kws_crc16(&buf[1], KWS_PROTO_HDR_BYTES - 1);
        buf[KWS_PROTO_HDR_BYTES]     = (uint8_t)(crc & 0xFF);
        buf[KWS_PROTO_HDR_BYTES + 1] = (uint8_t)(crc >> 8);
        send_raw(buf, n);
        if (wait_pkt(KWS_PKT_RSP_ERROR, &p, 1000) == 0)
            CHECK(p.payload[0] == KWS_ERR_BAD_VERSION, "bad-version error missing");
        printf("protocol error responses: ok\n");
    }

    /* 9: second session behaves like a fresh boot */
    cmd_ack(KWS_PKT_CMD_START_STREAM);
    ref_reset();
    unsigned before = g_evt_matched;
    for (uint32_t fn = 2000; fn < 2000 + KWS_SELFTEST_FRAMES; fn++) {
        send_feature(g_self[fn - 2000], fn, 0, 1);
        usleep(10000);
    }
    for (int i = 0; i < 50; i++) { pump(); usleep(10000); }
    CHECK(g_ref.evt_head == g_ref.evt_tail, "session 2: reference events missing");
    CHECK(g_evt_matched > before, "session 2 produced no events");
    printf("session 2: %u keyword events, all match the reference model\n",
           g_evt_matched - before);

    cmd_ack(KWS_PKT_CMD_STOP_STREAM);
    kws_serial_close(g_ser);

    if (g_errors == 0) {
        printf("== PASS: hardware validated against the reference model ==\n");
        return 0;
    }
    printf("== FAIL: %d errors ==\n", g_errors);
    return 1;
}
