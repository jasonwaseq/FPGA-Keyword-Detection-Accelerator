// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Bench   : tb_kws_core - full-system verification over a live UART link
//
// Drives kws_core through its serial interface exactly as the host would
// (real bit-level UART timing; BAUD_RATE is raised to CLK/16 via -G override
// purely to shorten wall-clock time) and locks every observable output
// against the bit-accurate C reference model:
//
//   * command plane: PING/ACK, READ_VERSION geometry contract, START/STOP,
//     RESET, RSP_ERROR for unknown type / bad length / bad version
//   * streaming plane: 90 feature frames including keyword bursts and two
//     CRC-corrupted packets; every EVT_KEYWORD must match the reference
//     event stream in order, class, confidence, votes and frame attribution
//   * statistics plane: frames_rx / crc_errors / inferences / keywords /
//     windows_drop / frames_dropped cross-checked against ground truth
//   * restart plane: a second START session must behave like a fresh boot
// -----------------------------------------------------------------------------
#include "tb_util.h"
#include "Vkws_core.h"
#include "kws_protocol.h"
#include "kws_ref.h"

#include <array>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const unsigned CPB = 16;   // 12 MHz / 750000 baud (-GBAUD_RATE)

// --- globals ------------------------------------------------------------------
static Harness<Vkws_core> *H;
static UartBfm  bfm(CPB);
static kws_parser_t parser;
static std::deque<kws_packet_t> rx_pkts;

// reference mirror
struct RefMirror {
    std::vector<std::array<int8_t, KWS_NUM_MFCC>> commits;
    std::vector<uint32_t> commit_fn;
    kws_smooth_t smooth;
    unsigned inferences = 0;
    struct Evt { uint8_t cls, conf, votes; uint32_t fn; };
    std::deque<Evt> events;

    void reset() {
        commits.clear();
        commit_fn.clear();
        kws_smooth_init(&smooth, 0);
        inferences = 0;
        events.clear();
    }
    void commit(const int8_t *feat, uint32_t fn) {
        std::array<int8_t, KWS_NUM_MFCC> a;
        memcpy(a.data(), feat, KWS_NUM_MFCC);
        commits.push_back(a);
        commit_fn.push_back(fn);
        size_t n = commits.size();
        if (n >= KWS_WINDOW_LEN && (n - KWS_WINDOW_LEN) % 8 == 0) {
            int8_t win[KWS_WINDOW_LEN][KWS_NUM_MFCC];
            for (size_t t = 0; t < KWS_WINDOW_LEN; t++)
                memcpy(win[t], commits[n - KWS_WINDOW_LEN + t].data(),
                       KWS_NUM_MFCC);
            int8_t logits[KWS_NUM_CLASSES];
            int winner;
            kws_ref_infer(win, logits, &winner, KWS_POOL_MAX);
            inferences++;
            kws_ref_event_t rev;
            if (kws_smooth_step(&smooth, logits, winner, &rev)) {
                events.push_back({rev.class_id, rev.confidence, rev.votes,
                                  commit_fn.back()});
            }
        }
    }
};
static RefMirror ref;
static unsigned evt_matched = 0;

static void handle_event(const kws_packet_t &p)
{
    kws_event_t e;
    CHECK(kws_event_decode(&p, &e) == 0, "malformed EVT_KEYWORD");
    CHECK(!ref.events.empty(), "unexpected event (class %u frame %u)",
          e.class_id, p.frame_num);
    if (ref.events.empty()) return;
    RefMirror::Evt exp = ref.events.front();
    ref.events.pop_front();
    CHECK(e.class_id == exp.cls,   "event class %u != ref %u", e.class_id, exp.cls);
    CHECK(e.confidence == exp.conf,"event conf %u != ref %u", e.confidence, exp.conf);
    CHECK(e.votes == exp.votes,    "event votes %u != ref %u", e.votes, exp.votes);
    CHECK(p.frame_num == exp.fn,   "event frame %u != ref %u", p.frame_num, exp.fn);
    CHECK(e.latency_cycles > 0 && e.latency_cycles < 100000,
          "implausible latency %u cycles", e.latency_cycles);
    evt_matched++;
}

// Self-test stimulus: emitted alongside the weights by model/kws_quant.py and
// PROVEN (at emit time) to fire >= 1 keyword event through the current model
// under both the clean and the fault-injection streaming patterns below.
// This keeps the bench weight-agnostic: retraining regenerates the stimulus.
static int8_t g_self[KWS_SELFTEST_FRAMES][KWS_NUM_MFCC];

static void load_selftest(void)
{
    FILE *f = fopen("weights/selftest_frames.mem", "r");
    if (!f) {
        printf("FAIL cannot open weights/selftest_frames.mem (run from repo root)\n");
        exit(1);
    }
    char line[128];
    int n = 0;
    while (n < KWS_SELFTEST_FRAMES * KWS_NUM_MFCC && fgets(line, sizeof(line), f)) {
        if (line[0] == '/' || line[0] == '\n' || line[0] == '\r') continue;
        ((int8_t *)g_self)[n++] = (int8_t)strtol(line, nullptr, 16);
    }
    fclose(f);
    if (n != KWS_SELFTEST_FRAMES * KWS_NUM_MFCC) {
        printf("FAIL selftest_frames.mem short: %d bytes\n", n);
        exit(1);
    }
}

// advance simulation, driving/monitoring the UART and collecting packets
static bool     g_debug = false;
static uint64_t g_txd_edges = 0;

static void step(unsigned cycles)
{
    static int last_txd = 1;
    for (unsigned i = 0; i < cycles; i++) {
        H->dut->uart_rxd_i = (uint8_t)bfm.drive();
        H->tick();
        if (H->dut->uart_txd_o != last_txd) {
            g_txd_edges++;
            last_txd = H->dut->uart_txd_o;
        }
        bfm.monitor(H->dut->uart_txd_o);
        while (!bfm.rx_q.empty()) {
            uint8_t b = bfm.rx_q.front();
            bfm.rx_q.pop_front();
            if (g_debug) printf("    [rx] %02x\n", b);
            if (kws_parser_feed(&parser, b)) {
                if (parser.pkt.type == KWS_PKT_EVT_KEYWORD)
                    handle_event(parser.pkt);
                else
                    rx_pkts.push_back(parser.pkt);
            }
        }
    }
}

static void drain_tx() { while (bfm.busy()) step(1); }

// wait for a non-event packet of the given type
static bool wait_pkt(uint8_t type, kws_packet_t *out, unsigned timeout)
{
    for (unsigned i = 0; i < timeout; i += 50) {
        step(50);
        while (!rx_pkts.empty()) {
            kws_packet_t p = rx_pkts.front();
            rx_pkts.pop_front();
            if (p.type == type) { if (out) *out = p; return true; }
            CHECK(false, "unexpected packet type 0x%02x while waiting for 0x%02x",
                  p.type, type);
        }
    }
    CHECK(false, "timeout waiting for packet type 0x%02x", type);
    return false;
}

static uint32_t g_fn = 0x4000;   // command frame numbering (distinct range)

static void send_cmd_expect_ack(uint8_t cmd)
{
    uint8_t buf[KWS_PROTO_MAX_PKT];
    uint32_t fn = g_fn++;
    size_t n = kws_pkt_build(buf, cmd, 0, 0, 0xAABB0000u | fn, fn);
    bfm.send(buf, n);
    drain_tx();
    kws_packet_t ack;
    wait_pkt(KWS_PKT_RSP_ACK, &ack, 300000);
    CHECK(ack.len == 1 && ack.payload[0] == cmd, "ACK echo wrong for 0x%02x", cmd);
    CHECK(ack.frame_num == fn, "ACK frame echo %u != %u", ack.frame_num, fn);
}

static void read_stats(uint32_t v[KWS_STAT_NUM])
{
    uint8_t buf[KWS_PROTO_MAX_PKT];
    size_t n = kws_pkt_build(buf, KWS_PKT_CMD_READ_STATS, 0, 0, 0, g_fn++);
    bfm.send(buf, n);
    drain_tx();
    kws_packet_t p;
    wait_pkt(KWS_PKT_RSP_STATS, &p, 500000);
    CHECK(p.len == 4 * KWS_STAT_NUM, "stats payload %u bytes", p.len);
    for (int i = 0; i < KWS_STAT_NUM; i++)
        v[i] = (uint32_t)p.payload[4*i] | (uint32_t)p.payload[4*i+1] << 8
             | (uint32_t)p.payload[4*i+2] << 16 | (uint32_t)p.payload[4*i+3] << 24;
}

static void send_feature(const int8_t *feat, uint32_t fn, bool corrupt,
                         bool streaming)
{
    uint8_t buf[KWS_PROTO_MAX_PKT];
    size_t n = kws_pkt_build(buf, KWS_PKT_DATA_FEATURE, (const uint8_t *)feat,
                             KWS_NUM_MFCC, fn * 10, fn);
    if (corrupt) buf[KWS_PROTO_HDR_BYTES + 7] ^= 0x10;   // payload bit flip
    bfm.send(buf, n);
    if (!corrupt && streaming) ref.commit(feat, fn);
    drain_tx();
    step(200);   // inter-frame gap
}

int main(int argc, char **argv)
{
    Harness<Vkws_core> h(argc, argv, "sim/out/tb_kws_core.vcd");
    H = &h;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "+debug")) g_debug = true;
    load_selftest();
    kws_parser_init(&parser);
    kws_smooth_init(&ref.smooth, 0);

    h.dut->uart_rxd_i = 1;
    h.reset(10);
    step(100);

    if (g_debug) {
        // Path discriminator: START_STREAM flips stream_active_o without
        // involving the TX path at all.
        uint8_t buf[KWS_PROTO_MAX_PKT];
        size_t n = kws_pkt_build(buf, KWS_PKT_CMD_START_STREAM, 0, 0, 0, 1);
        bfm.send(buf, n);
        drain_tx();
        step(2000);
        printf("  [dbg] stream_active=%d txd_edges=%llu\n",
               (int)H->dut->stream_active_o,
               (unsigned long long)g_txd_edges);
    }

    // --- 1. PING ---------------------------------------------------------------
    send_cmd_expect_ack(KWS_PKT_CMD_PING);

    // --- 2. READ_VERSION: geometry contract ------------------------------------
    {
        uint8_t buf[KWS_PROTO_MAX_PKT];
        size_t n = kws_pkt_build(buf, KWS_PKT_CMD_READ_VERSION, 0, 0, 0, g_fn++);
        bfm.send(buf, n);
        drain_tx();
        kws_packet_t v;
        wait_pkt(KWS_PKT_RSP_VERSION, &v, 300000);
        CHECK(v.len == 12, "version payload %u bytes", v.len);
        CHECK(v.payload[3] == KWS_PROTO_VERSION, "protocol version mismatch");
        CHECK(v.payload[4] == KWS_NUM_MFCC,     "NUM_MFCC mismatch");
        CHECK(v.payload[5] == KWS_WINDOW_LEN,   "WINDOW_LEN mismatch");
        CHECK(v.payload[6] == 8,                "WINDOW_STRIDE mismatch");
        CHECK(v.payload[7] == KWS_CONV_K,       "CONV_K mismatch");
        CHECK(v.payload[8] == KWS_CONV_OUT_CH,  "CONV_OUT_CH mismatch");
        CHECK(v.payload[9] == KWS_NUM_CLASSES,  "NUM_CLASSES mismatch");
        CHECK(v.payload[11] == 12,              "clk MHz mismatch");
    }

    // --- 3. START + stream the self-test frames (+ 2 corrupted packets) ---------
    send_cmd_expect_ack(KWS_PKT_CMD_START_STREAM);
    ref.reset();

    unsigned corrupted = 0;
    for (uint32_t fn = 0; fn < KWS_SELFTEST_FRAMES; fn++) {
        bool corrupt = (fn == 12 || fn == 55);
        if (corrupt) corrupted++;
        send_feature(g_self[fn], fn, corrupt, true);
    }
    step(60000);   // drain in-flight inference + events

    CHECK(evt_matched > 0, "stimulus produced no keyword events");
    CHECK(ref.events.empty(), "%zu reference events missing from hardware",
          ref.events.size());
    printf("  session 1: %u keyword events, all bit-exact\n", evt_matched);

    // --- 4. statistics cross-check ------------------------------------------------
    {
        uint32_t v[KWS_STAT_NUM];
        read_stats(v);
        CHECK(v[KWS_STAT_FRAMES_RX] == KWS_SELFTEST_FRAMES - corrupted,
              "frames_rx %u != %u", v[KWS_STAT_FRAMES_RX], KWS_SELFTEST_FRAMES - corrupted);
        CHECK(v[KWS_STAT_CRC_ERRORS] == corrupted,
              "crc_errors %u != %u", v[KWS_STAT_CRC_ERRORS], corrupted);
        CHECK(v[KWS_STAT_INFERENCES] == ref.inferences,
              "inferences %u != ref %u", v[KWS_STAT_INFERENCES], ref.inferences);
        CHECK(v[KWS_STAT_WINDOWS_SCHED] == ref.inferences,
              "windows_sched %u != %u", v[KWS_STAT_WINDOWS_SCHED], ref.inferences);
        CHECK(v[KWS_STAT_KEYWORDS] == evt_matched,
              "keywords %u != %u", v[KWS_STAT_KEYWORDS], evt_matched);
        CHECK(v[KWS_STAT_WINDOWS_DROP] == 0, "windows dropped: %u",
              v[KWS_STAT_WINDOWS_DROP]);
        CHECK(v[KWS_STAT_FRAMES_DROPPED] == 0, "frames dropped: %u",
              v[KWS_STAT_FRAMES_DROPPED]);
        CHECK(v[KWS_STAT_FIFO_OVERFLOWS] == 0, "fifo overflows: %u",
              v[KWS_STAT_FIFO_OVERFLOWS]);
        CHECK(v[KWS_STAT_LAT_MAX] > 0 && v[KWS_STAT_LAT_MAX] < 100000,
              "implausible max latency %u", v[KWS_STAT_LAT_MAX]);
    }

    // --- 5. STOP: features must be dropped, not committed --------------------------
    send_cmd_expect_ack(KWS_PKT_CMD_STOP_STREAM);
    {
        int8_t feat[KWS_NUM_MFCC] = {0};
        send_feature(feat, 1000, false, false);
        uint32_t v[KWS_STAT_NUM];
        read_stats(v);
        CHECK(v[KWS_STAT_FRAMES_DROPPED] == 1,
              "frames_dropped %u != 1 after STOP", v[KWS_STAT_FRAMES_DROPPED]);
        CHECK(v[KWS_STAT_FRAMES_RX] == KWS_SELFTEST_FRAMES - corrupted,
              "frames_rx changed after STOP");
    }

    // --- 6. RESET clears statistics -------------------------------------------------
    send_cmd_expect_ack(KWS_PKT_CMD_RESET);
    {
        uint32_t v[KWS_STAT_NUM];
        read_stats(v);
        CHECK(v[KWS_STAT_FRAMES_RX] == 0 && v[KWS_STAT_KEYWORDS] == 0
              && v[KWS_STAT_INFERENCES] == 0,
              "statistics not cleared by RESET");
    }

    // --- 7. protocol error responses -------------------------------------------------
    {
        uint8_t buf[KWS_PROTO_MAX_PKT];
        kws_packet_t p;
        // unknown type
        size_t n = kws_pkt_build(buf, 0x55, 0, 0, 0, g_fn++);
        bfm.send(buf, n);
        drain_tx();
        wait_pkt(KWS_PKT_RSP_ERROR, &p, 300000);
        CHECK(p.payload[0] == KWS_ERR_UNKNOWN_TYPE && p.payload[1] == 0x55,
              "unknown-type error payload %02x %02x", p.payload[0], p.payload[1]);
        // bad length: feature with 39 bytes
        uint8_t short_feat[39] = {0};
        n = kws_pkt_build(buf, KWS_PKT_DATA_FEATURE, short_feat, 39, 0, g_fn++);
        bfm.send(buf, n);
        drain_tx();
        wait_pkt(KWS_PKT_RSP_ERROR, &p, 300000);
        CHECK(p.payload[0] == KWS_ERR_BAD_LENGTH, "bad-length error missing");
        // bad protocol version (re-CRC after patching VER)
        n = kws_pkt_build(buf, KWS_PKT_CMD_PING, 0, 0, 0, g_fn++);
        buf[1] = 0x02;
        uint16_t crc = kws_crc16(&buf[1], KWS_PROTO_HDR_BYTES - 1);
        buf[KWS_PROTO_HDR_BYTES]     = (uint8_t)(crc & 0xFF);
        buf[KWS_PROTO_HDR_BYTES + 1] = (uint8_t)(crc >> 8);
        bfm.send(buf, n);
        drain_tx();
        wait_pkt(KWS_PKT_RSP_ERROR, &p, 300000);
        CHECK(p.payload[0] == KWS_ERR_BAD_VERSION, "bad-version error missing");
    }

    // --- 8. second session: START must behave like a fresh boot ----------------------
    send_cmd_expect_ack(KWS_PKT_CMD_START_STREAM);
    ref.reset();
    unsigned evt_before = evt_matched;
    for (uint32_t fn = 2000; fn < 2000 + KWS_SELFTEST_FRAMES; fn++) {
        send_feature(g_self[fn - 2000], fn, false, true);
    }
    step(60000);
    CHECK(ref.events.empty(), "session 2: reference events missing");
    CHECK(evt_matched > evt_before, "session 2 produced no events");
    printf("  session 2: %u keyword events, all bit-exact\n",
           evt_matched - evt_before);

    return tb_finish("tb_kws_core");
}
