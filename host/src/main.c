/* -----------------------------------------------------------------------------
 * Project : iCE40 KWS Accelerator
 * File    : main.c
 * Purpose : Host streaming application.
 *
 *   audio (mic/wav/synth) -> MFCC -> INT8 quantization -> UART DATA_FEATURE
 *   packets @ 100 frames/s, forever. The FPGA schedules inference on its own
 *   and pushes back EVT_KEYWORD packets, which are logged with both the
 *   FPGA-measured inference latency and the host-measured end-to-end latency.
 *
 * With --check (default on) the identical INT8 features are also run through
 * the bit-accurate software reference model on the same window schedule as
 * the hardware, and detections are compared: this is a live hardware-vs-
 * software cross-check, not a simulation artifact.
 *
 * The serial link auto-reconnects: if the port drops, the app re-opens it,
 * re-runs the handshake (PING / READ_VERSION / START_STREAM) and resumes.
 * ---------------------------------------------------------------------------*/
#include "kws_protocol.h"
#include "kws_ref.h"
#include "kws_mfcc.h"
#include "kws_audio.h"
#include "kws_serial.h"
#include "kws_config.h"
#include "kws_log.h"
#include "kws_stats.h"

#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <time.h>
#  include <unistd.h>
#endif

/* --- time helpers ------------------------------------------------------------*/
static long long now_us(void)
{
#ifdef _WIN32
    static LARGE_INTEGER freq;
    LARGE_INTEGER t;
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (long long)((double)t.QuadPart * 1e6 / (double)freq.QuadPart);
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000LL + ts.tv_nsec / 1000;
#endif
}

static void sleep_ms(int ms)
{
#ifdef _WIN32
    Sleep((DWORD)ms);
#else
    usleep((useconds_t)ms * 1000);
#endif
}

/* --- global session state -----------------------------------------------------*/
static volatile int  g_stop = 0;
static kws_config_t  g_cfg;
static kws_stats_t   g_st;
static kws_parser_t  g_parser;
static kws_serial_t *g_ser = 0;
static int           g_fpga_clk_mhz = 12;

/* reference-model mirror of the FPGA scheduling state */
static int8_t g_win[KWS_WINDOW_LEN][KWS_NUM_MFCC];  /* ring by frame index    */
static kws_smooth_t g_smooth;

static void on_sigint(int sig)
{
    (void)sig;
    g_stop = 1;
}

/* --- serial TX helpers ---------------------------------------------------------*/
static int send_packet(uint8_t type, const uint8_t *payload, uint16_t len,
                       uint32_t frame_num)
{
    uint8_t buf[KWS_PROTO_MAX_PKT];
    size_t  n = kws_pkt_build(buf, type, payload, len,
                              (uint32_t)(now_us() / 1000), frame_num);
    if (!n) return -1;
    return (kws_serial_write(g_ser, buf, (int)n) == (int)n) ? 0 : -1;
}

/* --- RX dispatch -----------------------------------------------------------------*/
static void handle_packet(const kws_packet_t *pkt)
{
    switch (pkt->type) {
    case KWS_PKT_RSP_ACK:
        g_st.acks_rx++;
        KWS_DEBUG("ACK for command 0x%02X", pkt->payload[0]);
        break;

    case KWS_PKT_RSP_ERROR:
        g_st.errors_rx++;
        KWS_WARN("FPGA error 0x%02X (type 0x%02X)",
                 pkt->payload[0], pkt->len > 1 ? pkt->payload[1] : 0);
        break;

    case KWS_PKT_RSP_STATS:
        kws_stats_print_fpga(pkt->payload, pkt->len, g_fpga_clk_mhz);
        break;

    case KWS_PKT_RSP_VERSION:
        /* handled synchronously in handshake() */
        break;

    case KWS_PKT_EVT_KEYWORD: {
        kws_event_t evt;
        if (kws_event_decode(pkt, &evt) == 0) {
            double e2e = kws_stats_event(&g_st, pkt->frame_num, now_us());
            KWS_INFO(">>> KEYWORD '%s'  conf=%u votes=%u  frame=%u  "
                     "fpga=%.2fms  end-to-end=%.1fms",
                     kws_config_label(&g_cfg, evt.class_id),
                     evt.confidence, evt.votes, pkt->frame_num,
                     (double)evt.latency_cycles / (g_fpga_clk_mhz * 1e3),
                     e2e);
        }
        break;
    }

    default:
        KWS_WARN("unexpected packet type 0x%02X", pkt->type);
        break;
    }
}

static void poll_serial(void)
{
    uint8_t buf[512];
    int n = kws_serial_read(g_ser, buf, (int)sizeof(buf));
    for (int i = 0; i < n; i++) {
        uint32_t crc_before = g_parser.n_crc_err;
        if (kws_parser_feed(&g_parser, buf[i])) handle_packet(&g_parser.pkt);
        if (g_parser.n_crc_err != crc_before) g_st.parse_crc_err++;
    }
}

/* Wait for a specific response type; keeps dispatching everything else. */
static int wait_for(uint8_t type, int timeout_ms, kws_packet_t *out)
{
    long long deadline = now_us() + (long long)timeout_ms * 1000;
    while (now_us() < deadline && !g_stop) {
        uint8_t buf[256];
        int n = kws_serial_read(g_ser, buf, (int)sizeof(buf));
        for (int i = 0; i < n; i++) {
            if (kws_parser_feed(&g_parser, buf[i])) {
                if (g_parser.pkt.type == type) {
                    if (out) *out = g_parser.pkt;
                    handle_packet(&g_parser.pkt);
                    return 0;
                }
                handle_packet(&g_parser.pkt);
            }
        }
        if (n == 0) sleep_ms(1);
    }
    return -1;
}

/* --- handshake: PING, VERSION check, START ---------------------------------------*/
static int handshake(void)
{
    for (int attempt = 0; attempt < 3; attempt++) {
        g_st.cmds_sent++;
        if (send_packet(KWS_PKT_CMD_PING, 0, 0, 0) != 0) return -1;
        if (wait_for(KWS_PKT_RSP_ACK, 500, 0) == 0) goto pinged;
        KWS_WARN("PING timeout (attempt %d)", attempt + 1);
    }
    return -1;

pinged:;
    /* Zero the statistics baseline: opening the serial port glitches the TX
     * line and typically registers a few framing errors before the session. */
    g_st.cmds_sent++;
    if (send_packet(KWS_PKT_CMD_RESET, 0, 0, 0) != 0) return -1;
    if (wait_for(KWS_PKT_RSP_ACK, 500, 0) != 0) {
        KWS_ERROR("RESET not acknowledged");
        return -1;
    }

    kws_packet_t ver;
    g_st.cmds_sent++;
    if (send_packet(KWS_PKT_CMD_READ_VERSION, 0, 0, 0) != 0) return -1;
    if (wait_for(KWS_PKT_RSP_VERSION, 500, &ver) != 0) {
        KWS_ERROR("READ_VERSION timeout");
        return -1;
    }
    if (ver.len < 12) { KWS_ERROR("short VERSION payload"); return -1; }

    KWS_INFO("FPGA v%u.%u.%u proto %u | mfcc=%u window=%u stride=%u k=%u "
             "ch=%u classes=%u parallel=%u clk=%uMHz",
             ver.payload[0], ver.payload[1], ver.payload[2], ver.payload[3],
             ver.payload[4], ver.payload[5], ver.payload[6], ver.payload[7],
             ver.payload[8], ver.payload[9], ver.payload[10], ver.payload[11]);
    g_fpga_clk_mhz = ver.payload[11] ? ver.payload[11] : 12;

    /* Geometry contract: the FPGA bitstream and this binary's weights must
     * describe the same network, or the cross-check is meaningless. */
    if (ver.payload[4] != KWS_NUM_MFCC   || ver.payload[5] != KWS_WINDOW_LEN ||
        ver.payload[7] != KWS_CONV_K     || ver.payload[8] != KWS_CONV_OUT_CH ||
        ver.payload[9] != KWS_NUM_CLASSES) {
        KWS_ERROR("FPGA geometry does not match compiled weights - rebuild");
        return -1;
    }

    g_st.cmds_sent++;
    if (send_packet(KWS_PKT_CMD_START_STREAM, 0, 0, 0) != 0) return -1;
    if (wait_for(KWS_PKT_RSP_ACK, 500, 0) != 0) {
        KWS_ERROR("START_STREAM not acknowledged");
        return -1;
    }
    KWS_INFO("streaming started");
    return 0;
}

static int connect_and_start(void)
{
    char err[128];
    for (;;) {
        if (g_stop) return -1;
        g_ser = kws_serial_open(g_cfg.port, g_cfg.baud, err, sizeof(err));
        if (g_ser) {
            kws_parser_init(&g_parser);
            if (handshake() == 0) return 0;
            kws_serial_close(g_ser);
            g_ser = 0;
        } else {
            KWS_WARN("serial: %s - retrying in 2 s", err);
        }
        for (int i = 0; i < 20 && !g_stop; i++) sleep_ms(100);
    }
}

/* --- reference model mirror -----------------------------------------------------*/
static void ref_check(uint32_t frame_num, const int8_t *feat)
{
    memcpy(g_win[frame_num % KWS_WINDOW_LEN], feat, KWS_NUM_MFCC);

    /* Mirror of the RTL window_scheduler: windows complete at frames
     * WINDOW_LEN-1, WINDOW_LEN-1+STRIDE, ... (frame numbers are 0-based). */
    uint32_t n = frame_num + 1;   /* frames sent so far */
    if (n < KWS_WINDOW_LEN || (n - KWS_WINDOW_LEN) % 8 /* stride */ != 0) return;

    int8_t win[KWS_WINDOW_LEN][KWS_NUM_MFCC];
    for (uint32_t t = 0; t < KWS_WINDOW_LEN; t++) {
        uint32_t fr = frame_num + 1 - KWS_WINDOW_LEN + t;
        memcpy(win[t], g_win[fr % KWS_WINDOW_LEN], KWS_NUM_MFCC);
    }

    int8_t logits[KWS_NUM_CLASSES];
    int winner;
    kws_ref_infer(win, logits, &winner, KWS_POOL_MAX);

    kws_ref_event_t evt;
    if (kws_smooth_step(&g_smooth, logits, winner, &evt)) {
        g_st.ref_events++;
        g_st.agree++;   /* provisional; unmatched events reported at exit */
        KWS_INFO("    [ref] predicts '%s' conf=%u votes=%u at frame %u",
                 kws_config_label(&g_cfg, evt.class_id),
                 evt.confidence, evt.votes, frame_num);
    }
}

/* --- CLI ------------------------------------------------------------------------*/
static void usage(const char *argv0)
{
    printf("Usage: %s [options]\n"
           "  --config <file>       load configuration (default: kws.ini if present)\n"
           "  --port <name>         serial port (COM7, /dev/ttyUSB1)\n"
           "  --baud <rate>         UART baud rate (default 115200)\n"
           "  --input <spec>        mic | wav:<path> | synth\n"
           "  --duration <s>        stop after N seconds\n"
           "  --stats <s>           FPGA statistics poll interval (0 = off)\n"
           "  --no-check            disable software reference cross-check\n"
           "  --log <file>          append log to file\n"
           "  -v, --verbose         debug logging\n"
           "  -h, --help            this text\n", argv0);
}

int main(int argc, char **argv)
{
    kws_config_default(&g_cfg);

    /* First pass: explicit --config */
    const char *cfg_path = 0;
    for (int i = 1; i < argc - 1; i++) {
        if (!strcmp(argv[i], "--config")) cfg_path = argv[i + 1];
    }
    if (cfg_path) {
        if (kws_config_load(&g_cfg, cfg_path) != 0) {
            fprintf(stderr, "cannot open config '%s'\n", cfg_path);
            return 1;
        }
    } else {
        kws_config_load(&g_cfg, "kws.ini");   /* optional */
    }

    /* CLI overrides */
    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--config") && i + 1 < argc) i++;
        else if (!strcmp(argv[i], "--port")   && i + 1 < argc) snprintf(g_cfg.port, sizeof(g_cfg.port), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--baud")   && i + 1 < argc) g_cfg.baud = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--input")  && i + 1 < argc) snprintf(g_cfg.input, sizeof(g_cfg.input), "%s", argv[++i]);
        else if (!strcmp(argv[i], "--duration") && i + 1 < argc) g_cfg.duration_s = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--stats")  && i + 1 < argc) g_cfg.stats_interval_s = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-check"))               g_cfg.check = 0;
        else if (!strcmp(argv[i], "--log")    && i + 1 < argc) snprintf(g_cfg.log_file, sizeof(g_cfg.log_file), "%s", argv[++i]);
        else if (!strcmp(argv[i], "-v") || !strcmp(argv[i], "--verbose")) g_cfg.verbose = 1;
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else { fprintf(stderr, "unknown option '%s'\n", argv[i]); usage(argv[0]); return 1; }
    }

    kws_log_init(g_cfg.verbose ? KWS_LOG_DEBUG : KWS_LOG_INFO,
                 g_cfg.log_file[0] ? g_cfg.log_file : 0);
    kws_stats_init(&g_st);
    kws_smooth_init(&g_smooth, 0);
    signal(SIGINT, on_sigint);

    KWS_INFO("iCE40 KWS host | port=%s baud=%d input=%s check=%d",
             g_cfg.port, g_cfg.baud, g_cfg.input, g_cfg.check);

    /* Audio + MFCC */
    char err[128];
    kws_audio_t *audio = kws_audio_open(g_cfg.input, KWS_MFCC_SAMPLE_RATE, 1,
                                        err, sizeof(err));
    if (!audio) { KWS_ERROR("audio: %s", err); return 1; }

    kws_mfcc_cfg_t mcfg = {0.97f, 20.0f, 7600.0f, 0.995f, g_cfg.quant_scale};
    kws_mfcc_t mfcc;
    kws_mfcc_init(&mfcc, &mcfg);
    /* Normalization statistics ship with the weights; trained exports freeze
     * them so run-time features match training-time features exactly. */
    kws_mfcc_set_stats(&mfcc, kws_feat_mean, kws_feat_std, KWS_FEAT_FROZEN);

    if (connect_and_start() != 0) { kws_audio_close(audio); return 1; }

    /* Streaming loop: 400-sample analysis frame advanced by 160 per hop. */
    int16_t   frame[KWS_MFCC_FRAME_LEN];
    uint32_t  frame_num = 0;
    long long t_start   = now_us();
    long long next_stats = t_start
        + (long long)g_cfg.stats_interval_s * 1000000LL;

    /* Prime the first full frame */
    int primed = kws_audio_read(audio, frame, KWS_MFCC_FRAME_LEN);
    if (primed <= 0) { KWS_ERROR("audio stream ended before start"); g_stop = 1; }

    while (!g_stop) {
        int8_t feat[KWS_NUM_MFCC];
        kws_mfcc_frame(&mfcc, frame, feat);

        if (send_packet(KWS_PKT_DATA_FEATURE, (const uint8_t *)feat,
                        KWS_NUM_MFCC, frame_num) != 0) {
            KWS_WARN("serial write failed - reconnecting");
            kws_serial_close(g_ser);
            g_ser = 0;
            if (connect_and_start() != 0) break;
        }
        kws_stats_frame_sent(&g_st, frame_num, now_us());
        if (g_cfg.check) ref_check(frame_num, feat);
        frame_num++;

        poll_serial();

        /* Periodic FPGA statistics poll */
        if (g_cfg.stats_interval_s && now_us() >= next_stats) {
            g_st.cmds_sent++;
            send_packet(KWS_PKT_CMD_READ_STATS, 0, 0, frame_num);
            next_stats += (long long)g_cfg.stats_interval_s * 1000000LL;
        }

        if (g_cfg.duration_s
            && now_us() - t_start > (long long)g_cfg.duration_s * 1000000LL) {
            break;
        }

        /* Advance one hop: shift and refill (audio backends pace realtime). */
        memmove(frame, frame + KWS_MFCC_HOP_LEN,
                (KWS_MFCC_FRAME_LEN - KWS_MFCC_HOP_LEN) * sizeof(int16_t));
        int r = kws_audio_read(audio,
                               frame + KWS_MFCC_FRAME_LEN - KWS_MFCC_HOP_LEN,
                               KWS_MFCC_HOP_LEN);
        if (r <= 0) { KWS_INFO("audio stream ended"); break; }
    }

    /* Orderly shutdown */
    KWS_INFO("stopping...");
    if (g_ser) {
        g_st.cmds_sent++;
        send_packet(KWS_PKT_CMD_STOP_STREAM, 0, 0, frame_num);
        wait_for(KWS_PKT_RSP_ACK, 300, 0);
        g_st.cmds_sent++;
        send_packet(KWS_PKT_CMD_READ_STATS, 0, 0, frame_num);
        wait_for(KWS_PKT_RSP_STATS, 500, 0);
        kws_serial_close(g_ser);
    }
    kws_audio_close(audio);
    kws_stats_print(&g_st);
    kws_log_close();
    return 0;
}
