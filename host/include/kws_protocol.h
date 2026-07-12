/* -----------------------------------------------------------------------------
 * Project : iCE40 KWS Accelerator
 * File    : kws_protocol.h
 * Purpose : Wire protocol definition and packet pack/parse API, shared by the
 *           host application and the Verilator testbenches. Mirrors
 *           rtl/kws_pkg.sv exactly; the full-system testbench cross-checks
 *           the two via the RSP_VERSION payload.
 *
 * Frame (little-endian multi-byte fields):
 *   [SOF][VER][TYPE][LEN_L][LEN_H][TS0..3][FN0..3][PAYLOAD*LEN][CRC_L][CRC_H]
 *   CRC16-CCITT-FALSE (poly 0x1021, init 0xFFFF) over VER..payload end.
 * ---------------------------------------------------------------------------*/
#ifndef KWS_PROTOCOL_H
#define KWS_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KWS_PROTO_SOF          0xA5u
#define KWS_PROTO_VERSION      0x01u
#define KWS_PROTO_HDR_BYTES    13u
#define KWS_PROTO_MAX_PAYLOAD  64u
#define KWS_PROTO_MAX_PKT      (KWS_PROTO_HDR_BYTES + KWS_PROTO_MAX_PAYLOAD + 2u)

/* Host -> FPGA */
#define KWS_PKT_CMD_PING          0x01u
#define KWS_PKT_CMD_RESET         0x02u
#define KWS_PKT_CMD_START_STREAM  0x03u
#define KWS_PKT_CMD_STOP_STREAM   0x04u
#define KWS_PKT_CMD_READ_STATS    0x05u
#define KWS_PKT_CMD_READ_VERSION  0x06u
#define KWS_PKT_DATA_FEATURE      0x10u

/* FPGA -> Host */
#define KWS_PKT_RSP_ACK           0x81u
#define KWS_PKT_RSP_ERROR         0x82u
#define KWS_PKT_RSP_STATS         0x83u
#define KWS_PKT_RSP_VERSION       0x84u
#define KWS_PKT_EVT_KEYWORD       0x90u

/* RSP_ERROR payload byte 0 */
#define KWS_ERR_BAD_LENGTH        0x01u
#define KWS_ERR_UNKNOWN_TYPE      0x02u
#define KWS_ERR_BAD_VERSION       0x03u

/* Statistics word indices (RSP_STATS payload, 16 x u32 LE) */
enum kws_stat_idx {
    KWS_STAT_FRAMES_RX = 0,  KWS_STAT_FRAMES_DROPPED, KWS_STAT_PKTS_RX,
    KWS_STAT_CRC_ERRORS,     KWS_STAT_FRAMING_ERRORS, KWS_STAT_FIFO_OVERFLOWS,
    KWS_STAT_WINDOWS_SCHED,  KWS_STAT_INFERENCES,     KWS_STAT_KEYWORDS,
    KWS_STAT_LAT_LAST,       KWS_STAT_LAT_MAX,        KWS_STAT_LAT_ACC,
    KWS_STAT_CONV_BUSY,      KWS_STAT_UPTIME_CYC,     KWS_STAT_WINDOWS_DROP,
    KWS_STAT_TX_PKTS,        KWS_STAT_NUM
};

/* A parsed packet */
typedef struct {
    uint8_t  type;
    uint16_t len;
    uint32_t timestamp;
    uint32_t frame_num;
    uint8_t  payload[KWS_PROTO_MAX_PAYLOAD];
} kws_packet_t;

/* EVT_KEYWORD payload layout */
typedef struct {
    uint8_t  class_id;
    uint8_t  confidence;
    uint8_t  votes;
    uint32_t latency_cycles;
} kws_event_t;

/* Incremental parser (mirror of rtl/packet_decoder.sv) */
typedef struct {
    int      state;          /* internal */
    uint16_t crc;
    uint16_t idx;
    uint8_t  crc_lo;
    kws_packet_t pkt;
    /* statistics */
    uint32_t n_ok;
    uint32_t n_crc_err;
    uint32_t n_desync;
} kws_parser_t;

uint16_t kws_crc16(const uint8_t *buf, size_t len);

/* Build a packet into buf (>= KWS_PROTO_MAX_PKT bytes). Returns total size,
 * or 0 if len exceeds KWS_PROTO_MAX_PAYLOAD. payload may be NULL if len==0. */
size_t kws_pkt_build(uint8_t *buf, uint8_t type,
                     const uint8_t *payload, uint16_t len,
                     uint32_t timestamp, uint32_t frame_num);

void kws_parser_init(kws_parser_t *p);

/* Feed one byte; returns 1 when p->pkt holds a complete CRC-verified packet. */
int kws_parser_feed(kws_parser_t *p, uint8_t byte);

/* Decode an EVT_KEYWORD payload. Returns 0 on success. */
int kws_event_decode(const kws_packet_t *pkt, kws_event_t *evt);

#ifdef __cplusplus
}
#endif
#endif /* KWS_PROTOCOL_H */
