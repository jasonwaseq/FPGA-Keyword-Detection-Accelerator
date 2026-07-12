/* -----------------------------------------------------------------------------
 * Project : iCE40 KWS Accelerator
 * File    : protocol.c
 * Purpose : Packet pack/parse implementation. The parser is a deliberate
 *           mirror of rtl/packet_decoder.sv (same states, same resync rules)
 *           so host and hardware can never disagree about framing.
 * ---------------------------------------------------------------------------*/
#include "kws_protocol.h"
#include <string.h>

uint16_t kws_crc16(const uint8_t *buf, size_t len)
{
    uint16_t crc = 0xFFFFu;
    for (size_t i = 0; i < len; i++) {
        crc ^= (uint16_t)((uint16_t)buf[i] << 8);
        for (int b = 0; b < 8; b++) {
            crc = (crc & 0x8000u) ? (uint16_t)((crc << 1) ^ 0x1021u)
                                  : (uint16_t)(crc << 1);
        }
    }
    return crc;
}

size_t kws_pkt_build(uint8_t *buf, uint8_t type,
                     const uint8_t *payload, uint16_t len,
                     uint32_t timestamp, uint32_t frame_num)
{
    if (len > KWS_PROTO_MAX_PAYLOAD) return 0;

    buf[0]  = KWS_PROTO_SOF;
    buf[1]  = KWS_PROTO_VERSION;
    buf[2]  = type;
    buf[3]  = (uint8_t)(len & 0xFFu);
    buf[4]  = (uint8_t)(len >> 8);
    buf[5]  = (uint8_t)(timestamp);
    buf[6]  = (uint8_t)(timestamp >> 8);
    buf[7]  = (uint8_t)(timestamp >> 16);
    buf[8]  = (uint8_t)(timestamp >> 24);
    buf[9]  = (uint8_t)(frame_num);
    buf[10] = (uint8_t)(frame_num >> 8);
    buf[11] = (uint8_t)(frame_num >> 16);
    buf[12] = (uint8_t)(frame_num >> 24);
    if (len) memcpy(&buf[KWS_PROTO_HDR_BYTES], payload, len);

    uint16_t crc = kws_crc16(&buf[1], (size_t)(KWS_PROTO_HDR_BYTES - 1 + len));
    buf[KWS_PROTO_HDR_BYTES + len]     = (uint8_t)(crc & 0xFFu);
    buf[KWS_PROTO_HDR_BYTES + len + 1] = (uint8_t)(crc >> 8);
    return (size_t)(KWS_PROTO_HDR_BYTES + len + 2);
}

/* Parser states (match rtl/packet_decoder.sv) */
enum { P_HUNT, P_VER, P_TYPE, P_LEN_L, P_LEN_H, P_TS, P_FN,
       P_PAYLOAD, P_CRC_L, P_CRC_H };

static uint16_t crc_step(uint16_t crc, uint8_t d)
{
    crc ^= (uint16_t)((uint16_t)d << 8);
    for (int b = 0; b < 8; b++) {
        crc = (crc & 0x8000u) ? (uint16_t)((crc << 1) ^ 0x1021u)
                              : (uint16_t)(crc << 1);
    }
    return crc;
}

void kws_parser_init(kws_parser_t *p)
{
    memset(p, 0, sizeof(*p));
    p->state = P_HUNT;
}

int kws_parser_feed(kws_parser_t *p, uint8_t byte)
{
    switch (p->state) {
    case P_HUNT:
        if (byte == KWS_PROTO_SOF) {
            p->crc   = 0xFFFFu;
            p->state = P_VER;
        }
        return 0;

    case P_VER:
        p->crc = crc_step(p->crc, byte);
        if (byte != KWS_PROTO_VERSION) { p->state = P_HUNT; p->n_desync++; return 0; }
        p->state = P_TYPE;
        return 0;

    case P_TYPE:
        p->crc = crc_step(p->crc, byte);
        p->pkt.type = byte;
        p->state = P_LEN_L;
        return 0;

    case P_LEN_L:
        p->crc = crc_step(p->crc, byte);
        p->pkt.len = byte;
        p->state = P_LEN_H;
        return 0;

    case P_LEN_H:
        p->crc = crc_step(p->crc, byte);
        p->pkt.len |= (uint16_t)((uint16_t)byte << 8);
        if (p->pkt.len > KWS_PROTO_MAX_PAYLOAD) {
            p->state = P_HUNT;
            p->n_desync++;
            return 0;
        }
        p->idx = 0;
        p->pkt.timestamp = 0;
        p->state = P_TS;
        return 0;

    case P_TS:
        p->crc = crc_step(p->crc, byte);
        p->pkt.timestamp |= (uint32_t)byte << (8 * p->idx);
        if (++p->idx == 4) { p->idx = 0; p->pkt.frame_num = 0; p->state = P_FN; }
        return 0;

    case P_FN:
        p->crc = crc_step(p->crc, byte);
        p->pkt.frame_num |= (uint32_t)byte << (8 * p->idx);
        if (++p->idx == 4) {
            p->idx   = 0;
            p->state = (p->pkt.len == 0) ? P_CRC_L : P_PAYLOAD;
        }
        return 0;

    case P_PAYLOAD:
        p->crc = crc_step(p->crc, byte);
        p->pkt.payload[p->idx] = byte;
        if (++p->idx == p->pkt.len) p->state = P_CRC_L;
        return 0;

    case P_CRC_L:
        p->crc_lo = byte;
        p->state  = P_CRC_H;
        return 0;

    case P_CRC_H:
        p->state = P_HUNT;
        if (((uint16_t)byte << 8 | p->crc_lo) == p->crc) {
            p->n_ok++;
            return 1;
        }
        p->n_crc_err++;
        return 0;

    default:
        p->state = P_HUNT;
        return 0;
    }
}

int kws_event_decode(const kws_packet_t *pkt, kws_event_t *evt)
{
    if (pkt->type != KWS_PKT_EVT_KEYWORD || pkt->len != 8) return -1;
    evt->class_id       = pkt->payload[0];
    evt->confidence     = pkt->payload[1];
    evt->votes          = pkt->payload[2];
    evt->latency_cycles = (uint32_t)pkt->payload[4]
                        | (uint32_t)pkt->payload[5] << 8
                        | (uint32_t)pkt->payload[6] << 16
                        | (uint32_t)pkt->payload[7] << 24;
    return 0;
}
