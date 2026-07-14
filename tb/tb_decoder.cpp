// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Bench   : tb_decoder - packet_decoder with an emulated RX FIFO
// Checks  : clean feature/command/error dispatch, speculative-write indices,
//           CRC rejection (single-bit corruption anywhere in a packet),
//           resynchronization through garbage, oversized-LEN recovery,
//           semantic errors (bad version / bad length / unknown type),
//           inter-byte timeout (verilated with TIMEOUT_CYCLES=2000).
// -----------------------------------------------------------------------------
#include "tb_util.h"
#include "Vpacket_decoder.h"
#include "kws_protocol.h"

struct FifoEmu {
    std::deque<uint8_t> q;
    void push(const uint8_t *b, size_t n) { q.insert(q.end(), b, b + n); }
};

int main(int argc, char **argv)
{
    Harness<Vpacket_decoder> h(argc, argv, "sim/out/tb_decoder.vcd");
    Rng rng(0xDEC0DEull);
    FifoEmu fifo;

    h.dut->fifo_empty_i    = 1;
    h.dut->fifo_rd_valid_i = 0;
    h.reset();

    // Counters mirrored from DUT strobes
    unsigned n_commit = 0, n_cmd = 0, n_err = 0, n_ok = 0, n_crc = 0, n_proto = 0;
    uint8_t  last_cmd = 0, last_err_code = 0, last_err_detail = 0;
    uint32_t last_fn = 0, last_ts = 0;
    int8_t   feat_cap[64];
    unsigned feat_wr_seen = 0;

    auto step = [&](unsigned cycles) {
        for (unsigned i = 0; i < cycles; i++) {
            // rd_en is combinational on current inputs: sample pre-edge.
            int rd_now = h.dut->fifo_rd_en_o;
            h.tick([&] {
                h.dut->fifo_rd_valid_i = 0;
                if (rd_now && !fifo.q.empty()) {
                    h.dut->fifo_rd_data_i  = fifo.q.front();
                    fifo.q.pop_front();
                    h.dut->fifo_rd_valid_i = 1;
                }
                h.dut->fifo_empty_i = fifo.q.empty();
            });
            if (h.dut->feat_wr_en_o) {
                CHECK(h.dut->feat_wr_idx_o < 40, "feature index %u out of range",
                      h.dut->feat_wr_idx_o);
                feat_cap[h.dut->feat_wr_idx_o] = (int8_t)h.dut->feat_wr_data_o;
                feat_wr_seen++;
            }
            if (h.dut->feat_commit_o) {
                n_commit++;
                last_fn = h.dut->feat_frame_num_o;
                last_ts = h.dut->feat_timestamp_o;
            }
            if (h.dut->cmd_valid_o) { n_cmd++; last_cmd = h.dut->cmd_type_o; }
            if (h.dut->err_valid_o) {
                n_err++;
                last_err_code   = h.dut->err_code_o;
                last_err_detail = h.dut->err_detail_o;
            }
            if (h.dut->pkt_ok_o)    n_ok++;
            if (h.dut->crc_err_o)   n_crc++;
            if (h.dut->proto_err_o) n_proto++;
        }
    };

    uint8_t pkt[KWS_PROTO_MAX_PKT];

    // --- 1. clean feature packets --------------------------------------------
    for (int f = 0; f < 20; f++) {
        int8_t feat[40];
        for (auto &v : feat) v = rng.i8();
        size_t n = kws_pkt_build(pkt, KWS_PKT_DATA_FEATURE,
                                 (uint8_t *)feat, 40, 0x1000u + f, 100u + f);
        fifo.push(pkt, n);
        step((unsigned)n + 20);
        CHECK(n_commit == (unsigned)(f + 1), "commit missing for frame %d", f);
        CHECK(last_fn == 100u + f && last_ts == 0x1000u + f,
              "frame metadata wrong: fn=%u ts=%u", last_fn, last_ts);
        for (int i = 0; i < 40; i++) {
            CHECK(feat_cap[i] == feat[i], "frame %d coef %d: %d != %d",
                  f, i, feat_cap[i], feat[i]);
        }
    }
    CHECK(feat_wr_seen == 20 * 40, "speculative write count %u", feat_wr_seen);

    // --- 2. all commands -------------------------------------------------------
    const uint8_t cmds[] = { KWS_PKT_CMD_PING, KWS_PKT_CMD_RESET,
                             KWS_PKT_CMD_START_STREAM, KWS_PKT_CMD_STOP_STREAM,
                             KWS_PKT_CMD_READ_STATS, KWS_PKT_CMD_READ_VERSION };
    for (uint8_t c : cmds) {
        unsigned before = n_cmd;
        size_t n = kws_pkt_build(pkt, c, 0, 0, 7, 8);
        fifo.push(pkt, n);
        step((unsigned)n + 20);
        CHECK(n_cmd == before + 1 && last_cmd == c,
              "command 0x%02x not dispatched", c);
    }

    // --- 3. CRC corruption: flip one random bit in each packet -----------------
    unsigned commits_before = n_commit;
    for (int f = 0; f < 30; f++) {
        int8_t feat[40];
        for (auto &v : feat) v = rng.i8();
        size_t n = kws_pkt_build(pkt, KWS_PKT_DATA_FEATURE,
                                 (uint8_t *)feat, 40, 1, 1);
        unsigned pos = rng.u32(1, (uint32_t)n - 1);   // anywhere after SOF
        pkt[pos] ^= (uint8_t)(1u << rng.u32(0, 7));
        unsigned crc_before = n_crc, proto_before = n_proto,
                 err_before = n_err;
        fifo.push(pkt, n);
        step((unsigned)n + 40);
        // Every outcome must be a rejection: no commit is ever allowed.
        CHECK(n_commit == commits_before,
              "corrupted packet %d committed a frame!", f);
        (void)crc_before; (void)proto_before; (void)err_before;
        // Recovery: a clean packet must still get through afterwards.
        size_t m = kws_pkt_build(pkt, KWS_PKT_DATA_FEATURE,
                                 (uint8_t *)feat, 40, 2, 2);
        fifo.push(pkt, m);
        step((unsigned)m + 60);
        CHECK(n_commit == commits_before + 1,
              "no recovery after corrupted packet %d", f);
        commits_before = n_commit;
    }
    CHECK(n_crc > 0, "corruption never produced a CRC error (n_crc=%u)", n_crc);

    // --- 4. garbage resync ------------------------------------------------------
    uint8_t junk[64];
    for (auto &j : junk) j = (uint8_t)rng.u32(0, 255);
    fifo.push(junk, sizeof(junk));
    size_t n = kws_pkt_build(pkt, KWS_PKT_CMD_PING, 0, 0, 0, 0);
    fifo.push(pkt, n);
    // The junk may open a false frame that swallows the first real packet;
    // send a second one to prove bounded-loss resync.
    fifo.push(pkt, n);
    unsigned cmd_before = n_cmd;
    step(600);
    CHECK(n_cmd > cmd_before, "no resync after garbage");

    // --- 5. oversized LEN -> proto error + resync -------------------------------
    n = kws_pkt_build(pkt, KWS_PKT_CMD_PING, 0, 0, 0, 0);
    pkt[4] = 0x7F;   // LEN_H: length 0x7F00 > MAX_PAYLOAD
    unsigned proto_before = n_proto;
    fifo.push(pkt, n);
    fifo.push(pkt, n);   // pkt[4] corrupt in both copies
    // rebuild clean and append
    n = kws_pkt_build(pkt, KWS_PKT_CMD_PING, 0, 0, 0, 0);
    fifo.push(pkt, n);
    cmd_before = n_cmd;
    step(600);
    CHECK(n_proto > proto_before, "oversized LEN not flagged");
    CHECK(n_cmd > cmd_before, "no recovery after oversized LEN");

    // --- 6. semantic errors ------------------------------------------------------
    //   unknown type
    n = kws_pkt_build(pkt, 0x55, 0, 0, 0, 0);
    fifo.push(pkt, n);
    step((unsigned)n + 40);
    CHECK(last_err_code == KWS_ERR_UNKNOWN_TYPE && last_err_detail == 0x55,
          "unknown type: err %02x detail %02x", last_err_code, last_err_detail);
    //   bad length (feature with 39 bytes)
    {
        uint8_t short_feat[39] = {0};
        n = kws_pkt_build(pkt, KWS_PKT_DATA_FEATURE, short_feat, 39, 0, 0);
        unsigned commit_b = n_commit;
        fifo.push(pkt, n);
        step((unsigned)n + 40);
        CHECK(last_err_code == KWS_ERR_BAD_LENGTH, "bad length not flagged");
        CHECK(n_commit == commit_b, "short feature packet committed");
    }
    //   bad version (patch VER byte, fix CRC accordingly)
    {
        n = kws_pkt_build(pkt, KWS_PKT_CMD_PING, 0, 0, 0, 0);
        pkt[1] = 0x02;
        uint16_t crc = kws_crc16(&pkt[1], KWS_PROTO_HDR_BYTES - 1);
        pkt[KWS_PROTO_HDR_BYTES]     = (uint8_t)(crc & 0xFF);
        pkt[KWS_PROTO_HDR_BYTES + 1] = (uint8_t)(crc >> 8);
        fifo.push(pkt, n);
        step((unsigned)n + 40);
        CHECK(last_err_code == KWS_ERR_BAD_VERSION, "bad version not flagged");
    }

    // --- 7. inter-byte timeout (TIMEOUT_CYCLES=2000 in this build) ---------------
    {
        n = kws_pkt_build(pkt, KWS_PKT_CMD_PING, 0, 0, 0, 0);
        fifo.push(pkt, 6);                    // truncated mid-header
        unsigned pb = n_proto;
        step(3000);                            // exceed timeout
        CHECK(n_proto == pb + 1, "timeout not flagged (proto %u)", n_proto);
        // full packet afterwards parses cleanly
        cmd_before = n_cmd;
        fifo.push(pkt, n);
        step((unsigned)n + 40);
        CHECK(n_cmd == cmd_before + 1, "no recovery after timeout");
    }

    return tb_finish("tb_decoder");
}
