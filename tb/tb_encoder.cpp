// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Bench   : tb_encoder - packet_encoder byte stream vs the C packet builder
// Checks  : exact wire bytes (header, payload, CRC) for random requests of
//           every length 0..64, correctness under random TX-FIFO
//           back-pressure, request handshake behaviour.
// -----------------------------------------------------------------------------
#include "tb_util.h"
#include "Vpacket_encoder.h"
#include "kws_protocol.h"

int main(int argc, char **argv)
{
    Harness<Vpacket_encoder> h(argc, argv, "sim/out/tb_encoder.vcd");
    Rng rng(0xE2C0DEull);

    h.dut->req_valid_i = 0;
    h.dut->tx_full_i   = 0;
    h.dut->pl_data_i   = 0;
    h.reset();

    uint8_t payload[KWS_PROTO_MAX_PAYLOAD];
    uint8_t pipe = 0;   // 1-deep pipe => 2-cycle payload fetch latency

    for (int iter = 0; iter < 120; iter++) {
        uint16_t len = (iter <= 64) ? (uint16_t)iter : (uint16_t)rng.u32(0, 64);
        uint8_t  type = (uint8_t)rng.u32(0x80, 0x9F);
        uint32_t ts = rng.u32(0, 0xFFFFFFFF), fn = rng.u32(0, 0xFFFFFFFF);
        for (int i = 0; i < len; i++) payload[i] = (uint8_t)rng.u32(0, 255);

        double p_full = (iter % 3 == 0) ? 0.0 : 0.4;   // back-pressure phases

        // issue request
        h.dut->req_type_i      = type;
        h.dut->req_len_i       = (uint8_t)len;
        h.dut->req_timestamp_i = ts;
        h.dut->req_frame_i     = fn;
        h.dut->req_valid_i     = 1;

        std::vector<uint8_t> wire;
        bool done = false;
        unsigned guard = 0;
        while (!done) {
            bool accepted = h.dut->req_ready_o && h.dut->req_valid_i;
            h.tick([&] {
                // payload source emulation (registered index, 1 pipe stage)
                uint8_t idx = h.dut->pl_idx_o;
                h.dut->pl_data_i = pipe;
                pipe = (idx < len) ? payload[idx] : 0;
                h.dut->tx_full_i = rng.chance(p_full);
            });
            if (accepted) h.dut->req_valid_i = 0;
            // NB: tx_wr_en is only honored by the FIFO when not full; the
            // encoder guarantees it never asserts wr_en while full held.
            if (h.dut->tx_wr_en_o) wire.push_back(h.dut->tx_wr_data_o);
            if (h.dut->pkt_done_o) done = true;
            CHECK(++guard < 20000, "encoder hung (iter %d len %u)", iter, len);
            if (guard >= 20000) break;
        }

        uint8_t exp[KWS_PROTO_MAX_PKT];
        size_t  exp_n = kws_pkt_build(exp, type, payload, len, ts, fn);
        CHECK(wire.size() == exp_n, "iter %d: %zu bytes, expected %zu",
              iter, wire.size(), exp_n);
        for (size_t i = 0; i < std::min(wire.size(), exp_n); i++) {
            CHECK(wire[i] == exp[i], "iter %d byte %zu: %02x != %02x",
                  iter, i, wire[i], exp[i]);
        }

        // a couple of idle cycles between packets
        for (unsigned g = rng.u32(0, 4); g; g--) h.tick();
    }

    return tb_finish("tb_encoder");
}
