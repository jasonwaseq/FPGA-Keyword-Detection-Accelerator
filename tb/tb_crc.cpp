// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Bench   : tb_crc - crc16 module vs the C reference (kws_crc16)
// Checks  : random-length random-content byte streams, re-seed via clear_i,
//           byte-at-a-time folding equivalence.
// -----------------------------------------------------------------------------
#include "tb_util.h"
#include "Vcrc16.h"
#include "kws_protocol.h"

int main(int argc, char **argv)
{
    Harness<Vcrc16> h(argc, argv, "sim/out/tb_crc.vcd");
    Rng rng(0xC5C5C5C5ull);

    h.dut->clear_i = 0;
    h.dut->en_i    = 0;
    h.dut->data_i  = 0;
    h.reset();

    for (int iter = 0; iter < 200; iter++) {
        // Re-seed
        h.dut->clear_i = 1;
        h.tick();
        h.dut->clear_i = 0;

        int len = (int)rng.u32(0, 96);
        std::vector<uint8_t> buf((size_t)len);
        for (auto &b : buf) b = (uint8_t)rng.u32(0, 255);

        for (uint8_t b : buf) {
            h.dut->en_i   = 1;
            h.dut->data_i = b;
            h.tick();
            h.dut->en_i = 0;
            // random idle gaps must not disturb the CRC
            for (unsigned g = rng.u32(0, 2); g; g--) h.tick();
        }
        h.tick();

        uint16_t exp = kws_crc16(buf.data(), buf.size());
        CHECK(h.dut->crc_o == exp,
              "iter %d len %d: crc %04x != expected %04x",
              iter, len, h.dut->crc_o, exp);
    }

    return tb_finish("tb_crc");
}
