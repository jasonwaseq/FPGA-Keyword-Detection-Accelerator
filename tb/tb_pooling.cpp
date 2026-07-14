// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Bench   : tb_pooling - pooling_engine vs kws_ref_pool, bit-exact
// Checks  : max and average modes over random signed activations (full INT8
//           range, not just post-ReLU values), address coverage, repeated
//           runs with mode switching.
// -----------------------------------------------------------------------------
#include "tb_util.h"
#include "Vpooling_engine.h"
#include "kws_ref.h"

int main(int argc, char **argv)
{
    Harness<Vpooling_engine> h(argc, argv, "sim/out/tb_pooling.vcd");
    Rng rng(0x900Cull);

    SyncRead<uint32_t, uint8_t> act_ram(KWS_CONV_OUT_LEN * KWS_CONV_OUT_CH);

    h.dut->start_i = 0;
    h.dut->mode_i  = 0;
    h.reset();

    for (int run = 0; run < 16; run++) {
        int mode = run & 1;   // alternate max / avg

        int8_t act[KWS_CONV_OUT_LEN][KWS_CONV_OUT_CH];
        for (int t = 0; t < KWS_CONV_OUT_LEN; t++)
            for (int c = 0; c < KWS_CONV_OUT_CH; c++) {
                act[t][c] = rng.i8();
                act_ram.mem[(size_t)(t * KWS_CONV_OUT_CH + c)] = (uint8_t)act[t][c];
            }

        int8_t pooled[KWS_POOL_OUT_LEN * KWS_CONV_OUT_CH];
        bool   written[KWS_POOL_OUT_LEN * KWS_CONV_OUT_CH] = {false};

        h.dut->mode_i  = (uint8_t)mode;
        h.dut->start_i = 1;

        unsigned guard = 0;
        bool done = false;
        while (!done) {
            h.tick([&] {
                h.dut->start_i = 0;
                h.dut->act_rd_data_i = act_ram.step(h.dut->act_rd_addr_o);
            });
            if (h.dut->pool_wr_en_o) {
                unsigned a = h.dut->pool_wr_addr_o;
                CHECK(a < KWS_POOL_OUT_LEN * KWS_CONV_OUT_CH,
                      "pool address %u out of range", a);
                pooled[a]  = (int8_t)h.dut->pool_wr_data_o;
                written[a] = true;
            }
            if (h.dut->done_o) done = true;
            CHECK(++guard < 50000, "pooling hung (run %d)", run);
            if (guard >= 50000) break;
        }

        int8_t ref[KWS_POOL_OUT_LEN][KWS_CONV_OUT_CH];
        kws_ref_pool(act, ref, mode ? KWS_POOL_AVG : KWS_POOL_MAX);
        for (int p = 0; p < KWS_POOL_OUT_LEN; p++)
            for (int c = 0; c < KWS_CONV_OUT_CH; c++) {
                int a = p * KWS_CONV_OUT_CH + c;
                CHECK(written[a], "run %d out[%d][%d] never written", run, p, c);
                CHECK(pooled[a] == ref[p][c],
                      "run %d mode %d out[%d][%d]: dut %d != ref %d",
                      run, mode, p, c, pooled[a], ref[p][c]);
            }
    }

    return tb_finish("tb_pooling");
}
