// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Bench   : tb_conv1d - conv1d_engine vs kws_ref_conv1d, bit-exact
// Checks  : full-window convolution results (requantized + ReLU) for random
//           feature windows at random circular-buffer bases including
//           wrap-around, across multiple back-to-back windows. Memory
//           interfaces (feature RAM, kernel banks, bias/parameter ROM) are
//           emulated in C++ with exact rom_sync/ram_dp_sync timing.
//
// Built at PARALLEL = 1, 2 and 4 (PARAM_PARALLEL define + -GPARALLEL).
// -----------------------------------------------------------------------------
#include "tb_util.h"
#include "Vconv1d_engine.h"
#include "kws_ref.h"

#ifndef PARAM_PARALLEL
#define PARAM_PARALLEL 2
#endif

static const int P     = PARAM_PARALLEL;
static const int KA_W  = 10;                       // clog2(8*3*40 = 960)
static const int HISTD = 64;

int main(int argc, char **argv)
{
    Harness<Vconv1d_engine> h(argc, argv, "sim/out/tb_conv1d.vcd");
    Rng rng(0xC04Dull + P);

    // --- memory emulation ------------------------------------------------------
    // feature RAM: {frame[5:0], coef[5:0]} -> 4096 bytes
    SyncRead<uint32_t, uint8_t> feat_ram(HISTD * 64);
    // kernel banks: each holds the full flattened array
    SyncRead<uint32_t, uint8_t> krn[4] = {
        SyncRead<uint32_t, uint8_t>(960), SyncRead<uint32_t, uint8_t>(960),
        SyncRead<uint32_t, uint8_t>(960), SyncRead<uint32_t, uint8_t>(960)
    };
    SyncRead<uint32_t, uint32_t> bias_rom(KWS_CONV_OUT_CH + 2);

    for (int oc = 0; oc < KWS_CONV_OUT_CH; oc++)
        for (int k = 0; k < KWS_CONV_K; k++)
            for (int ic = 0; ic < KWS_NUM_MFCC; ic++)
                for (int b = 0; b < P; b++)
                    krn[b].mem[(size_t)(oc * KWS_CONV_K + k) * KWS_NUM_MFCC + ic] =
                        (uint8_t)kws_conv_w[oc][k][ic];
    for (int oc = 0; oc < KWS_CONV_OUT_CH; oc++)
        bias_rom.mem[oc] = (uint32_t)kws_conv_b[oc];
    bias_rom.mem[KWS_CONV_OUT_CH]     = (uint32_t)KWS_M_CONV;
    bias_rom.mem[KWS_CONV_OUT_CH + 1] = (uint32_t)KWS_S_CONV;

    h.dut->start_i = 0;
    h.reset();

    int8_t act_ram[KWS_CONV_OUT_LEN * KWS_CONV_OUT_CH];
    bool   act_written[KWS_CONV_OUT_LEN * KWS_CONV_OUT_CH];

    for (int win = 0; win < 12; win++) {
        // Random window at a random base (exercises circular wrap)
        int base = (int)rng.u32(0, HISTD - 1);
        int8_t feat[KWS_WINDOW_LEN][KWS_NUM_MFCC];
        for (int t = 0; t < KWS_WINDOW_LEN; t++)
            for (int ic = 0; ic < KWS_NUM_MFCC; ic++) {
                feat[t][ic] = rng.i8();
                feat_ram.mem[(size_t)(((base + t) % HISTD) << 6 | ic)] =
                    (uint8_t)feat[t][ic];
            }

        memset(act_written, 0, sizeof(act_written));

        h.dut->win_base_i = (uint8_t)base;
        h.dut->start_i    = 1;

        unsigned guard = 0;
        bool done = false;
        while (!done) {
            h.tick([&] {
                h.dut->start_i = 0;
                // feature RAM (2-cycle total: registered addr + sync read)
                uint32_t fa = ((uint32_t)h.dut->feat_frame_o << 6)
                            | h.dut->feat_coef_o;
                h.dut->feat_data_i = feat_ram.step(fa);
                // kernel banks (packed [P][KA_W] address, [P][8] data)
                uint64_t ka = h.dut->krn_addr_o;
                uint64_t kd = 0;
                for (int b = 0; b < P; b++) {
                    uint32_t a = (uint32_t)((ka >> (KA_W * b)) & ((1u << KA_W) - 1));
                    kd |= (uint64_t)krn[b].step(a) << (8 * b);
                }
                if (P == 1)      h.dut->krn_data_i = (uint8_t)kd;
                else if (P == 2) h.dut->krn_data_i = (uint16_t)kd;
                else             h.dut->krn_data_i = (uint32_t)kd;
                // bias ROM
                h.dut->bias_data_i = bias_rom.step(h.dut->bias_addr_o);
            });
            if (h.dut->act_wr_en_o) {
                unsigned a = h.dut->act_wr_addr_o;
                CHECK(a < KWS_CONV_OUT_LEN * KWS_CONV_OUT_CH,
                      "activation address %u out of range", a);
                act_ram[a]     = (int8_t)h.dut->act_wr_data_o;
                act_written[a] = true;
            }
            if (h.dut->done_o) done = true;
            CHECK(++guard < 200000, "engine hung (window %d)", win);
            if (guard >= 200000) break;
        }
        CHECK(!h.dut->busy_o, "busy after done");

        // Compare against the reference
        int8_t ref[KWS_CONV_OUT_LEN][KWS_CONV_OUT_CH];
        kws_ref_conv1d(feat, ref);
        for (int t = 0; t < KWS_CONV_OUT_LEN; t++) {
            for (int oc = 0; oc < KWS_CONV_OUT_CH; oc++) {
                int a = t * KWS_CONV_OUT_CH + oc;
                CHECK(act_written[a], "win %d out[%d][%d] never written",
                      win, t, oc);
                CHECK(act_ram[a] == ref[t][oc],
                      "win %d out[%d][%d]: dut %d != ref %d",
                      win, t, oc, act_ram[a], ref[t][oc]);
            }
        }
    }

    printf("tb_conv1d: PARALLEL=%d, 12 windows bit-exact\n", P);
    return tb_finish("tb_conv1d");
}
