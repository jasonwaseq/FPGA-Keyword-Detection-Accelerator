// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Bench   : tb_smoothing - temporal_smoothing (+ confidence_accumulator,
//           argmax) vs kws_smooth_step, bit-exact decision equivalence
// Checks  : detection timing, class/confidence/votes payload equality,
//           threshold / vote / consecutive / debounce gating, target mask,
//           enable, across several configurations and stimulus profiles
//           (random noise, keyword bursts, alternating classes).
// -----------------------------------------------------------------------------
#include "tb_util.h"
#include "Vtemporal_smoothing.h"
#include "kws_ref.h"

struct Cfg {
    int8_t thresh; uint8_t vote_min, min_consec, debounce, mask, enable;
};

int main(int argc, char **argv)
{
    Harness<Vtemporal_smoothing> h(argc, argv, "sim/out/tb_smoothing.vcd");
    Rng rng(0x500full);

    // vote_min <= SMOOTH_DEPTH (4): higher values can never be satisfied.
    const Cfg cfgs[] = {
        { 25, 2, 1, 12, 0x0C, 1 },   // shipped defaults (tuned operating point)
        { 15, 2, 1,  4, 0x0C, 1 },   // permissive
        { 70, 4, 2, 20, 0x08, 1 },   // strict, class 3 only
        { 25, 2, 1, 12, 0x0C, 0 },   // disabled: must never fire
        {  0, 0, 0,  0, 0x0F, 1 },   // degenerate: everything fires
    };

    h.dut->update_i = 0;
    h.dut->clear_i  = 0;

    for (size_t ci = 0; ci < sizeof(cfgs) / sizeof(cfgs[0]); ci++) {
        const Cfg &c = cfgs[ci];

        // (Re)initialize DUT and reference
        h.reset();
        h.dut->en_i          = c.enable;
        h.dut->thresh_i      = (uint8_t)c.thresh;
        h.dut->vote_min_i    = c.vote_min;
        h.dut->min_consec_i  = c.min_consec;
        h.dut->debounce_i    = c.debounce;
        h.dut->target_mask_i = c.mask;

        kws_smooth_t ref;
        kws_smooth_cfg_t rc = { c.thresh, c.vote_min, c.min_consec,
                                c.debounce, c.mask, c.enable };
        kws_smooth_init(&ref, &rc);

        unsigned fired_dut = 0, fired_ref = 0;

        for (int step = 0; step < 600; step++) {
            // Stimulus phases: noise / class-2 burst / class-3 burst / alternate
            int phase = (step / 75) % 4;
            int8_t logits[KWS_NUM_CLASSES];
            for (int k = 0; k < KWS_NUM_CLASSES; k++)
                logits[k] = (int8_t)rng.u32(0, 40) - 20;
            if (phase == 1) logits[2] = (int8_t)rng.u32(70, 127);
            if (phase == 2) logits[3] = (int8_t)rng.u32(70, 127);
            if (phase == 3) logits[(step & 1) ? 2 : 3] = (int8_t)rng.u32(70, 127);

            int winner = kws_ref_argmax(logits, KWS_NUM_CLASSES);

            uint32_t packed = 0;
            for (int k = 0; k < KWS_NUM_CLASSES; k++)
                packed |= (uint32_t)(uint8_t)logits[k] << (8 * k);
            h.dut->logits_i = packed;
            h.dut->winner_i = (uint8_t)winner;
            h.dut->update_i = 1;
            h.tick();
            h.dut->update_i = 0;

            // Reference decision for this inference
            kws_ref_event_t rev;
            int rfire = kws_smooth_step(&ref, logits, winner, &rev);

            // DUT evaluates one cycle after the update; give it two.
            int dfire = 0;
            uint8_t dclass = 0, dconf = 0, dvotes = 0;
            for (int w = 0; w < 2; w++) {
                h.tick();
                if (h.dut->detect_o) {
                    dfire  = 1;
                    dclass = h.dut->det_class_o;
                    dconf  = h.dut->det_conf_o;
                    dvotes = h.dut->det_votes_o;
                }
            }

            CHECK(dfire == rfire, "cfg %zu step %d: dut fire=%d ref fire=%d",
                  ci, step, dfire, rfire);
            if (dfire && rfire) {
                fired_dut++; fired_ref++;
                CHECK(dclass == rev.class_id, "cfg %zu step %d class %u != %u",
                      ci, step, dclass, rev.class_id);
                CHECK(dconf == rev.confidence, "cfg %zu step %d conf %u != %u",
                      ci, step, dconf, rev.confidence);
                CHECK(dvotes == rev.votes, "cfg %zu step %d votes %u != %u",
                      ci, step, dvotes, rev.votes);
            }
        }

        if (c.enable && c.thresh <= 70) {
            CHECK(fired_ref > 0, "cfg %zu produced no detections - stimulus "
                  "too weak to be meaningful", ci);
        }
        if (!c.enable) {
            CHECK(fired_dut == 0, "cfg %zu fired while disabled", ci);
        }
        printf("  cfg %zu: %u detections, all bit-exact\n", ci, fired_dut);
    }

    return tb_finish("tb_smoothing");
}
