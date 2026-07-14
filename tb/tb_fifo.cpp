// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Bench   : tb_fifo - uart_fifo vs a C++ deque model
// Checks  : randomized concurrent push/pop, level tracking, full/empty flags,
//           overflow-drop semantics (write while full is discarded and
//           flagged), data integrity across wrap-around.
// -----------------------------------------------------------------------------
#include "tb_util.h"
#include "Vuart_fifo.h"

static const unsigned DEPTH = 128;   // default parameter

int main(int argc, char **argv)
{
    Harness<Vuart_fifo> h(argc, argv, "sim/out/tb_fifo.vcd");
    Rng rng(0xF1F0ull);

    h.dut->wr_en_i = 0;
    h.dut->rd_en_i = 0;
    h.reset();

    std::deque<uint8_t> model;
    unsigned overflows_seen = 0, overflows_expected = 0;
    bool     pop_pending = false;
    uint8_t  pop_expect  = 0;

    for (int cycle = 0; cycle < 200000; cycle++) {
        // Bias phases: fill-heavy, drain-heavy, mixed - exercises wrap + full.
        double p_wr = (cycle / 20000) % 3 == 0 ? 0.9
                    : (cycle / 20000) % 3 == 1 ? 0.2 : 0.5;

        bool do_wr = rng.chance(p_wr);
        bool do_rd = rng.chance(0.5) && !h.dut->empty_o && !pop_pending;

        uint8_t wdata = (uint8_t)rng.u32(0, 255);
        h.dut->wr_en_i   = do_wr;
        h.dut->wr_data_i = wdata;
        h.dut->rd_en_i   = do_rd;

        bool was_full = h.dut->full_o;

        if (do_wr) {
            if (was_full) overflows_expected++;
            else          model.push_back(wdata);
        }
        if (do_rd) {
            pop_pending = true;
            pop_expect  = model.front();
            model.pop_front();
        }

        h.tick();
        h.dut->wr_en_i = 0;
        h.dut->rd_en_i = 0;

        if (h.dut->overflow_o) overflows_seen++;
        if (h.dut->rd_valid_o) {
            CHECK(pop_pending, "spurious rd_valid");
            CHECK(h.dut->rd_data_o == pop_expect,
                  "pop 0x%02x expected 0x%02x", h.dut->rd_data_o, pop_expect);
            pop_pending = false;
        }

        CHECK(h.dut->level_o == model.size() + (pop_pending ? 0 : 0),
              "level %u != model %zu", h.dut->level_o, model.size());
        CHECK(h.dut->empty_o == (model.empty()),
              "empty flag mismatch (level %u)", h.dut->level_o);
        CHECK(h.dut->full_o == (model.size() == DEPTH), "full flag mismatch");
    }

    CHECK(overflows_seen == overflows_expected,
          "overflow count %u != expected %u",
          overflows_seen, overflows_expected);
    CHECK(overflows_expected > 0, "test never exercised overflow");

    return tb_finish("tb_fifo");
}
