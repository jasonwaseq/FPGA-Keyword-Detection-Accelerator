// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Bench   : tb_uart_rx - receiver vs the UART BFM driver
// Checks  : byte reception with random inter-byte gaps, glitch rejection
//           (short low pulse must not produce a byte), break-condition framing
//           errors, recovery after corruption.
// -----------------------------------------------------------------------------
#include "tb_util.h"
#include "Vuart_rx.h"

static const unsigned CPB = 12000000 / 115200;   // 104

int main(int argc, char **argv)
{
    Harness<Vuart_rx> h(argc, argv, "sim/out/tb_uart_rx.vcd");
    Rng rng(0x0A17ull);
    UartBfm bfm(CPB);

    h.dut->rxd_i = 1;
    h.reset();

    std::deque<uint8_t> expected;
    unsigned rx_count = 0, ferr_count = 0;

    auto run = [&](unsigned cycles) {
        for (unsigned i = 0; i < cycles; i++) {
            h.dut->rxd_i = (uint8_t)bfm.drive();
            h.tick();
            if (h.dut->valid_o) {
                CHECK(!expected.empty(), "unexpected byte 0x%02x", h.dut->data_o);
                if (!expected.empty()) {
                    CHECK(h.dut->data_o == expected.front(),
                          "got 0x%02x expected 0x%02x",
                          h.dut->data_o, expected.front());
                    expected.pop_front();
                }
                rx_count++;
            }
            if (h.dut->frame_err_o) ferr_count++;
        }
    };

    // Phase 1: 300 random bytes, random gaps
    for (int i = 0; i < 300; i++) {
        uint8_t b = (uint8_t)rng.u32(0, 255);
        expected.push_back(b);
        bfm.idle_gap = rng.u32(0, 3 * CPB);
        bfm.send(&b, 1);
        run(14 * CPB);
    }
    CHECK(expected.empty(), "%zu bytes were never received", expected.size());
    CHECK(rx_count == 300, "received %u of 300 bytes", rx_count);
    CHECK(ferr_count == 0, "spurious framing errors: %u", ferr_count);

    // Phase 2: glitches shorter than the majority filter must be ignored
    for (int i = 0; i < 20; i++) {
        for (int g = 0; g < 3; g++) { h.dut->rxd_i = 0; h.tick(); }
        h.dut->rxd_i = 1;
        run(12 * CPB);
    }
    CHECK(rx_count == 300, "glitch produced a byte (count %u)", rx_count);

    // Phase 3: break condition (line low for 15 bit times) -> framing error,
    // then clean recovery. Standard UART semantics: the first 10 bit times of
    // the break complete a frame with a low stop bit (framing error); the
    // receiver then re-arms inside the break and may assemble one garbage
    // byte as the line returns high - tolerated and flushed here.
    ferr_count = 0;
    unsigned garbage = 0;
    h.dut->rxd_i = 0;
    for (unsigned i = 0; i < 15 * CPB; i++) {
        h.tick();
        if (h.dut->frame_err_o) ferr_count++;
        if (h.dut->valid_o)     garbage++;
    }
    h.dut->rxd_i = 1;
    for (unsigned i = 0; i < 15 * CPB; i++) {   // flush partial break frame
        h.tick();
        if (h.dut->frame_err_o) ferr_count++;
        if (h.dut->valid_o)     garbage++;
    }
    CHECK(ferr_count >= 1, "break did not raise a framing error");
    printf("  break: %u framing errors, %u garbage bytes (flushed)\n",
           ferr_count, garbage);

    uint8_t b = 0x5A;
    expected.push_back(b);
    bfm.send(&b, 1);
    run(14 * CPB);
    CHECK(expected.empty(), "no reception after break recovery");

    return tb_finish("tb_uart_rx");
}
