// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Bench   : tb_uart_tx - transmitter vs the UART BFM monitor
// Checks  : valid/ready handshake, byte integrity, back-to-back streaming,
//           line idles high.
// -----------------------------------------------------------------------------
#include "tb_util.h"
#include "Vuart_tx.h"

static const unsigned CPB = 12000000 / 115200;   // 104

int main(int argc, char **argv)
{
    Harness<Vuart_tx> h(argc, argv, "sim/out/tb_uart_tx.vcd");
    Rng rng(0x7E57ull);
    UartBfm bfm(CPB);

    h.dut->valid_i = 0;
    h.dut->data_i  = 0;
    h.reset();

    std::deque<uint8_t> sent;

    for (int i = 0; i < 300; i++) {
        uint8_t b = (uint8_t)rng.u32(0, 255);

        // present the byte, wait for acceptance
        h.dut->data_i  = b;
        h.dut->valid_i = 1;
        unsigned guard = 0;
        while (!(h.dut->ready_o && h.dut->valid_i)) {
            bfm.monitor(h.dut->txd_o);
            h.tick();
            CHECK(++guard < 20 * CPB, "ready_o never asserted");
            if (guard >= 20 * CPB) break;
        }
        bfm.monitor(h.dut->txd_o);
        h.tick();               // transfer cycle
        h.dut->valid_i = 0;
        sent.push_back(b);

        // random spacing: sometimes queue immediately (back-to-back)
        unsigned gap = rng.chance(0.5) ? 0 : rng.u32(1, 3 * CPB);
        for (unsigned g = 0; g < gap; g++) {
            bfm.monitor(h.dut->txd_o);
            h.tick();
        }
    }

    // drain
    for (unsigned i = 0; i < 15 * CPB; i++) {
        bfm.monitor(h.dut->txd_o);
        h.tick();
    }

    CHECK(bfm.rx_q.size() == sent.size(),
          "monitor captured %zu of %zu bytes", bfm.rx_q.size(), sent.size());
    size_t n = std::min(bfm.rx_q.size(), sent.size());
    for (size_t i = 0; i < n; i++) {
        CHECK(bfm.rx_q[i] == sent[i], "byte %zu: 0x%02x != 0x%02x",
              i, bfm.rx_q[i], sent[i]);
    }
    CHECK(h.dut->txd_o == 1, "line does not idle high");

    return tb_finish("tb_uart_tx");
}
