// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// File    : tb_util.h
// Purpose : Shared Verilator testbench harness: clocking with optional VCD
//           trace, deterministic RNG, check macros, a UART bus-functional
//           model (driver + monitor) and a synchronous-read memory emulator
//           matching rom_sync / ram_dp_sync timing.
//
// All benches are self-checking: they print PASS/FAIL and return a nonzero
// exit code on failure (consumed by the regression runner and CI).
// -----------------------------------------------------------------------------
#ifndef TB_UTIL_H
#define TB_UTIL_H

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <string>
#include <vector>

#include "verilated.h"
#include "verilated_vcd_c.h"

// Legacy time hook referenced by the Verilator runtime (one bench per binary,
// so a single definition here is safe).
double sc_time_stamp() { return 0.0; }

// --- deterministic RNG (xorshift64*) ----------------------------------------
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    uint64_t next() {
        s ^= s >> 12; s ^= s << 25; s ^= s >> 27;
        return s * 0x2545F4914F6CDD1Dull;
    }
    uint32_t u32(uint32_t lo, uint32_t hi) {           // inclusive range
        uint64_t span = (uint64_t)hi - lo + 1;         // 64-bit: full-range safe
        return lo + (uint32_t)(next() % span);
    }
    int8_t  i8()  { return (int8_t)next(); }
    bool    chance(double p) { return (next() >> 11) * (1.0 / 9007199254740992.0) < p; }
};

// --- error accounting ---------------------------------------------------------
static int tb_errors = 0;

#define CHECK(cond, ...)                                                     \
    do {                                                                     \
        if (!(cond)) {                                                       \
            printf("FAIL %s:%d: ", __FILE__, __LINE__);                      \
            printf(__VA_ARGS__);                                             \
            printf("\n");                                                    \
            if (++tb_errors > 20) {                                          \
                printf("too many errors, aborting\n");                       \
                exit(1);                                                     \
            }                                                                \
        }                                                                    \
    } while (0)

static inline int tb_finish(const char *name)
{
    if (tb_errors == 0) { printf("PASS %s\n", name); return 0; }
    printf("FAIL %s (%d errors)\n", name, tb_errors);
    return 1;
}

// --- clocked harness ------------------------------------------------------------
// DUTs use ports clk_i / rst_ni. tick() = one full clock; on_posedge callback
// runs right after the rising edge (registered outputs stable) and may set
// inputs for the next cycle (synchronous-read emulation).
template <class DUT>
struct Harness {
    DUT           *dut;
    VerilatedVcdC *vcd   = nullptr;
    uint64_t       t     = 0;

    explicit Harness(int argc, char **argv, const char *vcd_name = nullptr) {
        Verilated::commandArgs(argc, argv);
        Verilated::assertOn(true);
        dut = new DUT;
        bool trace = false;
        for (int i = 1; i < argc; i++)
            if (!strcmp(argv[i], "+trace")) trace = true;
        if (trace && vcd_name) {
            Verilated::traceEverOn(true);
            vcd = new VerilatedVcdC;
            dut->trace(vcd, 99);
            vcd->open(vcd_name);
        }
    }
    ~Harness() {
        if (vcd) vcd->close();
        dut->final();
        delete dut;
    }

    template <class F>
    void tick(F on_posedge) {
        dut->clk_i = 1;
        dut->eval();
        on_posedge();
        dut->eval();
        if (vcd) vcd->dump(10 * t + 5);
        dut->clk_i = 0;
        dut->eval();
        if (vcd) vcd->dump(10 * t + 10);
        t++;
    }
    void tick() { tick([] {}); }

    void reset(int cycles = 5) {
        dut->rst_ni = 0;
        for (int i = 0; i < cycles; i++) tick();
        dut->rst_ni = 1;
        tick();
    }
};

// --- synchronous-read memory emulator ---------------------------------------------
// Mirrors rom_sync / ram_dp_sync: data_o <= mem[addr_i] at the clock edge, so
// a registered address set at edge N yields data the DUT samples at edge N+2.
// Call step() from the on_posedge callback with the DUT's current address,
// then assign the return value to the DUT's data input.
template <typename ADDR_T, typename DATA_T>
struct SyncRead {
    std::vector<DATA_T> mem;
    DATA_T pipe = 0;
    explicit SyncRead(size_t depth) : mem(depth, 0) {}
    DATA_T step(ADDR_T addr) {
        DATA_T out = pipe;
        pipe = (addr < mem.size()) ? mem[addr] : 0;
        return out;
    }
};

// --- UART bus-functional model ----------------------------------------------------
// Bit-banged 8N1 driver/monitor advanced once per clock via drive()/monitor().
struct UartBfm {
    unsigned clks_per_bit;
    // driver
    std::deque<uint8_t> tx_q;
    int      d_state = 0;      // 0 idle, 1 shifting
    unsigned d_cnt   = 0;
    unsigned d_bit   = 0;
    uint16_t d_shift = 0;
    unsigned idle_gap = 0;     // extra idle clocks between bytes
    // monitor
    int      m_state = 0;
    unsigned m_cnt   = 0;
    unsigned m_bit   = 0;
    uint8_t  m_shift = 0;
    std::deque<uint8_t> rx_q;

    explicit UartBfm(unsigned cpb) : clks_per_bit(cpb) {}

    void send(const uint8_t *buf, size_t n) {
        for (size_t i = 0; i < n; i++) tx_q.push_back(buf[i]);
    }

    // Returns the rxd line value for this cycle.
    int drive() {
        switch (d_state) {
        case 0:
            if (d_cnt > 0) { d_cnt--; return 1; }      // inter-byte gap
            if (tx_q.empty()) return 1;
            d_shift = (uint16_t)((1u << 9) | ((uint16_t)tx_q.front() << 1));
            tx_q.pop_front();                          // start(0) data stop(1)
            d_bit   = 0;
            d_cnt   = 0;
            d_state = 1;
            /* fallthrough */
        case 1: {
            int lvl = (d_shift >> d_bit) & 1;
            if (++d_cnt >= clks_per_bit) {
                d_cnt = 0;
                if (++d_bit == 10) { d_state = 0; d_cnt = idle_gap; }
            }
            return lvl;
        }
        default: return 1;
        }
    }

    // Feed the txd line value each cycle; received bytes appear in rx_q.
    void monitor(int txd) {
        switch (m_state) {
        case 0:
            if (!txd) { m_state = 1; m_cnt = 0; }
            break;
        case 1:                                        // confirm start mid-bit
            if (++m_cnt >= clks_per_bit / 2) {
                if (!txd) { m_state = 2; m_cnt = 0; m_bit = 0; m_shift = 0; }
                else      { m_state = 0; }
            }
            break;
        case 2:                                        // data bits
            if (++m_cnt >= clks_per_bit) {
                m_cnt = 0;
                m_shift = (uint8_t)((m_shift >> 1) | (txd ? 0x80 : 0));
                if (++m_bit == 8) m_state = 3;
            }
            break;
        case 3:                                        // stop bit
            if (++m_cnt >= clks_per_bit) {
                if (txd) rx_q.push_back(m_shift);
                m_state = 0;
                m_cnt   = 0;
            }
            break;
        default: m_state = 0;
        }
    }

    bool busy() const { return d_state != 0 || !tx_q.empty(); }
};

#endif // TB_UTIL_H
