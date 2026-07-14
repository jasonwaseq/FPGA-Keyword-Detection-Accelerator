// Temporary bring-up probe: sends one PING and watches the RX chain.
#include "tb_util.h"
#include "Vkws_core.h"
#include "Vkws_core___024root.h"
#include "kws_protocol.h"

static const unsigned CPB = 16;

int main(int argc, char **argv)
{
    Harness<Vkws_core> h(argc, argv, nullptr);
    UartBfm bfm(CPB);

    h.dut->uart_rxd_i = 1;
    h.reset(10);

    uint8_t buf[KWS_PROTO_MAX_PKT];
    size_t n = kws_pkt_build(buf, KWS_PKT_CMD_PING, 0, 0, 0x11223344, 0x55);
    bfm.send(buf, n);
    printf("sending %zu bytes\n", n);

    auto *r = h.dut->rootp;
    unsigned rx_valid = 0, ferr = 0, commits = 0, cmds = 0, crc_errs = 0,
             proto_errs = 0, pkt_oks = 0;
    int last_dec_state = -1, last_enc_state = -1;
    unsigned last_level = 0;

    for (unsigned c = 0; c < 60000; c++) {
        h.dut->uart_rxd_i = (uint8_t)bfm.drive();
        h.tick();
        if (r->kws_core__DOT__rx_valid) rx_valid++;
        if (r->kws_core__DOT__rx_ferr)  ferr++;
        if (r->kws_core__DOT__cmd_valid) cmds++;
        if (r->kws_core__DOT__pkt_ok)   pkt_oks++;
        if (r->kws_core__DOT__crc_err)  crc_errs++;
        if (r->kws_core__DOT__proto_err) proto_errs++;
        unsigned lvl = r->kws_core__DOT__rxf_level;
        int ds = r->kws_core__DOT__u_dec__DOT__state_q;
        int es = r->kws_core__DOT__u_enc__DOT__state_q;
        if (ds != last_dec_state || lvl != last_level) {
            printf("[%6u] dec_state=%d rxf_level=%u rx_valid=%u\n",
                   c, ds, lvl, rx_valid);
            last_dec_state = ds;
            last_level = lvl;
        }
        if (es != last_enc_state) {
            printf("[%6u] enc_state=%d rsp_req=%d arb=%d\n",
                   c, es, (int)r->kws_core__DOT__rsp_req,
                   (int)r->kws_core__DOT__arb_q);
            last_enc_state = es;
        }
    }
    printf("rx_valid=%u ferr=%u pkt_ok=%u cmds=%u crc_err=%u proto_err=%u\n",
           rx_valid, ferr, pkt_oks, cmds, crc_errs, proto_errs);
    return 0;
}
