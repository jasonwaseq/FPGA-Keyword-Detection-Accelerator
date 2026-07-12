// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : kws_top
// Purpose : iCEBreaker v1.1a board wrapper: clocking, reset conditioning,
//           UART pin hookup and LED/IRQ user I/O around kws_core.
//
// Clocking: the 12 MHz board oscillator drives the fabric directly - no PLL.
// Rationale: the workload needs ~2% of the compute budget at 12 MHz (see
// docs/pipeline.md), UART throughput is the true bottleneck, and skipping the
// PLL removes a timing variable and an iCE40 hard-macro dependency. The core
// is parameterized on CLK_FREQ_HZ if a PLL is introduced later.
//
// I/O map (constraints/icebreaker.pcf):
//   CLK 12 MHz          pin 35        BTN_N (user reset)   pin 10
//   UART RX (from FTDI) pin 6         UART TX (to FTDI)    pin 9
//   LEDR_N (error)      pin 11        LEDG_N (keyword/HB)  pin 37
//   IRQ (PMOD1A pin 1)  pin 4
//
// The green LED shows a ~1.5 Hz heartbeat while streaming is active and goes
// solid for ~200 ms on a keyword event; red flashes on link errors. Board
// LEDs are active low.
// -----------------------------------------------------------------------------
`default_nettype none

module kws_top #(
  parameter int unsigned CLK_HZ    = kws_pkg::CLK_FREQ_HZ,
  parameter int unsigned BAUD_RATE = kws_pkg::UART_BAUD
) (
  input  wire  CLK,      // 12 MHz oscillator
  input  wire  BTN_N,    // user button, active low: manual reset
  input  wire  RX,       // UART from host (FTDI)
  output logic TX,       // UART to host (FTDI)
  output logic LEDR_N,   // red LED, active low: link error
  output logic LEDG_N,   // green LED, active low: heartbeat / keyword
  output logic P1A1      // PMOD 1A pin 1: IRQ to external observer
);

  // Reset conditioning: power-on stretch + synchronized button release.
  logic rst_n;
  reset_sync #(
    .STAGES  (4),
    .POR_LEN (4095)
  ) u_rst (
    .clk_i   (CLK),
    .arst_ni (BTN_N),
    .rst_no  (rst_n)
  );

  logic irq, led_keyword, led_error, stream_active;

  kws_core #(
    .CLK_HZ    (CLK_HZ),
    .BAUD_RATE (BAUD_RATE)
  ) u_core (
    .clk_i           (CLK),
    .rst_ni          (rst_n),
    .uart_rxd_i      (RX),
    .uart_txd_o      (TX),
    .irq_o           (irq),
    .led_keyword_o   (led_keyword),
    .led_error_o     (led_error),
    .stream_active_o (stream_active)
  );

  // Heartbeat: ~1.5 Hz from a free-running divider while streaming.
  localparam int unsigned HB_W = 23;  // 2^23 / 12 MHz ~= 0.7 s period
  logic [HB_W-1:0] hb_q;
  always_ff @(posedge CLK or negedge rst_n) begin
    if (!rst_n) hb_q <= '0;
    else        hb_q <= hb_q + 1'b1;
  end

  assign LEDG_N = ~(led_keyword | (stream_active & hb_q[HB_W-1]));
  assign LEDR_N = ~led_error;
  assign P1A1   = irq;

endmodule : kws_top

`default_nettype wire
