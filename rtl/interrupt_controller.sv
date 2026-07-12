// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : interrupt_controller
// Purpose : Event aggregation for out-of-band observers. Latches sticky
//           pending flags per event class, drives a level IRQ output (routed
//           to a PMOD pin for an external MCU or logic analyzer), and
//           generates human-visible pulse-stretched LED outputs.
//
// Pending flags are cleared by ack_i (pulsed by the register file on
// READ_STATS, so a host polling statistics implicitly services the IRQ) or
// by clear_i (soft reset). LED stretch defaults to ~200 ms at 12 MHz.
// -----------------------------------------------------------------------------
`default_nettype none

module interrupt_controller #(
  parameter int unsigned STRETCH_CYCLES = 2_400_000,  // ~200 ms @ 12 MHz
  parameter logic  [2:0] IRQ_EN_MASK    = 3'b111      // {overflow, error, keyword}
) (
  input  wire        clk_i,
  input  wire        rst_ni,
  input  wire        clear_i,

  // Event inputs (1-cycle strobes)
  input  wire        keyword_i,
  input  wire        error_i,      // CRC / framing / protocol
  input  wire        overflow_i,   // RX FIFO drop

  // Service
  input  wire        ack_i,        // clear all pending flags

  // Outputs
  output logic       irq_o,        // level: any enabled pending flag
  output logic [2:0] pending_o,    // {overflow, error, keyword}
  output logic       led_keyword_o,
  output logic       led_error_o
);

  localparam int unsigned CW = $clog2(STRETCH_CYCLES + 1);

  logic [CW-1:0] kw_cnt_q, err_cnt_q;

  // Sticky pending flags
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      pending_o <= '0;
    end else if (clear_i || ack_i) begin
      pending_o <= {overflow_i, error_i, keyword_i};  // don't lose same-cycle events
    end else begin
      pending_o <= pending_o | {overflow_i, error_i, keyword_i};
    end
  end

  assign irq_o = |(pending_o & IRQ_EN_MASK);

  // LED pulse stretchers
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      kw_cnt_q  <= '0;
      err_cnt_q <= '0;
    end else begin
      if (keyword_i)           kw_cnt_q <= CW'(STRETCH_CYCLES);
      else if (kw_cnt_q != '0) kw_cnt_q <= kw_cnt_q - 1'b1;

      if (error_i || overflow_i) err_cnt_q <= CW'(STRETCH_CYCLES);
      else if (err_cnt_q != '0)  err_cnt_q <= err_cnt_q - 1'b1;
    end
  end

  assign led_keyword_o = (kw_cnt_q != '0);
  assign led_error_o   = (err_cnt_q != '0);

endmodule : interrupt_controller

`default_nettype wire
