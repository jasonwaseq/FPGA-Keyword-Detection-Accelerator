// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : timestamp_gen
// Purpose : Free-running time bases: a 32-bit millisecond counter (packet
//           timestamps) and a 32-bit raw cycle counter (latency measurement,
//           utilization statistics).
//
// Both counters wrap naturally; consumers use unsigned modular differences.
// ms rollover period at 2^32 ms is ~49.7 days; cycle rollover at 12 MHz is
// ~358 s and is documented in docs/protocol.md for the statistics fields.
// -----------------------------------------------------------------------------
`default_nettype none

module timestamp_gen #(
  parameter int unsigned CLK_FREQ_HZ = 12_000_000
) (
  input  wire         clk_i,
  input  wire         rst_ni,
  output logic [31:0] ms_o,      // milliseconds since reset
  output logic [31:0] cycles_o   // clock cycles since reset
);

  localparam int unsigned CYC_PER_MS = CLK_FREQ_HZ / 1000;
  localparam int unsigned DW = $clog2(CYC_PER_MS);

  logic [DW-1:0] div_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      div_q    <= '0;
      ms_o     <= '0;
      cycles_o <= '0;
    end else begin
      cycles_o <= cycles_o + 32'd1;
      if (div_q == DW'(CYC_PER_MS - 1)) begin
        div_q <= '0;
        ms_o  <= ms_o + 32'd1;
      end else begin
        div_q <= div_q + 1'b1;
      end
    end
  end

endmodule : timestamp_gen

`default_nettype wire
