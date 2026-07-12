// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : uart_fifo
// Purpose : Synchronous FIFO with registered (1-cycle) read, sized to map onto
//           iCE40 EBR (4 Kbit block RAM).
//
// Read protocol: assert rd_en_i while empty_o is low; rd_data_o is valid on
// the following cycle, flagged by rd_valid_o. Maximum sustained pop rate is
// therefore one word per two cycles - ample for this design, where both FIFO
// consumers are paced by the 115200-baud UART (~104 clocks per byte). A
// first-word-fall-through wrapper was considered and rejected: it costs an
// output register stage and prefetch control for throughput nothing here
// needs (see docs/architecture.md, "FIFO design").
//
// Write protocol: wr_en_i with wr_data_i; a write while full is dropped and
// pulses overflow_o (the producer - a UART line - cannot be back-pressured).
// -----------------------------------------------------------------------------
`default_nettype none

module uart_fifo #(
  parameter int unsigned WIDTH = 8,
  parameter int unsigned DEPTH = 128   // must be a power of two
) (
  input  wire              clk_i,
  input  wire              rst_ni,
  // Write port
  input  wire              wr_en_i,
  input  wire  [WIDTH-1:0] wr_data_i,
  output logic             full_o,
  output logic             overflow_o,   // 1-cycle strobe: write dropped
  // Read port
  input  wire              rd_en_i,
  output logic [WIDTH-1:0] rd_data_o,
  output logic             rd_valid_o,   // rd_data_o valid (cycle after rd_en_i)
  output logic             empty_o,
  // Status
  output logic [$clog2(DEPTH):0] level_o
);

  localparam int unsigned AW = $clog2(DEPTH);

  logic [WIDTH-1:0] mem [DEPTH];
  logic [AW-1:0]    wr_ptr_q, rd_ptr_q;
  logic [AW:0]      level_q;

  wire do_wr = wr_en_i && !full_o;
  wire do_rd = rd_en_i && !empty_o;

  assign full_o  = (level_q == (AW+1)'(DEPTH));
  assign empty_o = (level_q == '0);
  assign level_o = level_q;

  // Storage: synchronous write, synchronous read (maps to EBR)
  always_ff @(posedge clk_i) begin
    if (do_wr) mem[wr_ptr_q] <= wr_data_i;
    rd_data_o <= mem[rd_ptr_q];
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wr_ptr_q   <= '0;
      rd_ptr_q   <= '0;
      level_q    <= '0;
      rd_valid_o <= 1'b0;
      overflow_o <= 1'b0;
    end else begin
      rd_valid_o <= do_rd;
      overflow_o <= wr_en_i && full_o;

      if (do_wr) wr_ptr_q <= wr_ptr_q + 1'b1;
      if (do_rd) rd_ptr_q <= rd_ptr_q + 1'b1;

      unique case ({do_wr, do_rd})
        2'b10:   level_q <= level_q + 1'b1;
        2'b01:   level_q <= level_q - 1'b1;
        default: ;  // 00 or simultaneous 11: level unchanged
      endcase
    end
  end

`ifndef SYNTHESIS
  initial begin
    assert (DEPTH == (1 << AW))
      else $error("uart_fifo: DEPTH=%0d is not a power of two", DEPTH);
  end
  always_ff @(posedge clk_i) begin
    assert (!(rd_en_i && empty_o))
      else $error("uart_fifo: read while empty");
  end
`endif

endmodule : uart_fifo

`default_nettype wire
