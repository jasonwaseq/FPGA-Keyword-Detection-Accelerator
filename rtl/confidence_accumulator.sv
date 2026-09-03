// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : confidence_accumulator
// Purpose : Per-class moving average of classifier logits over the last DEPTH
//           inferences, maintained as running sums with O(1) update:
//
//             sum[c] += logit[c] - hist[head][c];  hist[head][c] = logit[c]
//
//           avg_o[c] = sum[c] >>> log2(DEPTH)  (combinational, always current)
//
// DEPTH must be a power of two so the average is a shift. Storage is
// DEPTH * N bytes of flip-flops (256 FF at defaults) - deliberately not EBR:
// all N sums update in the same cycle, which a 1R+1W block RAM cannot serve.
// -----------------------------------------------------------------------------
`default_nettype none

module confidence_accumulator #(
  parameter int unsigned N      = 4,
  parameter int unsigned DATA_W = 8,
  parameter int unsigned DEPTH  = 8    // power of two
) (
  input  wire                      clk_i,
  input  wire                      rst_ni,
  input  wire                      clear_i,
  input  wire                      update_i,
  input  wire  [N-1:0][DATA_W-1:0] logits_i,
  output logic [N-1:0][DATA_W-1:0] avg_o     // signed per-class averages
);

  localparam int unsigned SH    = $clog2(DEPTH);
  localparam int unsigned SUM_W = DATA_W + SH;

  logic signed [DATA_W-1:0] hist_q [DEPTH][N];
  logic signed [SUM_W-1:0]  sum_q  [N];
  logic [SH-1:0]            head_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      head_q <= '0;
      for (int c = 0; c < N; c++) sum_q[c] <= '0;
      for (int d = 0; d < DEPTH; d++)
        for (int c = 0; c < N; c++) hist_q[d][c] <= '0;
    end else if (clear_i) begin
      head_q <= '0;
      for (int c = 0; c < N; c++) sum_q[c] <= '0;
      for (int d = 0; d < DEPTH; d++)
        for (int c = 0; c < N; c++) hist_q[d][c] <= '0;
    end else if (update_i) begin
      for (int c = 0; c < N; c++) begin
        sum_q[c] <= sum_q[c]
                  + SUM_W'(signed'(logits_i[c]))
                  - SUM_W'(hist_q[head_q][c]);
        hist_q[head_q][c] <= signed'(logits_i[c]);
      end
      head_q <= head_q + 1'b1;  // natural wrap
    end
  end

  always_comb begin
    for (int c = 0; c < N; c++) begin
      avg_o[c] = DATA_W'(sum_q[c] >>> SH);
    end
  end

`ifndef SYNTHESIS
  initial begin
    assert (DEPTH == (1 << SH))
      else $error("confidence_accumulator: DEPTH must be a power of two");
  end
`endif

endmodule : confidence_accumulator

`default_nettype wire
