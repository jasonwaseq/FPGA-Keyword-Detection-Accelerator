// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : argmax
// Purpose : Combinational argmax over N signed values. Ties resolve to the
//           lowest index (deterministic, matches the C reference model).
//           Instantiated by the classifier (raw logits) and the temporal
//           smoother (averaged scores).
// -----------------------------------------------------------------------------
`default_nettype none

module argmax #(
  parameter int unsigned N      = 4,
  parameter int unsigned DATA_W = 8
) (
  input  wire  [N-1:0][DATA_W-1:0] values_i,   // signed values, packed
  output logic [$clog2(N)-1:0]     idx_o,
  output logic signed [DATA_W-1:0] max_o
);

  always_comb begin
    idx_o = '0;
    max_o = signed'(values_i[0]);
    for (int unsigned i = 1; i < N; i++) begin
      if (signed'(values_i[i]) > max_o) begin
        max_o = signed'(values_i[i]);
        idx_o = ($clog2(N))'(i);
      end
    end
  end

endmodule : argmax

`default_nettype wire
