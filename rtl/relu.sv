// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : relu
// Purpose : Combinational rectifier for signed activations: y = max(0, x),
//           with a bypass input for layers that keep signed outputs (the
//           classifier's logits).
// -----------------------------------------------------------------------------
`default_nettype none

module relu #(
  parameter int unsigned DATA_W = 8
) (
  input  wire  signed [DATA_W-1:0] x_i,
  input  wire                      bypass_i,
  output logic signed [DATA_W-1:0] y_o
);

  assign y_o = (!bypass_i && x_i[DATA_W-1]) ? '0 : x_i;

endmodule : relu

`default_nettype wire
