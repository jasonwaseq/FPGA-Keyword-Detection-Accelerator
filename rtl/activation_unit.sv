// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : activation_unit
// Purpose : Combinational requantization from an INT32-class accumulator to
//           INT8, with optional fused ReLU:
//
//             y = sat8( (acc * M + 2^(S-1)) >>> S )      then ReLU if enabled
//
// M is a positive multiplier < 2^15, S in [1, 31] (round-half-up toward
// +inf). This is the TFLite-Micro style fixed-point rescale; the identical
// arithmetic lives in model/kws_quant.py (calibration) and host ref_model.c
// (bit-accurate verification oracle).
//
// Implementation notes:
//  * acc is bounded to ACC_W=24 bits by construction (asserted); the 24x16
//    signed product fits in 40 bits and maps onto SB_MAC16 DSPs under
//    'synth_ice40 -dsp'.
//  * Fully combinational: at 12 MHz (83 ns budget) the multiply+shift+sat
//    chain closes with wide margin (see docs/timing.md). For higher clocks,
//    register the product - the callers already treat requantization as a
//    separate FSM state, so an added cycle is a localized change.
// -----------------------------------------------------------------------------
`default_nettype none

module activation_unit #(
  parameter int unsigned ACC_W  = 24,
  parameter int unsigned MULT_W = 16,
  parameter int unsigned DATA_W = 8
) (
  input  wire  signed [ACC_W-1:0]  acc_i,
  input  wire         [MULT_W-1:0] mult_i,     // M, positive, < 2^(MULT_W-1)
  input  wire         [4:0]        shift_i,    // S, 1..31
  input  wire                      relu_en_i,
  output logic signed [DATA_W-1:0] y_o
);

  localparam int unsigned PROD_W = ACC_W + MULT_W + 1;

  logic signed [PROD_W-1:0] prod, rounded, shifted;
  logic signed [DATA_W-1:0] sat;

  // Saturation detection by sign-extension check rather than magnitude
  // comparison: shifted fits INT8 iff its bits above [DATA_W-2] are a pure
  // sign extension. Equality reductions map to LUT trees; the > / <
  // comparators they replace each cost a PROD_W-long carry chain.
  logic [PROD_W-1:DATA_W-1] hi;
  always_comb begin
    prod    = acc_i * signed'({1'b0, mult_i});
    rounded = prod + (PROD_W'(1) <<< (shift_i - 5'd1));
    shifted = rounded >>> shift_i;

    hi = shifted[PROD_W-1:DATA_W-1];
    if ((hi == '0) || (hi == '1)) begin
      sat = shifted[DATA_W-1:0];                        // in range
    end else if (shifted[PROD_W-1]) begin
      sat = -(DATA_W'(2 ** (DATA_W - 1)));              // -128
    end else begin
      sat = DATA_W'(2 ** (DATA_W - 1) - 1);             //  127
    end
  end

  relu #(.DATA_W (DATA_W)) u_relu (
    .x_i      (sat),
    .bypass_i (!relu_en_i),
    .y_o
  );

endmodule : activation_unit

`default_nettype wire
