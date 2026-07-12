// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : reset_sync
// Purpose : Asynchronous-assert / synchronous-deassert reset synchronizer with
//           a power-on-reset stretch counter.
//
// arst_ni asserts the output reset immediately (asynchronously); release is
// synchronized through STAGES flops and additionally held until the POR
// counter saturates, guaranteeing a minimum reset pulse after configuration
// (iCE40 flip-flops wake at 0, so the counter starts counting from device
// configuration).
// -----------------------------------------------------------------------------
`default_nettype none

module reset_sync #(
  parameter int unsigned STAGES  = 4,
  parameter int unsigned POR_LEN = 255   // cycles reset is held after power-up
) (
  input  wire  clk_i,
  input  wire  arst_ni,   // asynchronous reset request, active low
  output logic rst_no     // synchronized reset, active low
);

  localparam int unsigned CW = $clog2(POR_LEN + 1);

  logic [STAGES-1:0] sync_q;
  logic [CW-1:0]     por_cnt_q;
  logic              por_done_q;

  always_ff @(posedge clk_i or negedge arst_ni) begin
    if (!arst_ni) sync_q <= '0;
    else          sync_q <= {sync_q[STAGES-2:0], 1'b1};
  end

  always_ff @(posedge clk_i) begin
    if (!por_done_q) begin
      por_cnt_q  <= por_cnt_q + 1'b1;
      por_done_q <= (por_cnt_q == CW'(POR_LEN - 1));
    end
  end

  assign rst_no = sync_q[STAGES-1] & por_done_q;

endmodule : reset_sync

`default_nettype wire
