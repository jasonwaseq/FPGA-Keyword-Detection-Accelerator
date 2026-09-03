// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : kernel_memory
// Purpose : Banked INT8 convolution-kernel ROM: PARALLEL banks, each holding
//           the complete flattened kernel array (index = (oc*K + k)*IC + ic)
//           and serving one MAC lane with an independent address.
//
// Banking tradeoff: each bank stores the FULL array rather than a 1/P slice.
// Slicing would require the build system to emit P layout-dependent .mem
// files per parallelism setting; full replication keeps one canonical file
// for every P and costs (P-1) * ceil(array/512) extra EBRs - at the default
// P=2 that is 2 of 30 EBRs. For P > 4 slice the initialization files instead
// (see docs/architecture.md, "Scaling the MAC array").
// -----------------------------------------------------------------------------
`default_nettype none

module kernel_memory #(
  parameter int unsigned DATA_W   = 8,
  parameter int unsigned OUT_CH   = 8,
  parameter int unsigned CONV_K   = 3,
  parameter int unsigned IN_CH    = 40,
  parameter int unsigned PARALLEL = 2,
  parameter              MEM_FILE = "weights/conv_weights.mem"
) (
  input  wire                        clk_i,
  input  wire  [PARALLEL-1:0][$clog2(OUT_CH*CONV_K*IN_CH)-1:0] addr_i,
  output logic [PARALLEL-1:0][DATA_W-1:0]                      data_o
);

  localparam int unsigned DEPTH = OUT_CH * CONV_K * IN_CH;

  for (genvar b = 0; b < PARALLEL; b++) begin : g_bank
    rom_sync #(
      .DATA_W   (DATA_W),
      .DEPTH    (DEPTH),
      .MEM_FILE (MEM_FILE)
    ) u_rom (
      .clk_i,
      .addr_i (addr_i[b]),
      .data_o (data_o[b])
    );
  end

`ifndef SYNTHESIS
  initial begin
    assert (OUT_CH % PARALLEL == 0)
      else $error("kernel_memory: PARALLEL must divide OUT_CH");
  end
`endif

endmodule : kernel_memory

`default_nettype wire
