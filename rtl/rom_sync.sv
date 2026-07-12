// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : rom_sync
// Purpose : Generic synchronous-read ROM initialized from a $readmemh file.
//           One-cycle read latency; maps onto iCE40 EBR for byte-wide arrays
//           (wider/shallower instances fall back to LUT logic, which is the
//           right answer for the tiny bias/parameter tables).
//
// MEM_FILE paths are resolved relative to the tool's working directory; the
// build system always invokes yosys/verilator from the repository root.
// -----------------------------------------------------------------------------
`default_nettype none

module rom_sync #(
  parameter int unsigned DATA_W   = 8,
  parameter int unsigned DEPTH    = 256,
  parameter              MEM_FILE = ""
) (
  input  wire                       clk_i,
  input  wire  [$clog2(DEPTH)-1:0]  addr_i,
  output logic [DATA_W-1:0]         data_o
);

  logic [DATA_W-1:0] mem [DEPTH];

  initial begin
    if (MEM_FILE != "") $readmemh(MEM_FILE, mem);
  end

  always_ff @(posedge clk_i) begin
    data_o <= mem[addr_i];
  end

endmodule : rom_sync

`default_nettype wire
