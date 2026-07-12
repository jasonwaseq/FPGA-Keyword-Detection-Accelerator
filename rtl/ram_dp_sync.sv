// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : ram_dp_sync
// Purpose : Generic simple-dual-port RAM: one synchronous write port, one
//           synchronous read port (1-cycle latency). Maps directly onto iCE40
//           EBR, whose native topology is exactly 1W + 1R.
//
// Read-during-write to the same address returns OLD data; no instance in this
// design ever reads an address in the same cycle it is written (the feature
// buffer's commit-pointer scheme and the strictly sequential layer engines
// guarantee it - asserted below).
// -----------------------------------------------------------------------------
`default_nettype none

module ram_dp_sync #(
  parameter int unsigned DATA_W = 8,
  parameter int unsigned DEPTH  = 256
) (
  input  wire                      clk_i,
  // Write port
  input  wire                      wr_en_i,
  input  wire [$clog2(DEPTH)-1:0]  wr_addr_i,
  input  wire [DATA_W-1:0]         wr_data_i,
  // Read port
  input  wire [$clog2(DEPTH)-1:0]  rd_addr_i,
  output logic [DATA_W-1:0]        rd_data_o
);

  logic [DATA_W-1:0] mem [DEPTH];

  always_ff @(posedge clk_i) begin
    if (wr_en_i) mem[wr_addr_i] <= wr_data_i;
    rd_data_o <= mem[rd_addr_i];
  end

`ifndef SYNTHESIS
  always_ff @(posedge clk_i) begin
    assert (!(wr_en_i && (wr_addr_i == rd_addr_i)))
      else $warning("ram_dp_sync: read-during-write at addr %0h", wr_addr_i);
  end
`endif

endmodule : ram_dp_sync

`default_nettype wire
