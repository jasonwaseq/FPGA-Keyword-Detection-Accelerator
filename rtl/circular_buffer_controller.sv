// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : circular_buffer_controller
// Purpose : Write-pointer and occupancy management for the circular feature
//           buffer, with commit semantics.
//
// The packet decoder writes payload bytes speculatively into the slot at
// wr_frame_o while the packet is still arriving. Only commit_i - asserted
// after the packet's CRC verifies - advances the pointer and publishes the
// frame; a corrupted packet is silently overwritten by the next one. This is
// what makes double buffering unnecessary: readers only ever see committed
// frames, and the writer only ever touches the one uncommitted slot ahead of
// them (see docs/architecture.md, "Feature buffer").
//
// frames_total_o saturates once the buffer holds enough history for a full
// window; the metadata of the newest committed frame (host frame number and
// host timestamp) is captured for event attribution.
// -----------------------------------------------------------------------------
`default_nettype none

module circular_buffer_controller #(
  parameter int unsigned HIST_DEPTH = 64,   // frames, power of two
  parameter int unsigned WINDOW_LEN = 32
) (
  input  wire                          clk_i,
  input  wire                          rst_ni,
  input  wire                          clear_i,        // soft reset: empty buffer
  // Commit interface (packet decoder, CRC-verified)
  input  wire                          commit_i,
  input  wire [31:0]                   frame_num_i,
  input  wire [31:0]                   timestamp_i,
  // Buffer state
  output logic [$clog2(HIST_DEPTH)-1:0] wr_frame_o,     // slot being filled
  output logic [$clog2(WINDOW_LEN):0]  frames_total_o, // saturates at WINDOW_LEN
  output logic [31:0]                  newest_frame_num_o,
  output logic [31:0]                  newest_timestamp_o
);

  localparam int unsigned TW = $clog2(WINDOW_LEN) + 1;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      wr_frame_o         <= '0;
      frames_total_o     <= '0;
      newest_frame_num_o <= '0;
      newest_timestamp_o <= '0;
    end else if (clear_i) begin
      wr_frame_o     <= '0;
      frames_total_o <= '0;
    end else if (commit_i) begin
      wr_frame_o         <= wr_frame_o + 1'b1;  // natural wrap (power of two)
      newest_frame_num_o <= frame_num_i;
      newest_timestamp_o <= timestamp_i;
      if (frames_total_o != TW'(WINDOW_LEN)) begin
        frames_total_o <= frames_total_o + 1'b1;
      end
    end
  end

`ifndef SYNTHESIS
  initial begin
    assert (HIST_DEPTH >= 2 * WINDOW_LEN)
      else $error("circular_buffer_controller: HIST_DEPTH must be >= 2x WINDOW_LEN");
  end
`endif

endmodule : circular_buffer_controller

`default_nettype wire
