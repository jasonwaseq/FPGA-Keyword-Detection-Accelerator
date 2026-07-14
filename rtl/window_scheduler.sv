// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : window_scheduler
// Purpose : Autonomous inference scheduling. Watches the feature buffer's
//           commit stream and issues an overlapping analysis window of
//           WINDOW_LEN frames every WINDOW_STRIDE commits - no host
//           involvement, ever.
//
// Scheduling law: the first window is issued the moment the buffer becomes
// primed (WINDOW_LEN frames committed, i.e. at commit #WINDOW_LEN); every
// subsequent window is due WINDOW_STRIDE commits after the previous issue,
// so windows land on commits WINDOW_LEN, WINDOW_LEN+STRIDE, ... Stride
// counting starts only after the first issue - crucially, credit does NOT
// accumulate during the initial buffer fill, which would otherwise issue
// several back-to-back windows over identical data.
//
// Stride credit is carried (stride_cnt -= WINDOW_STRIDE at issue) so window
// alignment stays on the stride grid even if an issue is briefly delayed by
// a busy engine. If a full extra stride accumulates while the engine is
// still busy, the oldest pending window is abandoned (drop_o) rather than
// letting scheduling debt grow without bound - by design margin this cannot
// occur (inference ~1.3 ms vs 80 ms stride period; see docs/pipeline.md) and
// the counter exists as a health indicator.
//
// The issued window spans absolute frames [wr_frame - WINDOW_LEN, wr_frame-1]
// (the WINDOW_LEN most recently committed frames). The schedule is mirrored
// by the host reference model: window k covers frames (n-WINDOW_LEN..n-1)
// for n = WINDOW_LEN + k*WINDOW_STRIDE.
// -----------------------------------------------------------------------------
`default_nettype none

module window_scheduler #(
  parameter int unsigned HIST_DEPTH    = 64,
  parameter int unsigned WINDOW_LEN    = 32,
  parameter int unsigned WINDOW_STRIDE = 8
) (
  input  wire                           clk_i,
  input  wire                           rst_ni,
  input  wire                           clear_i,       // soft reset
  input  wire                           enable_i,      // stream enabled
  // Feature buffer state
  input  wire                           commit_i,      // frame committed
  input  wire [$clog2(HIST_DEPTH)-1:0]  wr_frame_i,
  input  wire [$clog2(WINDOW_LEN):0]    frames_total_i,
  input  wire [31:0]                    newest_frame_num_i,
  input  wire [31:0]                    newest_timestamp_i,
  // Engine handshake
  input  wire                           engine_busy_i,
  output logic                          issue_o,          // 1-cycle strobe
  output logic [$clog2(HIST_DEPTH)-1:0] win_base_o,       // first frame of window
  output logic [31:0]                   win_frame_num_o,  // newest frame in window
  output logic [31:0]                   win_timestamp_o,  // host ts of that frame
  // Statistics
  output logic                          drop_o            // window abandoned
);

  localparam int unsigned FR_W = $clog2(HIST_DEPTH);
  localparam int unsigned SW   = 8;  // stride counter width (saturating)

  logic [SW-1:0] stride_cnt_q;
  logic          first_done_q;   // first window issued since clear

  wire primed     = (frames_total_i == ($clog2(WINDOW_LEN)+1)'(WINDOW_LEN));
  wire first_due  = primed && !first_done_q;
  wire stride_due = first_done_q && (stride_cnt_q >= SW'(WINDOW_STRIDE));
  wire issue      = (first_due || stride_due) && !engine_busy_i && enable_i;
  wire drop       = stride_due && engine_busy_i
                    && (stride_cnt_q >= SW'(2 * WINDOW_STRIDE));

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      stride_cnt_q    <= '0;
      first_done_q    <= 1'b0;
      issue_o         <= 1'b0;
      drop_o          <= 1'b0;
      win_base_o      <= '0;
      win_frame_num_o <= '0;
      win_timestamp_o <= '0;
    end else begin
      issue_o <= 1'b0;
      drop_o  <= 1'b0;

      if (clear_i) begin
        stride_cnt_q <= '0;
        first_done_q <= 1'b0;
      end else begin
        // Net stride bookkeeping: +1 per commit once the first window is out,
        // -STRIDE per issue or drop. A commit can coincide with an issue.
        stride_cnt_q <= stride_cnt_q
                        + ((commit_i && first_done_q && (stride_cnt_q != '1))
                             ? SW'(1) : SW'(0))
                        - (((issue && stride_due) || drop)
                             ? SW'(WINDOW_STRIDE) : SW'(0));

        if (issue) begin
          first_done_q    <= 1'b1;
          issue_o         <= 1'b1;
          win_base_o      <= wr_frame_i - FR_W'(WINDOW_LEN);  // modular
          win_frame_num_o <= newest_frame_num_i;
          win_timestamp_o <= newest_timestamp_i;
        end else if (drop) begin
          drop_o <= 1'b1;
        end
      end
    end
  end

`ifndef SYNTHESIS
  initial begin
    assert (WINDOW_STRIDE <= WINDOW_LEN)
      else $error("window_scheduler: stride larger than window is unsupported");
    assert (2 * WINDOW_STRIDE < (1 << SW) - 1)
      else $error("window_scheduler: stride counter too narrow");
  end
`endif

endmodule : window_scheduler

`default_nettype wire
