// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : temporal_smoothing
// Purpose : Decision layer between raw per-window classifications and keyword
//           events. A single inference never triggers; a detection requires
//           ALL of the following, evaluated one cycle after each classifier
//           result is folded in:
//
//   1. moving average : argmax of the DEPTH-deep per-class logit averages
//                       (confidence_accumulator) selects the winner
//   2. target mask    : winner must be an armed class (silence/unknown out)
//   3. threshold      : smoothed winner score >= thresh_i
//   4. majority vote  : winner matches >= vote_min_i of the last DEPTH
//                       per-window winners
//   5. consecutive    : conditions 1-4 held for >= min_consec_i inferences
//                       with the same winning class
//   6. debounce       : at least debounce_i inferences since the last event
//
// All knobs are runtime inputs driven by the register file (defaults from
// kws_pkg). The identical decision procedure is implemented in
// host/src/ref_model.c and locked down by tb_smoothing / tb_kws_core.
//
// Pipeline: update_i (cycle 0: fold logits, record winner) -> eval_q
// (cycle 1: decide, pulse detect_o). Back-to-back updates cannot occur -
// inferences are separated by thousands of cycles.
// -----------------------------------------------------------------------------
`default_nettype none

module temporal_smoothing #(
  parameter int unsigned N      = kws_pkg::NUM_CLASSES,
  parameter int unsigned DATA_W = kws_pkg::DATA_W,
  parameter int unsigned DEPTH  = kws_pkg::SMOOTH_DEPTH
) (
  input  wire                       clk_i,
  input  wire                       rst_ni,
  input  wire                       clear_i,

  // Per-inference result (classifier done strobe)
  input  wire                       update_i,
  input  wire [N-1:0][DATA_W-1:0]   logits_i,
  input  wire [$clog2(N)-1:0]       winner_i,     // per-window argmax

  // Runtime configuration (register file)
  input  wire                       en_i,
  input  wire signed [DATA_W-1:0]   thresh_i,
  input  wire        [3:0]          vote_min_i,
  input  wire        [3:0]          min_consec_i,
  input  wire        [7:0]          debounce_i,
  input  wire        [N-1:0]        target_mask_i,

  // Detection event
  output logic                      detect_o,      // 1-cycle strobe
  output logic [$clog2(N)-1:0]      det_class_o,
  output logic [7:0]                det_conf_o,    // smoothed score, clamped >= 0
  output logic [3:0]                det_votes_o,
  output logic                      busy_o         // evaluation in flight
);

  localparam int unsigned CW = $clog2(N);

  // --- moving averages -------------------------------------------------------
  logic [N-1:0][DATA_W-1:0] avg;
  confidence_accumulator #(
    .N      (N),
    .DATA_W (DATA_W),
    .DEPTH  (DEPTH)
  ) u_avg (
    .clk_i, .rst_ni, .clear_i,
    .update_i,
    .logits_i,
    .avg_o (avg)
  );

  // --- per-window winner history for majority voting -------------------------
  logic [CW-1:0]            win_hist_q [DEPTH];
  logic [$clog2(DEPTH):0]   hist_fill_q;           // entries valid since clear
  logic [$clog2(DEPTH)-1:0] whead_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      whead_q     <= '0;
      hist_fill_q <= '0;
      for (int d = 0; d < DEPTH; d++) win_hist_q[d] <= '0;
    end else if (clear_i) begin
      whead_q     <= '0;
      hist_fill_q <= '0;
      for (int d = 0; d < DEPTH; d++) win_hist_q[d] <= '0;
    end else if (update_i) begin
      win_hist_q[whead_q] <= winner_i;
      whead_q             <= whead_q + 1'b1;
      if (hist_fill_q != ($clog2(DEPTH)+1)'(DEPTH)) begin
        hist_fill_q <= hist_fill_q + 1'b1;
      end
    end
  end

  // --- smoothed winner (combinational over current averages) -----------------
  logic [CW-1:0]            sm_idx;
  logic signed [DATA_W-1:0] sm_val;
  argmax #(
    .N      (N),
    .DATA_W (DATA_W)
  ) u_argmax (
    .values_i (avg),
    .idx_o    (sm_idx),
    .max_o    (sm_val)
  );

  // Votes for the smoothed winner among recorded per-window winners.
  logic [3:0] votes;
  always_comb begin
    votes = '0;
    for (int d = 0; d < DEPTH; d++) begin
      if ((32'(d) < 32'(hist_fill_q)) && (win_hist_q[d] == sm_idx)) begin
        votes = votes + 4'd1;
      end
    end
  end

  // --- decision pipeline ------------------------------------------------------
  logic       eval_q;        // evaluate one cycle after the update landed
  logic [3:0] consec_q;
  logic [7:0] debounce_q;
  logic [CW-1:0] last_cand_q;

  wire candidate = en_i
                 && target_mask_i[sm_idx]
                 && (sm_val >= thresh_i)
                 && (votes >= vote_min_i);

  // Run length including the current evaluation: 0 if not a candidate, resets
  // to 1 on a class change, saturates at 15. This exact procedure is mirrored
  // in ref_model.c (kws_smooth_step).
  logic [3:0] run;
  always_comb begin
    if (!candidate) begin
      run = 4'd0;
    end else if ((consec_q != 4'd0) && (last_cand_q == sm_idx)) begin
      run = (consec_q == 4'hF) ? consec_q : consec_q + 4'd1;
    end else begin
      run = 4'd1;
    end
  end

  wire fire = candidate && (debounce_q == '0) && (run >= min_consec_i);

  assign busy_o = eval_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      eval_q      <= 1'b0;
      consec_q    <= '0;
      debounce_q  <= '0;
      last_cand_q <= '0;
      detect_o    <= 1'b0;
      det_class_o <= '0;
      det_conf_o  <= '0;
      det_votes_o <= '0;
    end else begin
      detect_o <= 1'b0;
      eval_q   <= update_i;

      if (clear_i) begin
        eval_q     <= 1'b0;
        consec_q   <= '0;
        debounce_q <= '0;
      end else if (eval_q) begin
        // One evaluation per inference: the accumulator and winner history
        // already include the newest sample (folded last cycle).
        if (candidate) last_cand_q <= sm_idx;

        if (fire) begin
          detect_o    <= 1'b1;
          det_class_o <= sm_idx;
          det_conf_o  <= sm_val[DATA_W-1] ? 8'd0 : 8'(sm_val);
          det_votes_o <= votes;
          debounce_q  <= debounce_i;
          consec_q    <= '0;   // a fresh run is required for the next event
        end else begin
          consec_q <= run;
          if (debounce_q != '0) debounce_q <= debounce_q - 1'b1;
        end
      end
    end
  end

endmodule : temporal_smoothing

`default_nettype wire
