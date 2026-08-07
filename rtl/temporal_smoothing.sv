// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : temporal_smoothing
// Purpose : Decision layer between raw per-window classifications and keyword
//           events. A single inference never triggers; a detection requires
//           ALL of the following, evaluated after each classifier result is
//           folded into the moving-average history:
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
// Pipeline: update_i starts a multi-cycle confidence fold; then a sequential
// argmax scan (one class/cycle) selects the smoothed winner; then eval_q
// fires. Sequential scan keeps the UP5K timing/LC budget with N=10.
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

  input  wire                       update_i,
  input  wire [N-1:0][DATA_W-1:0]   logits_i,
  input  wire [$clog2(N)-1:0]       winner_i,

  input  wire                       en_i,
  input  wire signed [DATA_W-1:0]   thresh_i,
  input  wire        [3:0]          vote_min_i,
  input  wire        [3:0]          min_consec_i,
  input  wire        [7:0]          debounce_i,
  input  wire        [N-1:0]        target_mask_i,

  output logic                      detect_o,
  output logic [$clog2(N)-1:0]      det_class_o,
  output logic [7:0]                det_conf_o,
  output logic [3:0]                det_votes_o,
  output logic                      busy_o
);

  localparam int unsigned CW = $clog2(N);

  logic [N-1:0][DATA_W-1:0] avg;
  logic                     acc_busy;
  confidence_accumulator #(
    .N      (N),
    .DATA_W (DATA_W),
    .DEPTH  (DEPTH)
  ) u_avg (
    .clk_i, .rst_ni, .clear_i,
    .update_i,
    .logits_i,
    .avg_o  (avg),
    .busy_o (acc_busy)
  );

  logic [CW-1:0]            win_hist_q [DEPTH];
  logic [$clog2(DEPTH):0]   hist_fill_q;
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

  // Sequential argmax over avg[] after the fold completes (ties -> lowest idx).
  typedef enum logic [1:0] {
    ST_IDLE,
    ST_SCAN,
    ST_EVAL
  } phase_e;

  phase_e                   phase_q;
  logic                     pend_q;
  logic                     acc_busy_d;
  logic [CW-1:0]            scan_c_q;
  logic [CW-1:0]            sm_idx_q;
  logic signed [DATA_W-1:0] sm_val_q;
  logic [3:0]               consec_q;
  logic [7:0]               debounce_q;
  logic [CW-1:0]            last_cand_q;

  logic [3:0] votes;
  always_comb begin
    votes = '0;
    for (int d = 0; d < DEPTH; d++) begin
      if ((32'(d) < 32'(hist_fill_q)) && (win_hist_q[d] == sm_idx_q)) begin
        votes = votes + 4'd1;
      end
    end
  end

  wire candidate = en_i
                 && target_mask_i[sm_idx_q]
                 && (sm_val_q >= thresh_i)
                 && (votes >= vote_min_i);

  logic [3:0] run;
  always_comb begin
    if (!candidate) begin
      run = 4'd0;
    end else if ((consec_q != 4'd0) && (last_cand_q == sm_idx_q)) begin
      run = (consec_q == 4'hF) ? consec_q : consec_q + 4'd1;
    end else begin
      run = 4'd1;
    end
  end

  wire fire = candidate && (debounce_q == '0) && (run >= min_consec_i);

  assign busy_o = acc_busy | pend_q | (phase_q != ST_IDLE);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      phase_q     <= ST_IDLE;
      pend_q      <= 1'b0;
      acc_busy_d  <= 1'b0;
      scan_c_q    <= '0;
      sm_idx_q    <= '0;
      sm_val_q    <= '0;
      consec_q    <= '0;
      debounce_q  <= '0;
      last_cand_q <= '0;
      detect_o    <= 1'b0;
      det_class_o <= '0;
      det_conf_o  <= '0;
      det_votes_o <= '0;
    end else begin
      detect_o   <= 1'b0;
      acc_busy_d <= acc_busy;

      if (clear_i) begin
        phase_q    <= ST_IDLE;
        pend_q     <= 1'b0;
        consec_q   <= '0;
        debounce_q <= '0;
      end else begin
        if (update_i) pend_q <= 1'b1;

        unique case (phase_q)
          ST_IDLE: begin
            if (pend_q && acc_busy_d && !acc_busy) begin
              // Seed scan with class 0; compare 1..N-1 next.
              sm_idx_q <= '0;
              sm_val_q <= signed'(avg[0]);
              scan_c_q <= (N > 1) ? CW'(1) : '0;
              pend_q   <= 1'b0;
              phase_q  <= (N > 1) ? ST_SCAN : ST_EVAL;
            end
          end

          ST_SCAN: begin
            if (signed'(avg[scan_c_q]) > sm_val_q) begin
              sm_val_q <= signed'(avg[scan_c_q]);
              sm_idx_q <= scan_c_q;
            end
            if (scan_c_q == CW'(N - 1)) begin
              phase_q <= ST_EVAL;
            end else begin
              scan_c_q <= scan_c_q + 1'b1;
            end
          end

          ST_EVAL: begin
            if (candidate) last_cand_q <= sm_idx_q;
            if (fire) begin
              detect_o    <= 1'b1;
              det_class_o <= sm_idx_q;
              det_conf_o  <= sm_val_q[DATA_W-1] ? 8'd0 : 8'(sm_val_q);
              det_votes_o <= votes;
              debounce_q  <= debounce_i;
              consec_q    <= '0;
            end else begin
              consec_q <= run;
              if (debounce_q != '0) debounce_q <= debounce_q - 1'b1;
            end
            phase_q <= ST_IDLE;
          end

          default: phase_q <= ST_IDLE;
        endcase
      end
    end
  end

endmodule : temporal_smoothing

`default_nettype wire
