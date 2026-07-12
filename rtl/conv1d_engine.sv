// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : conv1d_engine
// Purpose : Streaming INT8 temporal (1-D) convolution over the circular
//           feature buffer with PARALLEL output-channel MAC lanes.
//
//   out[t][oc] = requant( bias[oc] + sum_{k,ic} feat[base+t+k][ic] * w[oc][k][ic] )
//   followed by fused ReLU. Valid convolution: t in [0, WINDOW_LEN-K].
//
// Dataflow / reuse strategy (docs/architecture.md discusses alternatives):
//   * Loop nest: oc_group (outer) -> t -> (k, ic) (inner). Biases load once
//     per oc_group; kernel banks stream sequentially; every feature fetch is
//     shared by all PARALLEL lanes (feature reuse = PARALLEL, kernel reuse =
//     1 fetch/MAC out of dedicated banks). Weight-stationary alternatives
//     were rejected: with only OUT_CH*K*IC = 960 weights the ROMs are cheap,
//     while feature RAM has a single read port - sharing it is the win.
//   * All memory reads have a 2-cycle latency (registered address, registered
//     ROM/RAM output); the MAC issue pipeline runs one fetch per cycle with a
//     2-stage valid shifter, so each output row costs K*IC + 2 cycles.
//
// Cycle count per window (see docs/pipeline.md):
//   OUT_CH/P * ( 3P + 3 + OUT_LEN * (K*IC + 2 + 1 + P) )   + 5 (param load)
//   Default config (P=2): ~15.2k cycles = 1.27 ms @ 12 MHz.
//
// FSM: IDLE -> LD_M_A -> LD_M_W -> LD_M_C -> LD_S_W -> LD_S_C
//        -> [LD_B_A -> LD_B_W -> LD_B_C]xP -> {MAC -> REQ}xOUT_LEN -> ... -> IDLE
// -----------------------------------------------------------------------------
`default_nettype none

module conv1d_engine #(
  parameter int unsigned DATA_W     = kws_pkg::DATA_W,
  parameter int unsigned ACC_W      = kws_pkg::ACC_W,
  parameter int unsigned MULT_W     = kws_pkg::MULT_W,
  parameter int unsigned IN_CH      = kws_pkg::NUM_MFCC,
  parameter int unsigned WINDOW_LEN = kws_pkg::WINDOW_LEN,
  parameter int unsigned CONV_K     = kws_pkg::CONV_K,
  parameter int unsigned OUT_CH     = kws_pkg::CONV_OUT_CH,
  parameter int unsigned PARALLEL   = kws_pkg::PARALLEL_OUT_CH,
  parameter int unsigned HIST_DEPTH = kws_pkg::HIST_DEPTH
) (
  input  wire         clk_i,
  input  wire         rst_ni,

  // Control
  input  wire                            start_i,
  input  wire [$clog2(HIST_DEPTH)-1:0]   win_base_i,
  output logic                           busy_o,
  output logic                           done_o,      // 1-cycle strobe

  // Feature buffer read port (2-cycle latency)
  output logic [$clog2(HIST_DEPTH)-1:0]  feat_frame_o,
  output logic [$clog2(IN_CH)-1:0]       feat_coef_o,
  input  wire  [DATA_W-1:0]              feat_data_i,

  // Kernel memory, PARALLEL banks (2-cycle latency)
  output logic [PARALLEL-1:0][$clog2(OUT_CH*CONV_K*IN_CH)-1:0] krn_addr_o,
  input  wire  [PARALLEL-1:0][DATA_W-1:0]                      krn_data_i,

  // Bias/parameter ROM (2-cycle latency)
  output logic [$clog2(OUT_CH+2)-1:0]    bias_addr_o,
  input  wire  [31:0]                    bias_data_i,

  // Activation RAM write port
  output logic                                    act_wr_en_o,
  output logic [$clog2((WINDOW_LEN-CONV_K+1)*OUT_CH)-1:0] act_wr_addr_o,
  output logic [DATA_W-1:0]                       act_wr_data_o
);

  localparam int unsigned OUT_LEN = WINDOW_LEN - CONV_K + 1;
  localparam int unsigned FR_W    = $clog2(HIST_DEPTH);
  localparam int unsigned IC_W    = $clog2(IN_CH);
  localparam int unsigned KA_W    = $clog2(OUT_CH * CONV_K * IN_CH);
  localparam int unsigned OA_W    = $clog2(OUT_LEN * OUT_CH);
  localparam int unsigned TAPS    = CONV_K * IN_CH;      // MACs per output
  localparam int unsigned J_W     = $clog2(TAPS + 1);
  localparam int unsigned T_W     = $clog2(OUT_LEN);
  localparam int unsigned G_W     = $clog2(OUT_CH);      // oc column counter
  localparam int unsigned P_W     = (PARALLEL > 1) ? $clog2(PARALLEL) : 1;

  typedef enum logic [3:0] {
    ST_IDLE,
    ST_LD_M_A, ST_LD_M_W, ST_LD_M_C,   // multiplier M (bias ROM addr OUT_CH)
    ST_LD_S_W, ST_LD_S_C,              // shift S      (bias ROM addr OUT_CH+1)
    ST_LD_B_A, ST_LD_B_W, ST_LD_B_C,   // per-lane bias, PARALLEL iterations
    ST_MAC,                            // stream TAPS fetches + 2-cycle drain
    ST_REQ                             // requantize + write PARALLEL results
  } state_e;

  state_e                    state_q;
  logic [FR_W-1:0]           win_base_q;
  logic [MULT_W-1:0]         m_q;
  logic [4:0]                s_q;
  logic signed [ACC_W-1:0]   bias_q [PARALLEL];
  logic signed [ACC_W-1:0]   acc_q  [PARALLEL];

  // Loop counters
  logic [G_W-1:0]  oc_col_q;              // oc_group * PARALLEL
  logic [T_W-1:0]  t_q;                   // output time index
  logic [J_W-1:0]  j_q;                   // tap issue index 0..TAPS
  logic [IC_W-1:0] ic_q;                  // input channel sub-counter
  logic [$clog2(CONV_K)-1:0] k_q;         // kernel tap sub-counter
  logic [P_W-1:0]  lane_q;                // bias-load / requant lane
  logic [KA_W-1:0] lane_base_q [PARALLEL];// per-lane kernel base address
  logic [OA_W-1:0] out_row_q;             // t * OUT_CH (incremental)

  // MAC issue/consume pipeline (2-cycle memory latency)
  logic v1_q, v2_q;
  wire  issue_v = (state_q == ST_MAC) && (j_q != J_W'(TAPS));
  wire  mac_last_grp = (32'(oc_col_q) == OUT_CH - PARALLEL);
  wire  mac_last_t   = (32'(t_q) == OUT_LEN - 1);

  // Shared requantization unit, serialized over lanes in ST_REQ.
  logic signed [DATA_W-1:0] req_y;
  activation_unit #(
    .ACC_W  (ACC_W),
    .MULT_W (MULT_W),
    .DATA_W (DATA_W)
  ) u_requant (
    .acc_i     (acc_q[lane_q]),
    .mult_i    (m_q),
    .shift_i   (s_q),
    .relu_en_i (1'b1),
    .y_o       (req_y)
  );

  assign busy_o = (state_q != ST_IDLE);

  // The parameter ROM is uniformly 32-bit; only ACC_W-bit biases and
  // MULT_W/5-bit requant fields are consumed here.
  if (ACC_W < 32) begin : g_bias_unused
    wire _unused_bias_hi = &{1'b0, bias_data_i[31:ACC_W]};
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q       <= ST_IDLE;
      win_base_q    <= '0;
      m_q           <= '0;
      s_q           <= '0;
      oc_col_q      <= '0;
      t_q           <= '0;
      j_q           <= '0;
      ic_q          <= '0;
      k_q           <= '0;
      lane_q        <= '0;
      out_row_q     <= '0;
      v1_q          <= 1'b0;
      v2_q          <= 1'b0;
      done_o        <= 1'b0;
      feat_frame_o  <= '0;
      feat_coef_o   <= '0;
      bias_addr_o   <= '0;
      act_wr_en_o   <= 1'b0;
      act_wr_addr_o <= '0;
      act_wr_data_o <= '0;
      for (int l = 0; l < PARALLEL; l++) begin
        bias_q[l]      <= '0;
        acc_q[l]       <= '0;
        lane_base_q[l] <= '0;
        krn_addr_o[l]  <= '0;
      end
    end else begin
      done_o      <= 1'b0;
      act_wr_en_o <= 1'b0;

      // MAC datapath: consume fetched operands two cycles after issue.
      v1_q <= issue_v;
      v2_q <= v1_q;
      if (v2_q) begin
        for (int l = 0; l < PARALLEL; l++) begin
          acc_q[l] <= acc_q[l]
                    + ACC_W'(signed'(feat_data_i) * signed'(krn_data_i[l]));
        end
      end

      unique case (state_q)
        ST_IDLE: begin
          if (start_i) begin
            win_base_q <= win_base_i;
            oc_col_q   <= '0;
            for (int l = 0; l < PARALLEL; l++) begin
              lane_base_q[l] <= KA_W'(l * CONV_K * IN_CH);
            end
            state_q <= ST_LD_M_A;
          end
        end

        // --- layer parameter load: M then S -------------------------------
        ST_LD_M_A: begin
          bias_addr_o <= ($clog2(OUT_CH+2))'(OUT_CH);
          state_q     <= ST_LD_M_W;
        end
        ST_LD_M_W: state_q <= ST_LD_M_C;
        ST_LD_M_C: begin
          m_q         <= bias_data_i[MULT_W-1:0];
          bias_addr_o <= ($clog2(OUT_CH+2))'(OUT_CH + 1);
          state_q     <= ST_LD_S_W;
        end
        ST_LD_S_W: state_q <= ST_LD_S_C;
        ST_LD_S_C: begin
          s_q     <= bias_data_i[4:0];
          lane_q  <= '0;
          state_q <= ST_LD_B_A;
        end

        // --- per-oc_group bias load (one lane per A/W/C triplet) ----------
        ST_LD_B_A: begin
          bias_addr_o <= ($clog2(OUT_CH+2))'(32'(oc_col_q) + 32'(lane_q));
          state_q     <= ST_LD_B_W;
        end
        ST_LD_B_W: state_q <= ST_LD_B_C;
        ST_LD_B_C: begin
          bias_q[lane_q] <= ACC_W'(signed'(bias_data_i));
          if (32'(lane_q) == PARALLEL - 1) begin
            // Enter the MAC loop for this oc_group at t = 0.
            t_q       <= '0;
            out_row_q <= '0;
            j_q       <= '0;
            ic_q      <= '0;
            k_q       <= '0;
            for (int l = 0; l < PARALLEL; l++) acc_q[l] <= bias_q[l];
            // Note: bias_q[lane_q] is written this cycle; for the last lane
            // the fresh value must seed acc directly.
            acc_q[lane_q] <= ACC_W'(signed'(bias_data_i));
            state_q <= ST_MAC;
          end else begin
            lane_q  <= lane_q + 1'b1;
            state_q <= ST_LD_B_A;
          end
        end

        // --- MAC streaming: one operand fetch per cycle -------------------
        ST_MAC: begin
          if (issue_v) begin
            feat_frame_o <= win_base_q + FR_W'(32'(t_q) + 32'(k_q));
            feat_coef_o  <= ic_q;
            for (int l = 0; l < PARALLEL; l++) begin
              krn_addr_o[l] <= lane_base_q[l] + KA_W'(j_q);
            end
            j_q <= j_q + 1'b1;
            if (32'(ic_q) == IN_CH - 1) begin
              ic_q <= '0;
              k_q  <= k_q + 1'b1;
            end else begin
              ic_q <= ic_q + 1'b1;
            end
          end else if (!v1_q && !v2_q) begin
            lane_q  <= '0;
            state_q <= ST_REQ;
          end
        end

        // --- requantize + ReLU + write, one lane per cycle ----------------
        ST_REQ: begin
          act_wr_en_o   <= 1'b1;
          act_wr_addr_o <= out_row_q + OA_W'(32'(oc_col_q) + 32'(lane_q));
          act_wr_data_o <= DATA_W'(req_y);

          if (32'(lane_q) == PARALLEL - 1) begin
            // Row complete: advance t, or advance oc_group, or finish.
            j_q  <= '0;
            ic_q <= '0;
            k_q  <= '0;
            for (int l = 0; l < PARALLEL; l++) acc_q[l] <= bias_q[l];

            if (!mac_last_t) begin
              t_q       <= t_q + 1'b1;
              out_row_q <= out_row_q + OA_W'(OUT_CH);
              state_q   <= ST_MAC;
            end else if (!mac_last_grp) begin
              oc_col_q <= oc_col_q + G_W'(PARALLEL);
              for (int l = 0; l < PARALLEL; l++) begin
                lane_base_q[l] <= lane_base_q[l] + KA_W'(PARALLEL*CONV_K*IN_CH);
              end
              lane_q  <= '0;
              state_q <= ST_LD_B_A;
            end else begin
              done_o  <= 1'b1;
              state_q <= ST_IDLE;
            end
          end else begin
            lane_q <= lane_q + 1'b1;
          end
        end

        default: state_q <= ST_IDLE;
      endcase
    end
  end

`ifndef SYNTHESIS
  initial begin
    assert (OUT_CH % PARALLEL == 0)
      else $error("conv1d_engine: PARALLEL must divide OUT_CH");
    assert (WINDOW_LEN > CONV_K)
      else $error("conv1d_engine: window shorter than kernel");
  end
  // Accumulators must stay inside ACC_W (guaranteed by weight export bounds).
  always_ff @(posedge clk_i) begin
    if (v2_q) begin
      for (int l = 0; l < PARALLEL; l++) begin
        automatic logic signed [ACC_W+1:0] chk =
          (ACC_W+2)'(acc_q[l])
          + (ACC_W+2)'(signed'(feat_data_i) * signed'(krn_data_i[l]));
        assert ((chk <= (2**(ACC_W-1))-1) && (chk >= -(2**(ACC_W-1))))
          else $error("conv1d_engine: accumulator overflow lane %0d", l);
      end
    end
  end
`endif

endmodule : conv1d_engine

`default_nettype wire
