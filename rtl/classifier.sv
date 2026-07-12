// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : classifier
// Purpose : Final INT8 dense layer over the pooled activations, producing one
//           signed logit per class, plus a combinational argmax (winner index
//           and value).
//
//   logit[c] = requant( bias[c] + sum_i pooled[i] * w[c][i] )   (no ReLU)
//
// The weight ROM layout (class-major, index = c*IN_LEN + i) matches the
// iteration order exactly, so a single continuously incrementing weight
// address walks the whole layer; the pooled-RAM address is the inner index i.
// Reads have 2-cycle latency and stream at one MAC per cycle with a 2-stage
// valid shifter, identical in style to conv1d_engine.
//
// Cycle cost: 5 (M/S load) + NUM_CLASSES * (3 + IN_LEN + 2 + 1) ~= 509 at
// defaults - negligible next to the convolution.
//
// FSM: IDLE -> LD_M_A -> LD_M_W -> LD_M_C -> LD_S_W -> LD_S_C
//        -> [LD_B_A -> LD_B_W -> LD_B_C -> MAC -> REQ] x NUM_CLASSES -> IDLE
// -----------------------------------------------------------------------------
`default_nettype none

module classifier #(
  parameter int unsigned DATA_W      = kws_pkg::DATA_W,
  parameter int unsigned ACC_W       = kws_pkg::ACC_W,
  parameter int unsigned MULT_W      = kws_pkg::MULT_W,
  parameter int unsigned IN_LEN      = kws_pkg::DENSE_IN,
  parameter int unsigned NUM_CLASSES = kws_pkg::NUM_CLASSES
) (
  input  wire         clk_i,
  input  wire         rst_ni,

  // Control
  input  wire         start_i,
  output logic        busy_o,
  output logic        done_o,       // 1-cycle strobe; logits_o valid

  // Pooled activation RAM read port (2-cycle latency)
  output logic [$clog2(IN_LEN)-1:0]              pool_rd_addr_o,
  input  wire  [DATA_W-1:0]                      pool_rd_data_i,

  // Dense weight ROM (2-cycle latency)
  output logic [$clog2(NUM_CLASSES*IN_LEN)-1:0]  wgt_addr_o,
  input  wire  [DATA_W-1:0]                      wgt_data_i,

  // Bias/parameter ROM (2-cycle latency)
  output logic [$clog2(NUM_CLASSES+2)-1:0]       bias_addr_o,
  input  wire  [31:0]                            bias_data_i,

  // Results
  output logic [NUM_CLASSES-1:0][DATA_W-1:0]     logits_o,
  output logic [$clog2(NUM_CLASSES)-1:0]         winner_idx_o,
  output logic signed [DATA_W-1:0]               winner_val_o
);

  localparam int unsigned J_W = $clog2(IN_LEN + 1);
  localparam int unsigned C_W = $clog2(NUM_CLASSES);
  localparam int unsigned W_AW = $clog2(NUM_CLASSES * IN_LEN);

  typedef enum logic [3:0] {
    ST_IDLE,
    ST_LD_M_A, ST_LD_M_W, ST_LD_M_C,
    ST_LD_S_W, ST_LD_S_C,
    ST_LD_B_A, ST_LD_B_W, ST_LD_B_C,
    ST_MAC, ST_REQ
  } state_e;

  state_e                  state_q;
  logic [MULT_W-1:0]       m_q;
  logic [4:0]              s_q;
  logic signed [ACC_W-1:0] acc_q;
  logic [C_W-1:0]          cls_q;
  logic [J_W-1:0]          j_q;
  logic [W_AW-1:0]         waddr_q;   // runs 0..NUM_CLASSES*IN_LEN-1 monotonically
  logic                    v1_q, v2_q;

  wire issue_v  = (state_q == ST_MAC) && (j_q != J_W'(IN_LEN));
  wire last_cls = (32'(cls_q) == NUM_CLASSES - 1);

  logic signed [DATA_W-1:0] req_y;
  activation_unit #(
    .ACC_W  (ACC_W),
    .MULT_W (MULT_W),
    .DATA_W (DATA_W)
  ) u_requant (
    .acc_i     (acc_q),
    .mult_i    (m_q),
    .shift_i   (s_q),
    .relu_en_i (1'b0),        // logits stay signed
    .y_o       (req_y)
  );

  argmax #(
    .N      (NUM_CLASSES),
    .DATA_W (DATA_W)
  ) u_argmax (
    .values_i (logits_o),
    .idx_o    (winner_idx_o),
    .max_o    (winner_val_o)
  );

  assign busy_o = (state_q != ST_IDLE);

  // The parameter ROM is uniformly 32-bit; only ACC_W-bit biases and
  // MULT_W/5-bit requant fields are consumed here.
  if (ACC_W < 32) begin : g_bias_unused
    wire _unused_bias_hi = &{1'b0, bias_data_i[31:ACC_W]};
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q        <= ST_IDLE;
      m_q            <= '0;
      s_q            <= '0;
      acc_q          <= '0;
      cls_q          <= '0;
      j_q            <= '0;
      waddr_q        <= '0;
      v1_q           <= 1'b0;
      v2_q           <= 1'b0;
      done_o         <= 1'b0;
      pool_rd_addr_o <= '0;
      wgt_addr_o     <= '0;
      bias_addr_o    <= '0;
      logits_o       <= '0;
    end else begin
      done_o <= 1'b0;

      v1_q <= issue_v;
      v2_q <= v1_q;
      if (v2_q) begin
        acc_q <= acc_q + ACC_W'(signed'(pool_rd_data_i) * signed'(wgt_data_i));
      end

      unique case (state_q)
        ST_IDLE: begin
          if (start_i) begin
            cls_q   <= '0;
            waddr_q <= '0;
            state_q <= ST_LD_M_A;
          end
        end

        ST_LD_M_A: begin
          bias_addr_o <= ($clog2(NUM_CLASSES+2))'(NUM_CLASSES);
          state_q     <= ST_LD_M_W;
        end
        ST_LD_M_W: state_q <= ST_LD_M_C;
        ST_LD_M_C: begin
          m_q         <= bias_data_i[MULT_W-1:0];
          bias_addr_o <= ($clog2(NUM_CLASSES+2))'(NUM_CLASSES + 1);
          state_q     <= ST_LD_S_W;
        end
        ST_LD_S_W: state_q <= ST_LD_S_C;
        ST_LD_S_C: begin
          s_q     <= bias_data_i[4:0];
          state_q <= ST_LD_B_A;
        end

        ST_LD_B_A: begin
          bias_addr_o <= ($clog2(NUM_CLASSES+2))'(cls_q);
          state_q     <= ST_LD_B_W;
        end
        ST_LD_B_W: state_q <= ST_LD_B_C;
        ST_LD_B_C: begin
          acc_q   <= ACC_W'(signed'(bias_data_i));
          j_q     <= '0;
          state_q <= ST_MAC;
        end

        ST_MAC: begin
          if (issue_v) begin
            pool_rd_addr_o <= ($clog2(IN_LEN))'(j_q);
            wgt_addr_o     <= waddr_q;
            waddr_q        <= waddr_q + 1'b1;
            j_q            <= j_q + 1'b1;
          end else if (!v1_q && !v2_q) begin
            state_q <= ST_REQ;
          end
        end

        ST_REQ: begin
          logits_o[cls_q] <= DATA_W'(req_y);
          if (last_cls) begin
            done_o  <= 1'b1;
            state_q <= ST_IDLE;
          end else begin
            cls_q   <= cls_q + 1'b1;
            state_q <= ST_LD_B_A;
          end
        end

        default: state_q <= ST_IDLE;
      endcase
    end
  end

`ifndef SYNTHESIS
  always_ff @(posedge clk_i) begin
    if (v2_q) begin
      automatic logic signed [ACC_W+1:0] chk =
        (ACC_W+2)'(acc_q)
        + (ACC_W+2)'(signed'(pool_rd_data_i) * signed'(wgt_data_i));
      assert ((chk <= (2**(ACC_W-1))-1) && (chk >= -(2**(ACC_W-1))))
        else $error("classifier: accumulator overflow");
    end
  end
`endif

endmodule : classifier

`default_nettype wire
