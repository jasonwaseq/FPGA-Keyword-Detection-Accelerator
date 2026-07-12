// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : pooling_engine
// Purpose : Temporal pooling over the convolution activation RAM. Reduces
//           IN_LEN time steps to IN_LEN/POOL_SIZE, per channel. Supports
//           max pooling and average pooling, selected at run time by
//           mode_i (from the register file).
//
// Average pooling uses an arithmetic right shift by log2(POOL_SIZE) (floor),
// matching the reference model exactly; POOL_SIZE must be a power of two.
//
// Memory layouts (both RAMs are time-major, channel-minor):
//   read  addr = (p*POOL_SIZE + q) * CH + ch     (activation RAM)
//   write addr =  p * CH + ch                    (pooled RAM)
// Both address sequences are generated incrementally - the iteration order
// (p outer, ch mid, q inner) visits input rows contiguously, so no multiplier
// is needed anywhere in the address path.
//
// FSM: IDLE -> {RD_A -> RD_W -> RD_C}xPOOL_SIZE -> WR -> ... -> IDLE
// Cycle cost: OUT_LEN * CH * (3*POOL_SIZE + 1) ~= 840 cycles at defaults.
// -----------------------------------------------------------------------------
`default_nettype none

module pooling_engine #(
  parameter int unsigned DATA_W    = kws_pkg::DATA_W,
  parameter int unsigned IN_LEN    = kws_pkg::CONV_OUT_LEN,
  parameter int unsigned CH        = kws_pkg::CONV_OUT_CH,
  parameter int unsigned POOL_SIZE = kws_pkg::POOL_SIZE
) (
  input  wire         clk_i,
  input  wire         rst_ni,

  // Control
  input  wire         start_i,
  input  wire         mode_i,      // 0 = max pool, 1 = average pool
  output logic        busy_o,
  output logic        done_o,      // 1-cycle strobe

  // Activation RAM read port (2-cycle latency)
  output logic [$clog2(IN_LEN*CH)-1:0]           act_rd_addr_o,
  input  wire  [DATA_W-1:0]                      act_rd_data_i,

  // Pooled RAM write port
  output logic                                   pool_wr_en_o,
  output logic [$clog2((IN_LEN/POOL_SIZE)*CH)-1:0] pool_wr_addr_o,
  output logic [DATA_W-1:0]                      pool_wr_data_o
);

  localparam int unsigned OUT_LEN = IN_LEN / POOL_SIZE;
  localparam int unsigned RA_W    = $clog2(IN_LEN * CH);
  localparam int unsigned WA_W    = $clog2(OUT_LEN * CH);
  localparam int unsigned SUM_W   = DATA_W + $clog2(POOL_SIZE);
  localparam int unsigned SH      = $clog2(POOL_SIZE);

  typedef enum logic [2:0] { ST_IDLE, ST_RD_A, ST_RD_W, ST_RD_C, ST_WR } state_e;

  state_e                   state_q;
  logic [$clog2(OUT_LEN)-1:0] p_q;
  logic [$clog2(CH)-1:0]      ch_q;
  logic [SH:0]                q_q;          // 0..POOL_SIZE-1 (+guard bit)
  logic [RA_W-1:0]            in_base_q;    // p*POOL_SIZE*CH (incremental)
  logic [WA_W-1:0]            out_idx_q;    // p*CH + ch (incremental)
  logic signed [SUM_W-1:0]    red_q;        // running max or sum

  assign busy_o = (state_q != ST_IDLE);

  // Reduction step: fold the arriving sample into red_q.
  wire signed [SUM_W-1:0] sample = SUM_W'(signed'(act_rd_data_i));
  wire signed [SUM_W-1:0] red_next =
      (q_q == '0) ? sample
    : mode_i      ? (red_q + sample)
    : ((sample > red_q) ? sample : red_q);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q        <= ST_IDLE;
      p_q            <= '0;
      ch_q           <= '0;
      q_q            <= '0;
      in_base_q      <= '0;
      out_idx_q      <= '0;
      red_q          <= '0;
      done_o         <= 1'b0;
      act_rd_addr_o  <= '0;
      pool_wr_en_o   <= 1'b0;
      pool_wr_addr_o <= '0;
      pool_wr_data_o <= '0;
    end else begin
      done_o       <= 1'b0;
      pool_wr_en_o <= 1'b0;

      unique case (state_q)
        ST_IDLE: begin
          if (start_i) begin
            p_q       <= '0;
            ch_q      <= '0;
            q_q       <= '0;
            in_base_q <= '0;
            out_idx_q <= '0;
            state_q   <= ST_RD_A;
          end
        end

        ST_RD_A: begin
          // addr = in_base + q*CH + ch ; q*CH realized by repeated addition
          // is folded into the q loop below via in_base staying constant and
          // the small product q_q*CH (CH is a localparam power constant).
          act_rd_addr_o <= in_base_q + RA_W'(32'(q_q) * CH + 32'(ch_q));
          state_q       <= ST_RD_W;
        end

        ST_RD_W: state_q <= ST_RD_C;

        ST_RD_C: begin
          red_q <= red_next;
          if (32'(q_q) == POOL_SIZE - 1) begin
            state_q <= ST_WR;
          end else begin
            q_q     <= q_q + 1'b1;
            state_q <= ST_RD_A;
          end
        end

        ST_WR: begin
          // en/addr/data are registered together, so the RAM sees a
          // consistent triple one cycle later; out_idx_q advances for the
          // next output in the same edge.
          pool_wr_en_o   <= 1'b1;
          pool_wr_addr_o <= out_idx_q;
          pool_wr_data_o <= mode_i ? DATA_W'(red_q >>> SH) : DATA_W'(red_q);
          out_idx_q      <= out_idx_q + 1'b1;
          q_q            <= '0;

          if (32'(ch_q) == CH - 1) begin
            ch_q      <= '0;
            in_base_q <= in_base_q + RA_W'(POOL_SIZE * CH);
            if (32'(p_q) == OUT_LEN - 1) begin
              done_o  <= 1'b1;
              state_q <= ST_IDLE;
            end else begin
              p_q     <= p_q + 1'b1;
              state_q <= ST_RD_A;
            end
          end else begin
            ch_q    <= ch_q + 1'b1;
            state_q <= ST_RD_A;
          end
        end

        default: state_q <= ST_IDLE;
      endcase
    end
  end

`ifndef SYNTHESIS
  initial begin
    assert (POOL_SIZE == (1 << SH))
      else $error("pooling_engine: POOL_SIZE must be a power of two");
    assert (IN_LEN % POOL_SIZE == 0)
      else $error("pooling_engine: POOL_SIZE must divide IN_LEN");
  end
`endif

endmodule : pooling_engine

`default_nettype wire
