// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : confidence_accumulator
// Purpose : Per-class moving average of classifier logits over the last DEPTH
//           inferences, maintained as running sums:
//
//             sum[c] += logit[c] - hist[head][c];  hist[head][c] = logit[c]
//
//           avg_o[c] = sum[c] >>> log2(DEPTH)  (combinational, always current)
//
// DEPTH must be a power of two so the average is a shift. History lives in a
// dual-port EBR and is updated one class per cycle (shared adder) so that
// NUM_CLASSES=10 still fits the UP5K LC budget; the parallel-FF alternative
// was ~320 history flops plus N adder trees and pushed utilisation over 110%.
// A fold takes N + a few cycles; inferences are thousands of cycles apart.
// -----------------------------------------------------------------------------
`default_nettype none

module confidence_accumulator #(
  parameter int unsigned N      = 4,
  parameter int unsigned DATA_W = 8,
  parameter int unsigned DEPTH  = 8    // power of two
) (
  input  wire                      clk_i,
  input  wire                      rst_ni,
  input  wire                      clear_i,
  input  wire                      update_i,
  input  wire  [N-1:0][DATA_W-1:0] logits_i,
  output logic [N-1:0][DATA_W-1:0] avg_o,    // signed per-class averages
  output logic                     busy_o
);

  localparam int unsigned SH      = $clog2(DEPTH);
  localparam int unsigned SUM_W   = DATA_W + SH;
  localparam int unsigned HIST_D  = DEPTH * N;
  localparam int unsigned HA_W    = $clog2(HIST_D);
  localparam int unsigned C_W     = $clog2(N);

  typedef enum logic [2:0] {
    ST_IDLE,
    ST_CLR,
    ST_RD,
    ST_WAIT,
    ST_UPD
  } state_e;

  state_e                  state_q;
  logic [SH-1:0]           head_q;
  logic [C_W-1:0]          c_q;
  logic [HA_W-1:0]         clr_addr_q;
  logic [N-1:0][DATA_W-1:0] logits_q;
  logic signed [SUM_W-1:0] sum_q [N];

  // hist address = head * N + c  (N need not be a power of two)
  wire [HA_W-1:0] fold_addr = HA_W'(head_q) * HA_W'(N) + HA_W'(c_q);

  logic              hist_we;
  logic [HA_W-1:0]   hist_waddr;
  logic [DATA_W-1:0] hist_wdata;
  logic [HA_W-1:0]   hist_raddr;
  logic [DATA_W-1:0] hist_rdata;

  ram_dp_sync #(
    .DATA_W (DATA_W),
    .DEPTH  (HIST_D)
  ) u_hist (
    .clk_i,
    .wr_en_i   (hist_we),
    .wr_addr_i (hist_waddr),
    .wr_data_i (hist_wdata),
    .rd_addr_i (hist_raddr),
    .rd_data_o (hist_rdata)
  );

  assign busy_o = (state_q != ST_IDLE);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q    <= ST_CLR;   // zero EBR-backed history after reset
      head_q     <= '0;
      c_q        <= '0;
      clr_addr_q <= '0;
      logits_q   <= '0;
      hist_we    <= 1'b0;
      hist_waddr <= '0;
      hist_wdata <= '0;
      hist_raddr <= '0;
      for (int c = 0; c < N; c++) sum_q[c] <= '0;
    end else begin
      hist_we <= 1'b0;

      unique case (state_q)
        ST_IDLE: begin
          if (clear_i) begin
            head_q     <= '0;
            clr_addr_q <= '0;
            for (int c = 0; c < N; c++) sum_q[c] <= '0;
            state_q    <= ST_CLR;
          end else if (update_i) begin
            logits_q <= logits_i;
            c_q      <= '0;
            state_q  <= ST_RD;
          end
        end

        ST_CLR: begin
          hist_we    <= 1'b1;
          hist_waddr <= clr_addr_q;
          hist_wdata <= '0;
          if (clr_addr_q == HA_W'(HIST_D - 1)) begin
            state_q <= ST_IDLE;
          end else begin
            clr_addr_q <= clr_addr_q + 1'b1;
          end
        end

        ST_RD: begin
          hist_raddr <= fold_addr;
          state_q    <= ST_WAIT;
        end

        ST_WAIT: begin
          state_q <= ST_UPD;
        end

        ST_UPD: begin
          sum_q[c_q] <= sum_q[c_q]
                      + SUM_W'(signed'(logits_q[c_q]))
                      - SUM_W'(signed'(hist_rdata));
          hist_we    <= 1'b1;
          hist_waddr <= fold_addr;
          hist_wdata <= logits_q[c_q];
          if (c_q == C_W'(N - 1)) begin
            head_q  <= head_q + 1'b1;
            state_q <= ST_IDLE;
          end else begin
            c_q     <= c_q + 1'b1;
            state_q <= ST_RD;
          end
        end

        default: state_q <= ST_IDLE;
      endcase
    end
  end

  always_comb begin
    for (int c = 0; c < N; c++) begin
      avg_o[c] = DATA_W'(sum_q[c] >>> SH);
    end
  end

`ifndef SYNTHESIS
  initial begin
    assert (DEPTH == (1 << SH))
      else $error("confidence_accumulator: DEPTH must be a power of two");
    assert (HIST_D >= 2)
      else $error("confidence_accumulator: HIST_D too small");
  end
`endif

endmodule : confidence_accumulator

`default_nettype wire
