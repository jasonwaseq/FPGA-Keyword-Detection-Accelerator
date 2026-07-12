// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : uart_tx
// Purpose : 8N1 UART transmitter with valid/ready handshake.
//
// A byte presented on data_i with valid_i high is accepted when ready_o is
// high (single-cycle transfer). The line idles high.
//
// FSM: IDLE -> START -> DATA(x8) -> STOP -> IDLE
// -----------------------------------------------------------------------------
`default_nettype none

module uart_tx #(
  parameter int unsigned CLK_FREQ_HZ = 12_000_000,
  parameter int unsigned BAUD_RATE   = 115_200
) (
  input  wire        clk_i,
  input  wire        rst_ni,
  input  wire  [7:0] data_i,
  input  wire        valid_i,
  output logic       ready_o,   // accepts data_i when high
  output logic       txd_o      // serial output, idles high
);

  localparam int unsigned CLKS_PER_BIT = CLK_FREQ_HZ / BAUD_RATE;
  localparam int unsigned CNT_W = $clog2(CLKS_PER_BIT);

  typedef enum logic [1:0] { ST_IDLE, ST_START, ST_DATA, ST_STOP } state_e;
  state_e           state_q;
  logic [CNT_W-1:0] clk_cnt_q;
  logic [2:0]       bit_idx_q;
  logic [7:0]       shift_q;

  wire cell_end = (clk_cnt_q == CNT_W'(CLKS_PER_BIT - 1));

  assign ready_o = (state_q == ST_IDLE);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q   <= ST_IDLE;
      clk_cnt_q <= '0;
      bit_idx_q <= '0;
      shift_q   <= '0;
      txd_o     <= 1'b1;
    end else begin
      unique case (state_q)
        ST_IDLE: begin
          txd_o     <= 1'b1;
          clk_cnt_q <= '0;
          bit_idx_q <= '0;
          if (valid_i) begin
            shift_q <= data_i;
            txd_o   <= 1'b0;       // start bit begins immediately
            state_q <= ST_START;
          end
        end

        ST_START: begin
          if (cell_end) begin
            clk_cnt_q <= '0;
            txd_o     <= shift_q[0];
            shift_q   <= {1'b1, shift_q[7:1]};
            state_q   <= ST_DATA;
          end else begin
            clk_cnt_q <= clk_cnt_q + 1'b1;
          end
        end

        ST_DATA: begin
          if (cell_end) begin
            clk_cnt_q <= '0;
            if (bit_idx_q == 3'd7) begin
              txd_o   <= 1'b1;     // stop bit
              state_q <= ST_STOP;
            end else begin
              txd_o     <= shift_q[0];
              shift_q   <= {1'b1, shift_q[7:1]};
              bit_idx_q <= bit_idx_q + 1'b1;
            end
          end else begin
            clk_cnt_q <= clk_cnt_q + 1'b1;
          end
        end

        ST_STOP: begin
          if (cell_end) begin
            clk_cnt_q <= '0;
            state_q   <= ST_IDLE;
          end else begin
            clk_cnt_q <= clk_cnt_q + 1'b1;
          end
        end

        default: state_q <= ST_IDLE;
      endcase
    end
  end

endmodule : uart_tx

`default_nettype wire
