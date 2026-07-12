// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : uart_rx
// Purpose : 8N1 UART receiver with 2-FF input synchronizer and triple mid-bit
//           majority sampling for glitch immunity.
//
// Operation: a synchronized falling edge arms the receiver; the bit counter
// then samples each bit cell at mid-1, mid, mid+1 clocks and takes the
// majority. A low stop bit raises frame_err_o (data is discarded).
//
// FSM (see docs/fsm.md): IDLE -> START -> DATA(x8) -> STOP -> IDLE
//
// Timing: CLKS_PER_BIT = CLK_FREQ_HZ / BAUD_RATE, integer division; at
// 12 MHz / 115200 = 104.17 -> 104 (0.16% error, well within the 2% budget).
// -----------------------------------------------------------------------------
`default_nettype none

module uart_rx #(
  parameter int unsigned CLK_FREQ_HZ = 12_000_000,
  parameter int unsigned BAUD_RATE   = 115_200
) (
  input  wire        clk_i,
  input  wire        rst_ni,
  input  wire        rxd_i,        // asynchronous serial input
  output logic [7:0] data_o,       // received byte
  output logic       valid_o,      // 1-cycle strobe, data_o valid
  output logic       frame_err_o   // 1-cycle strobe, bad stop bit
);

  localparam int unsigned CLKS_PER_BIT = CLK_FREQ_HZ / BAUD_RATE;
  localparam int unsigned CNT_W = $clog2(CLKS_PER_BIT);
  localparam int unsigned MID   = CLKS_PER_BIT / 2;

  // Input synchronizer (rxd_i crosses from the pad clock domain)
  logic [1:0] sync_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) sync_q <= 2'b11;
    else         sync_q <= {sync_q[0], rxd_i};
  end
  wire rxd_s = sync_q[1];

  typedef enum logic [1:0] { ST_IDLE, ST_START, ST_DATA, ST_STOP } state_e;
  state_e            state_q;
  logic [CNT_W-1:0]  clk_cnt_q;   // position within current bit cell
  logic [2:0]        bit_idx_q;   // data bit index 0..7
  logic [7:0]        shift_q;     // LSB-first assembly register
  logic [2:0]        samp_q;      // three mid-bit samples

  wire cell_end = (clk_cnt_q == CNT_W'(CLKS_PER_BIT - 1));
  wire maj      = (samp_q[0] & samp_q[1]) | (samp_q[1] & samp_q[2])
                | (samp_q[0] & samp_q[2]);

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      state_q     <= ST_IDLE;
      clk_cnt_q   <= '0;
      bit_idx_q   <= '0;
      shift_q     <= '0;
      samp_q      <= '0;
      data_o      <= '0;
      valid_o     <= 1'b0;
      frame_err_o <= 1'b0;
    end else begin
      valid_o     <= 1'b0;
      frame_err_o <= 1'b0;

      // Majority sample capture around the bit-cell midpoint
      if (clk_cnt_q == CNT_W'(MID - 1)) samp_q[0] <= rxd_s;
      if (clk_cnt_q == CNT_W'(MID    )) samp_q[1] <= rxd_s;
      if (clk_cnt_q == CNT_W'(MID + 1)) samp_q[2] <= rxd_s;

      unique case (state_q)
        ST_IDLE: begin
          clk_cnt_q <= '0;
          bit_idx_q <= '0;
          if (!rxd_s) state_q <= ST_START;  // start bit edge
        end

        ST_START: begin
          if (cell_end) begin
            clk_cnt_q <= '0;
            // Confirm the start bit at its midpoint; reject glitches.
            state_q <= maj ? ST_IDLE : ST_DATA;
          end else begin
            clk_cnt_q <= clk_cnt_q + 1'b1;
          end
        end

        ST_DATA: begin
          if (cell_end) begin
            clk_cnt_q <= '0;
            shift_q   <= {maj, shift_q[7:1]};  // LSB first
            if (bit_idx_q == 3'd7) state_q <= ST_STOP;
            else                   bit_idx_q <= bit_idx_q + 1'b1;
          end else begin
            clk_cnt_q <= clk_cnt_q + 1'b1;
          end
        end

        ST_STOP: begin
          if (cell_end) begin
            clk_cnt_q <= '0;
            state_q   <= ST_IDLE;
            if (maj) begin
              data_o  <= shift_q;
              valid_o <= 1'b1;
            end else begin
              frame_err_o <= 1'b1;  // stop bit low: framing error
            end
          end else begin
            clk_cnt_q <= clk_cnt_q + 1'b1;
          end
        end

        default: state_q <= ST_IDLE;
      endcase
    end
  end

`ifndef SYNTHESIS
  initial begin
    assert (CLKS_PER_BIT >= 16)
      else $error("uart_rx: CLKS_PER_BIT=%0d too small for majority sampling",
                  CLKS_PER_BIT);
  end
`endif

endmodule : uart_rx

`default_nettype wire
