// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : statistics_counters
// Purpose : Sixteen 32-bit performance/health counters (map: kws_pkg
//           stat_idx_e) with a snapshot bank for tear-free readout.
//
// A READ_STATS command pulses snapshot_i; the counter bank is copied into a
// snapshot RAM over NUM cycles (one word per cycle), and the RSP_STATS
// payload is served from that RAM with a synchronous read. Each word is
// captured atomically; cross-word skew is bounded by NUM cycles (1.3 us at
// 12 MHz), invisible next to the 5.6 ms wire time of the readout. The RAM
// implementation replaces an earlier 512-flip-flop shadow register bank -
// two EBRs are far cheaper than 512 LCs on the UP5K. The copy provably
// finishes before the encoder fetches payload byte 0 (the packet header
// alone takes >= NUM cycles to emit; asserted in simulation).
//
// Wrap behaviour: all counters wrap mod 2^32. STAT_UPTIME_CYC and
// STAT_CONV_BUSY wrap in ~358 s at 12 MHz; hosts must difference successive
// snapshots (the host app does).
// -----------------------------------------------------------------------------
`default_nettype none

module statistics_counters #(
  parameter int unsigned NUM = kws_pkg::STATS_NUM
) (
  input  wire         clk_i,
  input  wire         rst_ni,
  input  wire         clear_i,          // soft reset: zero all counters

  // Event strobes
  input  wire         frame_rx_i,
  input  wire         frame_dropped_i,
  input  wire         pkt_rx_i,
  input  wire         crc_err_i,
  input  wire         framing_err_i,
  input  wire         fifo_ovf_i,
  input  wire         win_sched_i,
  input  wire         infer_done_i,
  input  wire         keyword_i,
  input  wire         win_drop_i,
  input  wire         tx_pkt_i,

  // Latency capture (with infer_done_i)
  input  wire  [31:0] latency_i,

  // Utilization levels (counted every cycle they are high)
  input  wire         conv_busy_i,

  // Snapshot + readout (synchronous read: rd_data_o valid 1 cycle after
  // rd_idx_i; the packet encoder's 2-cycle payload fetch absorbs this)
  input  wire         snapshot_i,
  input  wire  [3:0]  rd_idx_i,
  output logic [31:0] rd_data_o
);

  logic [31:0] cnt_q [NUM];

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      for (int i = 0; i < NUM; i++) begin
        cnt_q[i] <= '0;
      end
    end else if (clear_i) begin
      for (int i = 0; i < NUM; i++) cnt_q[i] <= '0;
    end else begin
      if (frame_rx_i)      cnt_q[kws_pkg::STAT_FRAMES_RX]      <= cnt_q[kws_pkg::STAT_FRAMES_RX]      + 1;
      if (frame_dropped_i) cnt_q[kws_pkg::STAT_FRAMES_DROPPED] <= cnt_q[kws_pkg::STAT_FRAMES_DROPPED] + 1;
      if (pkt_rx_i)        cnt_q[kws_pkg::STAT_PKTS_RX]        <= cnt_q[kws_pkg::STAT_PKTS_RX]        + 1;
      if (crc_err_i)       cnt_q[kws_pkg::STAT_CRC_ERRORS]     <= cnt_q[kws_pkg::STAT_CRC_ERRORS]     + 1;
      if (framing_err_i)   cnt_q[kws_pkg::STAT_FRAMING_ERRORS] <= cnt_q[kws_pkg::STAT_FRAMING_ERRORS] + 1;
      if (fifo_ovf_i)      cnt_q[kws_pkg::STAT_FIFO_OVERFLOWS] <= cnt_q[kws_pkg::STAT_FIFO_OVERFLOWS] + 1;
      if (win_sched_i)     cnt_q[kws_pkg::STAT_WINDOWS_SCHED]  <= cnt_q[kws_pkg::STAT_WINDOWS_SCHED]  + 1;
      if (keyword_i)       cnt_q[kws_pkg::STAT_KEYWORDS]       <= cnt_q[kws_pkg::STAT_KEYWORDS]       + 1;
      if (win_drop_i)      cnt_q[kws_pkg::STAT_WINDOWS_DROP]   <= cnt_q[kws_pkg::STAT_WINDOWS_DROP]   + 1;
      if (tx_pkt_i)        cnt_q[kws_pkg::STAT_TX_PKTS]        <= cnt_q[kws_pkg::STAT_TX_PKTS]        + 1;
      if (conv_busy_i)     cnt_q[kws_pkg::STAT_CONV_BUSY]      <= cnt_q[kws_pkg::STAT_CONV_BUSY]      + 1;
      cnt_q[kws_pkg::STAT_UPTIME_CYC] <= cnt_q[kws_pkg::STAT_UPTIME_CYC] + 1;

      if (infer_done_i) begin
        cnt_q[kws_pkg::STAT_INFERENCES] <= cnt_q[kws_pkg::STAT_INFERENCES] + 1;
        cnt_q[kws_pkg::STAT_LAT_LAST]   <= latency_i;
        cnt_q[kws_pkg::STAT_LAT_ACC]    <= cnt_q[kws_pkg::STAT_LAT_ACC] + latency_i;
        if (latency_i > cnt_q[kws_pkg::STAT_LAT_MAX]) begin
          cnt_q[kws_pkg::STAT_LAT_MAX] <= latency_i;
        end
      end
    end
  end

  // Snapshot RAM (EBR): sequential copy, one counter per cycle.
  logic [31:0]            snap_ram [NUM];
  logic [$clog2(NUM)-1:0] copy_idx_q;
  logic                   copy_busy_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      copy_idx_q  <= '0;
      copy_busy_q <= 1'b0;
    end else if (snapshot_i) begin
      copy_idx_q  <= '0;
      copy_busy_q <= 1'b1;
    end else if (copy_busy_q) begin
      copy_idx_q <= copy_idx_q + 1'b1;
      if (32'(copy_idx_q) == NUM - 1) copy_busy_q <= 1'b0;
    end
  end

  always_ff @(posedge clk_i) begin
    if (copy_busy_q) snap_ram[copy_idx_q] <= cnt_q[copy_idx_q];
    rd_data_o <= snap_ram[rd_idx_i];
  end

`ifndef SYNTHESIS
  // The copy must never still be running when a readout could land; the
  // encoder needs >= 15 cycles of header emission before the first fetch.
  always_ff @(posedge clk_i) begin
    assert (!(snapshot_i && copy_busy_q))
      else $error("statistics_counters: snapshot re-triggered during copy");
  end
`endif

endmodule : statistics_counters

`default_nettype wire
