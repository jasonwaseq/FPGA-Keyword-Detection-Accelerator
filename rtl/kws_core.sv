// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : kws_core
// Purpose : Complete streaming keyword-spotting accelerator, board-agnostic.
//           Instantiates and connects the full pipeline:
//
//   uart_rx -> rx_fifo -> packet_decoder -> feature_buffer -> window_scheduler
//     -> conv1d_engine -> pooling_engine -> classifier -> temporal_smoothing
//     -> keyword_detector -> [arbiter] -> packet_encoder -> tx_fifo -> uart_tx
//
// plus the control plane (register_file), observability (statistics_counters,
// interrupt_controller) and time bases (timestamp_gen).
//
// Reset domains:
//   rst_ni     - full reset (UART, protocol, everything).
//   rst_eng_n  - rst_ni gated by soft/stream clear pulses: resets the layer
//                engines and event queue so RESET/START never strand an FSM
//                mid-inference. Driven by a registered pulse, so assertion is
//                glitch-free and deassertion is clock-aligned. The protocol
//                path deliberately stays alive to deliver the ACK.
//
// Inference sequencing is a simple done->start daisy chain (conv -> pool ->
// classifier -> smoothing). Layers are NOT overlapped across windows: at 12
// MHz an inference occupies ~1.3 ms of an 80 ms stride interval (<2%), so
// inter-layer pipelining would buy nothing and cost double-buffered
// activation RAMs (tradeoff analysis in docs/pipeline.md). Reception is
// fully concurrent with inference via the dual-ported feature buffer.
// -----------------------------------------------------------------------------
`default_nettype none

module kws_core #(
  parameter int unsigned CLK_HZ       = kws_pkg::CLK_FREQ_HZ,
  parameter int unsigned BAUD_RATE    = kws_pkg::UART_BAUD,
  parameter int unsigned PARALLEL     = kws_pkg::PARALLEL_OUT_CH,
  parameter int unsigned RX_FIFO_DEPTH = 256,
  parameter int unsigned TX_FIFO_DEPTH = 128,
  parameter              CONV_W_FILE  = "weights/conv_weights.mem",
  parameter              CONV_B_FILE  = "weights/conv_bias.mem",
  parameter              DENSE_W_FILE = "weights/dense_weights.mem",
  parameter              DENSE_B_FILE = "weights/dense_bias.mem"
) (
  input  wire  clk_i,
  input  wire  rst_ni,
  input  wire  uart_rxd_i,
  output logic uart_txd_o,
  output logic irq_o,
  output logic led_keyword_o,
  output logic led_error_o,
  output logic stream_active_o
);

  localparam int unsigned FR_W  = $clog2(kws_pkg::HIST_DEPTH);
  localparam int unsigned CO_W  = $clog2(kws_pkg::NUM_MFCC);
  localparam int unsigned LEN_W = $clog2(kws_pkg::PROTO_MAX_PAYLOAD + 1);
  localparam int unsigned IDX_W = $clog2(kws_pkg::PROTO_MAX_PAYLOAD);

  // ===========================================================================
  // Time bases
  // ===========================================================================
  logic [31:0] ms, cycles;

  timestamp_gen #(.CLK_FREQ_HZ (CLK_HZ)) u_time (
    .clk_i, .rst_ni, .ms_o (ms), .cycles_o (cycles)
  );

  // ===========================================================================
  // UART receive path
  // ===========================================================================
  logic [7:0] rx_byte;
  logic       rx_valid, rx_ferr;

  uart_rx #(.CLK_FREQ_HZ (CLK_HZ), .BAUD_RATE (BAUD_RATE)) u_uart_rx (
    .clk_i, .rst_ni,
    .rxd_i       (uart_rxd_i),
    .data_o      (rx_byte),
    .valid_o     (rx_valid),
    .frame_err_o (rx_ferr)
  );

  logic       rxf_full, rxf_ovf, rxf_empty, rxf_rd_en, rxf_rd_valid;
  logic [7:0] rxf_rd_data;
  logic [$clog2(RX_FIFO_DEPTH):0] rxf_level;

  uart_fifo #(.WIDTH (8), .DEPTH (RX_FIFO_DEPTH)) u_rx_fifo (
    .clk_i, .rst_ni,
    .wr_en_i    (rx_valid),
    .wr_data_i  (rx_byte),
    .full_o     (rxf_full),
    .overflow_o (rxf_ovf),
    .rd_en_i    (rxf_rd_en),
    .rd_data_o  (rxf_rd_data),
    .rd_valid_o (rxf_rd_valid),
    .empty_o    (rxf_empty),
    .level_o    (rxf_level)
  );

  // ===========================================================================
  // Packet decoder
  // ===========================================================================
  logic            feat_wr_en, feat_commit;
  logic [CO_W-1:0] feat_wr_idx;
  logic [7:0]      feat_wr_data;
  logic [31:0]     feat_fn, feat_ts;
  logic            cmd_valid, err_valid, pkt_ok, crc_err, proto_err;
  logic [7:0]      cmd_type, err_code, err_detail;
  logic [31:0]     cmd_fn, cmd_ts;

  packet_decoder #(
    .NUM_MFCC    (kws_pkg::NUM_MFCC),
    .MAX_PAYLOAD (kws_pkg::PROTO_MAX_PAYLOAD),
    .TIMEOUT_CYCLES (CLK_HZ / 10)   // 100 ms
  ) u_dec (
    .clk_i, .rst_ni,
    .fifo_empty_i     (rxf_empty),
    .fifo_rd_en_o     (rxf_rd_en),
    .fifo_rd_data_i   (rxf_rd_data),
    .fifo_rd_valid_i  (rxf_rd_valid),
    .feat_wr_en_o     (feat_wr_en),
    .feat_wr_idx_o    (feat_wr_idx),
    .feat_wr_data_o   (feat_wr_data),
    .feat_commit_o    (feat_commit),
    .feat_frame_num_o (feat_fn),
    .feat_timestamp_o (feat_ts),
    .cmd_valid_o      (cmd_valid),
    .cmd_type_o       (cmd_type),
    .cmd_frame_num_o  (cmd_fn),
    .cmd_timestamp_o  (cmd_ts),
    .err_valid_o      (err_valid),
    .err_code_o       (err_code),
    .err_detail_o     (err_detail),
    .pkt_ok_o         (pkt_ok),
    .crc_err_o        (crc_err),
    .proto_err_o      (proto_err)
  );

  // ===========================================================================
  // Control plane
  // ===========================================================================
  logic        stream_en, soft_clear, stream_clear, snapshot;
  logic [3:0]  stats_idx;
  logic [31:0] stats_data;
  logic signed [7:0]      cfg_thresh;
  logic [3:0]             cfg_vote_min, cfg_min_consec;
  logic [7:0]             cfg_debounce;
  logic [kws_pkg::NUM_CLASSES-1:0] cfg_target_mask;
  logic                   cfg_pool_mode, cfg_smooth_en;
  logic        rsp_req, rsp_ack;
  logic [7:0]  rsp_type;
  logic [LEN_W-1:0] rsp_len;
  logic [31:0] rsp_ts, rsp_fn;
  logic [7:0]  rf_pl_data;
  logic [IDX_W-1:0] enc_pl_idx;

  register_file #(
    .CLK_HZ      (CLK_HZ),
    .N_CLASSES   (kws_pkg::NUM_CLASSES),
    .MAX_PAYLOAD (kws_pkg::PROTO_MAX_PAYLOAD),
    .PARALLEL    (PARALLEL)
  ) u_regs (
    .clk_i, .rst_ni,
    .cmd_valid_i       (cmd_valid),
    .cmd_type_i        (cmd_type),
    .err_valid_i       (err_valid),
    .err_code_i        (err_code),
    .err_detail_i      (err_detail),
    .cmd_frame_num_i   (cmd_fn),
    .ts_ms_i           (ms),
    .stream_en_o       (stream_en),
    .soft_clear_o      (soft_clear),
    .stream_clear_o    (stream_clear),
    .snapshot_o        (snapshot),
    .stats_idx_o       (stats_idx),
    .stats_data_i      (stats_data),
    .cfg_thresh_o      (cfg_thresh),
    .cfg_vote_min_o    (cfg_vote_min),
    .cfg_min_consec_o  (cfg_min_consec),
    .cfg_debounce_o    (cfg_debounce),
    .cfg_target_mask_o (cfg_target_mask),
    .cfg_pool_mode_o   (cfg_pool_mode),
    .cfg_smooth_en_o   (cfg_smooth_en),
    .rsp_req_o         (rsp_req),
    .rsp_ack_i         (rsp_ack),
    .rsp_type_o        (rsp_type),
    .rsp_len_o         (rsp_len),
    .rsp_timestamp_o   (rsp_ts),
    .rsp_frame_o       (rsp_fn),
    .pl_idx_i          (enc_pl_idx),
    .pl_data_o         (rf_pl_data)
  );

  assign stream_active_o = stream_en;

  // Clear fan-out: START re-arms the streaming state, RESET also clears
  // statistics. Engines get a registered 1-cycle reset pulse.
  wire  clr_stream = soft_clear | stream_clear;
  logic clr_pulse_q;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) clr_pulse_q <= 1'b0;
    else         clr_pulse_q <= clr_stream;
  end
  wire rst_eng_n = rst_ni & ~clr_pulse_q;

  // Feature commits honored only while streaming is enabled.
  wire commit_ok     = feat_commit & stream_en;
  wire frame_dropped = feat_commit & ~stream_en;

  // ===========================================================================
  // Feature buffer + window scheduler
  // ===========================================================================
  logic [FR_W-1:0] wr_frame, feat_rd_frame, win_base;
  logic [$clog2(kws_pkg::WINDOW_LEN):0] frames_total;
  logic [31:0]     newest_fn, newest_ts;
  logic [CO_W-1:0] feat_rd_coef;
  logic [7:0]      feat_rd_data;
  logic            issue, win_drop;
  logic [31:0]     win_fn, win_ts;

  feature_buffer #(
    .DATA_W     (kws_pkg::DATA_W),
    .NUM_MFCC   (kws_pkg::NUM_MFCC),
    .HIST_DEPTH (kws_pkg::HIST_DEPTH),
    .WINDOW_LEN (kws_pkg::WINDOW_LEN)
  ) u_fbuf (
    .clk_i, .rst_ni,
    .clear_i            (clr_stream),
    .wr_en_i            (feat_wr_en),
    .wr_coef_i          (feat_wr_idx),
    .wr_data_i          (feat_wr_data),
    .commit_i           (commit_ok),
    .frame_num_i        (feat_fn),
    .timestamp_i        (feat_ts),
    .rd_frame_i         (feat_rd_frame),
    .rd_coef_i          (feat_rd_coef),
    .rd_data_o          (feat_rd_data),
    .wr_frame_o         (wr_frame),
    .frames_total_o     (frames_total),
    .newest_frame_num_o (newest_fn),
    .newest_timestamp_o (newest_ts)
  );

  logic conv_busy, pool_busy, cls_busy, smooth_busy;
  wire  engine_busy = conv_busy | pool_busy | cls_busy | smooth_busy;

  window_scheduler #(
    .HIST_DEPTH    (kws_pkg::HIST_DEPTH),
    .WINDOW_LEN    (kws_pkg::WINDOW_LEN),
    .WINDOW_STRIDE (kws_pkg::WINDOW_STRIDE)
  ) u_sched (
    .clk_i, .rst_ni,
    .clear_i            (clr_stream),
    .enable_i           (stream_en),
    .commit_i           (commit_ok),
    .wr_frame_i         (wr_frame),
    .frames_total_i     (frames_total),
    .newest_frame_num_i (newest_fn),
    .newest_timestamp_i (newest_ts),
    .engine_busy_i      (engine_busy),
    .issue_o            (issue),
    .win_base_o         (win_base),
    .win_frame_num_o    (win_fn),
    .win_timestamp_o    (win_ts),
    .drop_o             (win_drop)
  );

  // Latency reference: cycle stamp of the most recent window issue. The
  // elapsed-time subtraction is registered so the 32-bit carry chain never
  // concatenates with its consumers (statistics adder / event capture) in
  // one cycle - lat_now is one cycle stale, an error of 83 ns on a
  // millisecond-scale measurement.
  logic [31:0] lat_start_q, lat_now;
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      lat_start_q <= '0;
      lat_now     <= '0;
    end else begin
      if (issue) lat_start_q <= cycles;
      lat_now <= cycles - lat_start_q;
    end
  end

  // ===========================================================================
  // Layer engines and their memories
  // ===========================================================================
  // --- convolution -----------------------------------------------------------
  localparam int unsigned KA_W = $clog2(kws_pkg::CONV_OUT_CH * kws_pkg::CONV_K * kws_pkg::NUM_MFCC);
  localparam int unsigned AA_W = $clog2(kws_pkg::CONV_OUT_LEN * kws_pkg::CONV_OUT_CH);

  logic                          conv_done;
  logic [PARALLEL-1:0][KA_W-1:0] krn_addr;
  logic [PARALLEL-1:0][7:0]      krn_data;
  logic [$clog2(kws_pkg::CONV_OUT_CH+2)-1:0] cbias_addr;
  logic [31:0]                   cbias_data;
  logic                          act_wr_en;
  logic [AA_W-1:0]               act_wr_addr, act_rd_addr;
  logic [7:0]                    act_wr_data, act_rd_data;

  kernel_memory #(
    .DATA_W   (kws_pkg::DATA_W),
    .OUT_CH   (kws_pkg::CONV_OUT_CH),
    .CONV_K   (kws_pkg::CONV_K),
    .IN_CH    (kws_pkg::NUM_MFCC),
    .PARALLEL (PARALLEL),
    .MEM_FILE (CONV_W_FILE)
  ) u_kmem (
    .clk_i,
    .addr_i (krn_addr),
    .data_o (krn_data)
  );

  bias_memory #(
    .N_BIAS   (kws_pkg::CONV_OUT_CH),
    .MEM_FILE (CONV_B_FILE)
  ) u_cbias (
    .clk_i,
    .addr_i (cbias_addr),
    .data_o (cbias_data)
  );

  conv1d_engine #(
    .DATA_W     (kws_pkg::DATA_W),
    .ACC_W      (kws_pkg::ACC_W),
    .MULT_W     (kws_pkg::MULT_W),
    .IN_CH      (kws_pkg::NUM_MFCC),
    .WINDOW_LEN (kws_pkg::WINDOW_LEN),
    .CONV_K     (kws_pkg::CONV_K),
    .OUT_CH     (kws_pkg::CONV_OUT_CH),
    .PARALLEL   (PARALLEL),
    .HIST_DEPTH (kws_pkg::HIST_DEPTH)
  ) u_conv (
    .clk_i,
    .rst_ni        (rst_eng_n),
    .start_i       (issue),
    .win_base_i    (win_base),
    .busy_o        (conv_busy),
    .done_o        (conv_done),
    .feat_frame_o  (feat_rd_frame),
    .feat_coef_o   (feat_rd_coef),
    .feat_data_i   (feat_rd_data),
    .krn_addr_o    (krn_addr),
    .krn_data_i    (krn_data),
    .bias_addr_o   (cbias_addr),
    .bias_data_i   (cbias_data),
    .act_wr_en_o   (act_wr_en),
    .act_wr_addr_o (act_wr_addr),
    .act_wr_data_o (act_wr_data)
  );

  // Conv activation RAM (written by conv, read by pooling)
  ram_dp_sync #(
    .DATA_W (kws_pkg::DATA_W),
    .DEPTH  (kws_pkg::CONV_OUT_LEN * kws_pkg::CONV_OUT_CH)
  ) u_act_ram (
    .clk_i,
    .wr_en_i   (act_wr_en),
    .wr_addr_i (act_wr_addr),
    .wr_data_i (act_wr_data),
    .rd_addr_i (act_rd_addr),
    .rd_data_o (act_rd_data)
  );

  // --- pooling ----------------------------------------------------------------
  localparam int unsigned PA_W = $clog2(kws_pkg::DENSE_IN);

  logic            pool_done, pool_wr_en;
  logic [PA_W-1:0] pool_wr_addr, pool_rd_addr;
  logic [7:0]      pool_wr_data, pool_rd_data;

  pooling_engine #(
    .DATA_W    (kws_pkg::DATA_W),
    .IN_LEN    (kws_pkg::CONV_OUT_LEN),
    .CH        (kws_pkg::CONV_OUT_CH),
    .POOL_SIZE (kws_pkg::POOL_SIZE)
  ) u_pool (
    .clk_i,
    .rst_ni         (rst_eng_n),
    .start_i        (conv_done),
    .mode_i         (cfg_pool_mode),
    .busy_o         (pool_busy),
    .done_o         (pool_done),
    .act_rd_addr_o  (act_rd_addr),
    .act_rd_data_i  (act_rd_data),
    .pool_wr_en_o   (pool_wr_en),
    .pool_wr_addr_o (pool_wr_addr),
    .pool_wr_data_o (pool_wr_data)
  );

  // Pooled activation RAM (written by pooling, read by classifier)
  ram_dp_sync #(
    .DATA_W (kws_pkg::DATA_W),
    .DEPTH  (kws_pkg::DENSE_IN)
  ) u_pool_ram (
    .clk_i,
    .wr_en_i   (pool_wr_en),
    .wr_addr_i (pool_wr_addr),
    .wr_data_i (pool_wr_data),
    .rd_addr_i (pool_rd_addr),
    .rd_data_o (pool_rd_data)
  );

  // --- classifier ---------------------------------------------------------------
  logic        cls_done;
  logic [$clog2(kws_pkg::NUM_CLASSES*kws_pkg::DENSE_IN)-1:0] wgt_addr;
  logic [7:0]  wgt_data;
  logic [$clog2(kws_pkg::NUM_CLASSES+2)-1:0] dbias_addr;
  logic [31:0] dbias_data;
  logic [kws_pkg::NUM_CLASSES-1:0][7:0]      logits;
  logic [$clog2(kws_pkg::NUM_CLASSES)-1:0]   winner_idx;
  logic signed [7:0]                winner_val;

  weight_memory #(
    .DATA_W      (kws_pkg::DATA_W),
    .NUM_CLASSES (kws_pkg::NUM_CLASSES),
    .IN_LEN      (kws_pkg::DENSE_IN),
    .MEM_FILE    (DENSE_W_FILE)
  ) u_dwgt (
    .clk_i,
    .addr_i (wgt_addr),
    .data_o (wgt_data)
  );

  bias_memory #(
    .N_BIAS   (kws_pkg::NUM_CLASSES),
    .MEM_FILE (DENSE_B_FILE)
  ) u_dbias (
    .clk_i,
    .addr_i (dbias_addr),
    .data_o (dbias_data)
  );

  classifier #(
    .DATA_W      (kws_pkg::DATA_W),
    .ACC_W       (kws_pkg::ACC_W),
    .MULT_W      (kws_pkg::MULT_W),
    .IN_LEN      (kws_pkg::DENSE_IN),
    .NUM_CLASSES (kws_pkg::NUM_CLASSES)
  ) u_cls (
    .clk_i,
    .rst_ni         (rst_eng_n),
    .start_i        (pool_done),
    .busy_o         (cls_busy),
    .done_o         (cls_done),
    .pool_rd_addr_o (pool_rd_addr),
    .pool_rd_data_i (pool_rd_data),
    .wgt_addr_o     (wgt_addr),
    .wgt_data_i     (wgt_data),
    .bias_addr_o    (dbias_addr),
    .bias_data_i    (dbias_data),
    .logits_o       (logits),
    .winner_idx_o   (winner_idx),
    .winner_val_o   (winner_val)
  );

  // --- temporal smoothing + event generation ----------------------------------
  logic        detect;
  logic [$clog2(kws_pkg::NUM_CLASSES)-1:0] det_class;
  logic [7:0]  det_conf;
  logic [3:0]  det_votes;

  temporal_smoothing #(
    .N      (kws_pkg::NUM_CLASSES),
    .DATA_W (kws_pkg::DATA_W),
    .DEPTH  (kws_pkg::SMOOTH_DEPTH)
  ) u_smooth (
    .clk_i,
    .rst_ni        (rst_eng_n),
    .clear_i       (1'b0),          // engine reset covers clears
    .update_i      (cls_done),
    .logits_i      (logits),
    .winner_i      (winner_idx),
    .en_i          (cfg_smooth_en),
    .thresh_i      (cfg_thresh),
    .vote_min_i    (cfg_vote_min),
    .min_consec_i  (cfg_min_consec),
    .debounce_i    (cfg_debounce),
    .target_mask_i (cfg_target_mask),
    .detect_o      (detect),
    .det_class_o   (det_class),
    .det_conf_o    (det_conf),
    .det_votes_o   (det_votes),
    .busy_o        (smooth_busy)
  );

  logic        evt_req, evt_ack, kwd_drop;
  logic [7:0]  evt_type;
  logic [LEN_W-1:0] evt_len;
  logic [31:0] evt_ts, evt_fn;
  logic [7:0]  kwd_pl_data;

  keyword_detector #(
    .N           (kws_pkg::NUM_CLASSES),
    .MAX_PAYLOAD (kws_pkg::PROTO_MAX_PAYLOAD)
  ) u_kwd (
    .clk_i,
    .rst_ni          (rst_eng_n),
    .detect_i        (detect),
    .class_i         (det_class),
    .conf_i          (det_conf),
    .votes_i         (det_votes),
    .latency_i       (lat_now),
    .frame_num_i     (win_fn),
    .ts_ms_i         (ms),
    .evt_req_o       (evt_req),
    .evt_ack_i       (evt_ack),
    .evt_type_o      (evt_type),
    .evt_len_o       (evt_len),
    .evt_timestamp_o (evt_ts),
    .evt_frame_o     (evt_fn),
    .pl_idx_i        (enc_pl_idx),
    .pl_data_o       (kwd_pl_data),
    .drop_o          (kwd_drop)
  );

  // ===========================================================================
  // Encoder arbitration (events over responses: detection latency is the KPI;
  // both sources hold their requests, so nothing is ever lost by priority)
  // ===========================================================================
  typedef enum logic [1:0] { A_IDLE, A_REQ, A_WAIT } arb_e;
  arb_e arb_q;
  logic grant_evt_q;

  logic       enc_req_valid, enc_req_ready, pkt_done;
  logic [7:0] enc_type;
  logic [LEN_W-1:0] enc_len;
  logic [31:0] enc_ts, enc_fn;
  logic [7:0]  enc_pl_data;
  logic        txf_full;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      arb_q       <= A_IDLE;
      grant_evt_q <= 1'b0;
    end else begin
      unique case (arb_q)
        A_IDLE: begin
          if (evt_req) begin
            grant_evt_q <= 1'b1;
            arb_q       <= A_REQ;
          end else if (rsp_req) begin
            grant_evt_q <= 1'b0;
            arb_q       <= A_REQ;
          end
        end
        A_REQ:  if (enc_req_ready) arb_q <= A_WAIT;  // accepted this cycle
        A_WAIT: if (pkt_done)      arb_q <= A_IDLE;
        default: arb_q <= A_IDLE;
      endcase
    end
  end

  assign enc_req_valid = (arb_q == A_REQ);
  assign evt_ack = (arb_q == A_REQ) && enc_req_ready &&  grant_evt_q;
  assign rsp_ack = (arb_q == A_REQ) && enc_req_ready && !grant_evt_q;

  assign enc_type    = grant_evt_q ? evt_type : rsp_type;
  assign enc_len     = grant_evt_q ? evt_len  : rsp_len;
  assign enc_ts      = grant_evt_q ? evt_ts   : rsp_ts;
  assign enc_fn      = grant_evt_q ? evt_fn   : rsp_fn;
  assign enc_pl_data = grant_evt_q ? kwd_pl_data : rf_pl_data;

  logic       enc_wr_en;
  logic [7:0] enc_wr_data;
  logic       txf_afull;

  packet_encoder #(
    .MAX_PAYLOAD (kws_pkg::PROTO_MAX_PAYLOAD)
  ) u_enc (
    .clk_i, .rst_ni,
    .req_valid_i     (enc_req_valid),
    .req_ready_o     (enc_req_ready),
    .req_type_i      (enc_type),
    .req_len_i       (enc_len),
    .req_timestamp_i (enc_ts),
    .req_frame_i     (enc_fn),
    .pl_idx_o        (enc_pl_idx),
    .pl_data_i       (enc_pl_data),
    .tx_wr_en_o      (enc_wr_en),
    .tx_wr_data_o    (enc_wr_data),
    .tx_full_i       (txf_afull),
    .pkt_done_o      (pkt_done)
  );

  // ===========================================================================
  // UART transmit path
  // ===========================================================================
  logic       txf_ovf, txf_empty, txf_rd_en, txf_rd_valid;
  logic [7:0] txf_rd_data;
  logic [$clog2(TX_FIFO_DEPTH):0] txf_level;
  logic       tx_ready, txp_inflight_q;

  uart_fifo #(.WIDTH (8), .DEPTH (TX_FIFO_DEPTH)) u_tx_fifo (
    .clk_i, .rst_ni,
    .wr_en_i    (enc_wr_en),
    .wr_data_i  (enc_wr_data),
    .full_o     (txf_full),
    .overflow_o (txf_ovf),
    .rd_en_i    (txf_rd_en),
    .rd_data_o  (txf_rd_data),
    .rd_valid_o (txf_rd_valid),
    .empty_o    (txf_empty),
    .level_o    (txf_level)
  );

  // Almost-full for the encoder: its registered write lags the flow-control
  // check by one cycle, so it must see 'full' two entries early (it is the
  // FIFO's only writer, so exactly one write can be in flight).
  assign txf_afull = (32'(txf_level) >= TX_FIFO_DEPTH - 2);

  // Pop pump: one byte in flight; uart_tx is guaranteed idle when the popped
  // byte arrives, so the valid/ready transfer always completes immediately.
  assign txf_rd_en = !txf_empty && tx_ready && !txp_inflight_q;

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)            txp_inflight_q <= 1'b0;
    else if (txf_rd_en)     txp_inflight_q <= 1'b1;
    else if (txf_rd_valid)  txp_inflight_q <= 1'b0;
  end

  uart_tx #(.CLK_FREQ_HZ (CLK_HZ), .BAUD_RATE (BAUD_RATE)) u_uart_tx (
    .clk_i, .rst_ni,
    .data_i  (txf_rd_data),
    .valid_i (txf_rd_valid),
    .ready_o (tx_ready),
    .txd_o   (uart_txd_o)
  );

  // ===========================================================================
  // Statistics + interrupts
  // ===========================================================================
  statistics_counters #(.NUM (kws_pkg::STATS_NUM)) u_stats (
    .clk_i, .rst_ni,
    .clear_i         (soft_clear),
    .frame_rx_i      (commit_ok),
    .frame_dropped_i (frame_dropped),
    .pkt_rx_i        (pkt_ok),
    .crc_err_i       (crc_err),
    .framing_err_i   (rx_ferr | proto_err | err_valid),
    .fifo_ovf_i      (rxf_ovf),
    .win_sched_i     (issue),
    .infer_done_i    (cls_done),
    .keyword_i       (detect),
    .win_drop_i      (win_drop),
    .tx_pkt_i        (pkt_done),
    .latency_i       (lat_now),
    .conv_busy_i     (conv_busy),
    .snapshot_i      (snapshot),
    .rd_idx_i        (stats_idx),
    .rd_data_o       (stats_data)
  );

  logic [2:0] irq_pending;

  interrupt_controller #(
    .STRETCH_CYCLES (CLK_HZ / 5)   // ~200 ms
  ) u_irq (
    .clk_i, .rst_ni,
    .clear_i      (soft_clear),
    .keyword_i    (detect),
    .error_i      (crc_err | proto_err | rx_ferr | err_valid),
    .overflow_i   (rxf_ovf),
    .ack_i        (snapshot),
    .irq_o        (irq_o),
    .pending_o    (irq_pending),       // folded into irq_o; kept for waves
    .led_keyword_o (led_keyword_o),
    .led_error_o   (led_error_o)
  );

  // ---------------------------------------------------------------------------
  // Deliberately unused signals, consumed for lint cleanliness.
  // win_ts (host timestamp of window) is carried for future event payloads;
  // winner_val/kwd_drop are covered by det_conf/assertions; FIFO levels and
  // rxf_full are debug visibility.
  // ---------------------------------------------------------------------------
  wire _unused_ok = &{1'b0, win_ts, winner_val, kwd_drop, cmd_ts, txf_full,
                      rxf_level, txf_level, rxf_full, txf_ovf, irq_pending};

`ifndef SYNTHESIS
  always_ff @(posedge clk_i) begin
    // txf_ovf resets to 0, so no reset guard is needed.
    assert (!txf_ovf)
      else $error("kws_core: TX FIFO overflow (encoder must respect full)");
  end
`endif

endmodule : kws_core

`default_nettype wire
