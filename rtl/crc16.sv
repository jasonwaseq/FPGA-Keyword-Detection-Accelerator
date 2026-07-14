// -----------------------------------------------------------------------------
// Project : iCE40 KWS Accelerator
// Module  : crc16
// Purpose : Byte-parallel CRC16-CCITT-FALSE engine (poly 0x1021, init 0xFFFF,
//           MSB-first, no reflection, no final XOR).
//
// One byte is folded into the running CRC per clock when en_i is high; the
// eight polynomial steps are unrolled combinationally (a few levels of XOR,
// far below the 83 ns cycle budget at 12 MHz). clear_i re-seeds to 0xFFFF and
// takes precedence over en_i.
//
// Shared by packet_decoder (check) and packet_encoder (generate); the same
// next-state function guarantees the two ends can never disagree.
// -----------------------------------------------------------------------------
`default_nettype none

module crc16 #(
  parameter logic [15:0] POLY = 16'h1021,
  parameter logic [15:0] INIT = 16'hFFFF
) (
  input  wire         clk_i,
  input  wire         rst_ni,
  input  wire         clear_i,   // re-seed to INIT (wins over en_i)
  input  wire         en_i,      // fold data_i into the CRC this cycle
  input  wire   [7:0] data_i,
  output logic [15:0] crc_o
);

  // Function-name assignment (not 'return') keeps the Yosys SV frontend happy.
  function automatic logic [15:0] crc16_next(input logic [15:0] c,
                                             input logic [7:0]  d);
    logic [15:0] x;
    begin
      x = c ^ {d, 8'h00};
      for (int i = 0; i < 8; i++) begin
        x = x[15] ? ((x << 1) ^ POLY) : (x << 1);
      end
      crc16_next = x;
    end
  endfunction

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni)      crc_o <= INIT;
    else if (clear_i) crc_o <= INIT;
    else if (en_i)    crc_o <= crc16_next(crc_o, data_i);
  end

endmodule : crc16

`default_nettype wire
