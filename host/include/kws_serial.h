/* -----------------------------------------------------------------------------
 * Project : iCE40 KWS Accelerator
 * File    : kws_serial.h
 * Purpose : Minimal cross-platform serial port API (Win32 / POSIX termios)
 *           with non-blocking reads, used for the FPGA UART link.
 * ---------------------------------------------------------------------------*/
#ifndef KWS_SERIAL_H
#define KWS_SERIAL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct kws_serial kws_serial_t;

/* port: "COM7" (Windows) or "/dev/ttyUSB1" (POSIX; iCEBreaker UART is the
 * second FTDI channel). Returns NULL on failure with a message in err. */
kws_serial_t *kws_serial_open(const char *port, int baud,
                              char *err, size_t errlen);

/* Write all n bytes. Returns n or -1 on error. */
int kws_serial_write(kws_serial_t *s, const uint8_t *buf, int n);

/* Non-blocking read of up to n bytes. Returns bytes read (0 if none) or -1. */
int kws_serial_read(kws_serial_t *s, uint8_t *buf, int n);

void kws_serial_close(kws_serial_t *s);

#ifdef __cplusplus
}
#endif
#endif /* KWS_SERIAL_H */
