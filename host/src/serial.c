/* -----------------------------------------------------------------------------
 * Project : iCE40 KWS Accelerator
 * File    : serial.c
 * Purpose : Serial port backends (see kws_serial.h).
 * ---------------------------------------------------------------------------*/
#include "kws_serial.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>

struct kws_serial { HANDLE h; };

kws_serial_t *kws_serial_open(const char *port, int baud,
                              char *err, size_t errlen)
{
    char path[64];
    snprintf(path, sizeof(path), "\\\\.\\%s", port);

    HANDLE h = CreateFileA(path, GENERIC_READ | GENERIC_WRITE, 0, NULL,
                           OPEN_EXISTING, 0, NULL);
    if (h == INVALID_HANDLE_VALUE) {
        snprintf(err, errlen, "cannot open %s (error %lu)",
                 port, (unsigned long)GetLastError());
        return NULL;
    }

    DCB dcb;
    memset(&dcb, 0, sizeof(dcb));
    dcb.DCBlength = sizeof(dcb);
    GetCommState(h, &dcb);
    dcb.BaudRate = (DWORD)baud;
    dcb.ByteSize = 8;
    dcb.Parity   = NOPARITY;
    dcb.StopBits = ONESTOPBIT;
    dcb.fBinary  = TRUE;
    dcb.fDtrControl = DTR_CONTROL_ENABLE;
    dcb.fRtsControl = RTS_CONTROL_ENABLE;
    if (!SetCommState(h, &dcb)) {
        snprintf(err, errlen, "SetCommState failed on %s", port);
        CloseHandle(h);
        return NULL;
    }

    COMMTIMEOUTS to;
    memset(&to, 0, sizeof(to));
    to.ReadIntervalTimeout = MAXDWORD;   /* non-blocking reads */
    SetCommTimeouts(h, &to);
    PurgeComm(h, PURGE_RXCLEAR | PURGE_TXCLEAR);

    kws_serial_t *s = (kws_serial_t *)calloc(1, sizeof(*s));
    s->h = h;
    return s;
}

int kws_serial_write(kws_serial_t *s, const uint8_t *buf, int n)
{
    DWORD done = 0;
    if (!WriteFile(s->h, buf, (DWORD)n, &done, NULL) || (int)done != n) {
        return -1;
    }
    return n;
}

int kws_serial_read(kws_serial_t *s, uint8_t *buf, int n)
{
    DWORD done = 0;
    if (!ReadFile(s->h, buf, (DWORD)n, &done, NULL)) return -1;
    return (int)done;
}

void kws_serial_close(kws_serial_t *s)
{
    if (!s) return;
    CloseHandle(s->h);
    free(s);
}

#else /* POSIX */

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <errno.h>

struct kws_serial { int fd; };

static speed_t baud_const(int baud)
{
    switch (baud) {
    case 9600:    return B9600;
    case 19200:   return B19200;
    case 38400:   return B38400;
    case 57600:   return B57600;
    case 115200:  return B115200;
#ifdef B230400
    case 230400:  return B230400;
#endif
#ifdef B921600
    case 921600:  return B921600;
#endif
    default:      return 0;
    }
}

kws_serial_t *kws_serial_open(const char *port, int baud,
                              char *err, size_t errlen)
{
    speed_t sp = baud_const(baud);
    if (!sp) { snprintf(err, errlen, "unsupported baud %d", baud); return NULL; }

    int fd = open(port, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd < 0) {
        snprintf(err, errlen, "cannot open %s: %s", port, strerror(errno));
        return NULL;
    }

    struct termios t;
    if (tcgetattr(fd, &t) != 0) {
        snprintf(err, errlen, "tcgetattr failed on %s", port);
        close(fd);
        return NULL;
    }
    cfmakeraw(&t);
    cfsetispeed(&t, sp);
    cfsetospeed(&t, sp);
    t.c_cflag |= CLOCAL | CREAD;
    t.c_cc[VMIN]  = 0;
    t.c_cc[VTIME] = 0;
    if (tcsetattr(fd, TCSANOW, &t) != 0) {
        snprintf(err, errlen, "tcsetattr failed on %s", port);
        close(fd);
        return NULL;
    }
    tcflush(fd, TCIOFLUSH);

    kws_serial_t *s = (kws_serial_t *)calloc(1, sizeof(*s));
    s->fd = fd;
    return s;
}

int kws_serial_write(kws_serial_t *s, const uint8_t *buf, int n)
{
    int done = 0;
    while (done < n) {
        ssize_t r = write(s->fd, buf + done, (size_t)(n - done));
        if (r < 0) {
            if (errno == EAGAIN) { usleep(500); continue; }
            return -1;
        }
        done += (int)r;
    }
    return n;
}

int kws_serial_read(kws_serial_t *s, uint8_t *buf, int n)
{
    ssize_t r = read(s->fd, buf, (size_t)n);
    if (r < 0) return (errno == EAGAIN) ? 0 : -1;
    return (int)r;
}

void kws_serial_close(kws_serial_t *s)
{
    if (!s) return;
    close(s->fd);
    free(s);
}

#endif
