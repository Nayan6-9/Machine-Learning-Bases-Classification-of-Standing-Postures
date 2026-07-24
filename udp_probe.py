#!/usr/bin/env python3
"""
udp_probe.py — confirm the C# bridge is actually streaming.

Run this AFTER starting WiiBoardBridge.exe and stepping on the board:
    python udp_probe.py
    python udp_probe.py --port 8674

If you see lines like  t,tr,tl,br,bl  scrolling, the bridge works and you can
run wbb_gui.py / run_wbb.py. Ctrl+C to stop.
"""
import argparse
import socket


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8674)
    args = ap.parse_args()

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((args.host, args.port))
    s.settimeout(2.0)
    print("Listening on udp://%s:%d  (Ctrl+C to stop)" % (args.host, args.port))

    n = 0
    try:
        while True:
            try:
                data, _ = s.recvfrom(256)
            except socket.timeout:
                print("  ...no data yet — is WiiBoardBridge.exe running "
                      "and the board powered on?")
                continue
            n += 1
            line = data.decode("ascii", "ignore").strip()
            if n <= 5 or n % 30 == 0:
                print("  [%d] %s" % (n, line))
    except KeyboardInterrupt:
        print("\nReceived %d datagrams total." % n)
    finally:
        s.close()


if __name__ == "__main__":
    main()
