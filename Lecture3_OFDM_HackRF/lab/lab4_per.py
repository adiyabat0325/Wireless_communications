#!/usr/bin/env python3
"""
Lab 4 - Packet error rate (PER) from the text file written by the OFDM receiver.

The transmitter repeats 1000 numbered packets:
    "NUM OFDM LAB seq=0000 | Hello from HackRF Pro TX      \\n"   (64 bytes)
The GNU Radio OFDM Receiver drops packets whose header or CRC32 check fails,
so a missing sequence number = a lost packet.

Usage:
    python3 lab4_per.py rx_packets.txt
"""
import re
import sys

MSG = "NUM OFDM LAB seq=%04d | Hello from HackRF Pro TX"


def main(path):
    text = open(path, "rb").read().decode("ascii", errors="replace")
    lines = [l for l in text.split("\n") if l.strip()]
    seqs, bad = [], 0
    for l in lines:
        m = re.search(r"seq=(\d{4})", l)
        if m and l.rstrip() == MSG % int(m.group(1)):
            seqs.append(int(m.group(1)))
        else:
            bad += 1
    if len(seqs) < 2:
        print(f"{len(lines)} lines, {len(seqs)} valid packets - not enough to compute PER")
        return
    # unwrap the 0..999 counter to count how many packets were sent in the span
    expected, prev = 1, seqs[0]
    for s in seqs[1:]:
        expected += (s - prev) % 1000 or 1000
        prev = s
    received = len(seqs)
    per = 1 - received / expected
    print(f"lines in file      : {len(lines)}")
    print(f"valid packets      : {received}")
    print(f"corrupted lines    : {bad}")
    print(f"packets sent (span): {expected}")
    print(f"PER                : {per:.4f}  ({per * 100:.2f} %)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rx_packets.txt")
