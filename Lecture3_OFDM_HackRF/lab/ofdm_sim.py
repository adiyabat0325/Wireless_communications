#!/usr/bin/env python3
"""
Lab 3 (Session 1) - OFDM BER and channel-capacity simulation
Wireless Communication Engineering II, Lecture 3: OFDM for wireless broadband

Reproduces the main results of the lecture with Monte-Carlo simulation and
compares them to theory:

  Part A  Two-path channel, single carrier BPSK  -> ISI error floor (slide 9)
          vs. OFDM + cyclic prefix + FDE on the same channel
  Part B  OFDM over a frequency-selective Rayleigh channel:
          average BER for BPSK/QPSK/16QAM/64QAM vs. theory (slide 18)
  Part C  Average channel capacity vs. bandwidth (slides 10 and 19):
          ideal noise / band-limited noise / single carrier with ISI / OFDM

Usage:
    python3 ofdm_sim.py                 # run everything, figures -> ./results
    python3 ofdm_sim.py --part A        # only one part
    python3 ofdm_sim.py --nsym 4000     # more OFDM symbols = smoother curves

Students: the TODO markers show where you are asked to change parameters
(see the lab guide, Lab 3).
"""
import argparse
import os

import numpy as np
from scipy.special import erfc

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(2026)
OUT = "results"


# --------------------------------------------------------------------------
# Modulation helpers (Gray-coded square QAM, unit average symbol energy)
# --------------------------------------------------------------------------
def _gray_pam(bits_per_dim):
    m = 2 ** bits_per_dim
    levels = np.arange(-(m - 1), m, 2, dtype=float)       # -3,-1,1,3 ...
    gray = np.arange(m) ^ (np.arange(m) >> 1)
    table = np.empty(m)
    table[gray] = levels
    return table


MODS = {"BPSK": 1, "QPSK": 2, "16QAM": 4, "64QAM": 6}


def modulate(bits, mod):
    k = MODS[mod]
    if mod == "BPSK":
        return 1.0 - 2.0 * bits.astype(float)
    b = bits.reshape(-1, k)
    h = k // 2
    pam = _gray_pam(h)
    w = 2 ** np.arange(h - 1, -1, -1)
    i = pam[b[:, :h] @ w]
    q = pam[b[:, h:] @ w]
    norm = np.sqrt(2 * (2 ** k - 1) / 3)
    return (i + 1j * q) / norm


def demodulate(sym, mod):
    k = MODS[mod]
    if mod == "BPSK":
        return (sym.real < 0).astype(np.uint8)
    h = k // 2
    norm = np.sqrt(2 * (2 ** k - 1) / 3)
    pam = _gray_pam(h)
    m = 2 ** h

    def slice_dim(x):
        idx = np.clip(np.round((x * norm + (m - 1)) / 2), 0, m - 1).astype(int)
        lev = np.arange(-(m - 1), m, 2)[idx]
        code = np.searchsorted(np.sort(pam), lev)  # position in sorted levels
        order = np.argsort(pam)                   # gray code of each sorted level
        g = order[code]
        return ((g[:, None] >> np.arange(h - 1, -1, -1)) & 1).astype(np.uint8)

    return np.hstack([slice_dim(sym.real), slice_dim(sym.imag)]).ravel()


# Approximate BER  Pb ~ a * Q(sqrt(b * gamma)),  gamma = Es/N0 (per subcarrier)
BER_COEF = {"BPSK": (1.0, 2.0), "QPSK": (1.0, 1.0),
            "16QAM": (3 / 4, 1 / 5), "64QAM": (7 / 12, 1 / 21)}


def Q(x):
    return 0.5 * erfc(x / np.sqrt(2))


def ber_awgn_theory(gamma, mod):
    a, b = BER_COEF[mod]
    return a * Q(np.sqrt(b * gamma))


def ber_rayleigh_theory(gamma_avg, mod):
    """Average of a*Q(sqrt(b*gamma)) over the exponential PDF of the SNR
    f(gamma) = 1/gamma_avg * exp(-gamma/gamma_avg)   (lecture slide 18)."""
    a, b = BER_COEF[mod]
    x = b * gamma_avg / 2
    return a / 2 * (1 - np.sqrt(x / (1 + x)))


# --------------------------------------------------------------------------
# OFDM transmitter / channel / receiver
# --------------------------------------------------------------------------
def ofdm_link(mod, snr_db, n_fft=64, cp=16, nsym=2000, taps_fn=None, use_cp=True):
    """Return BER of an OFDM link with ZF frequency-domain equalisation.

    taps_fn() returns a new channel impulse response h (normalised so that
    E[sum |h_l|^2] = 1) for every OFDM symbol (block fading)."""
    k = MODS[mod]
    nbits = n_fft * k * nsym
    bits = RNG.integers(0, 2, nbits, dtype=np.uint8)
    S = modulate(bits, mod).reshape(nsym, n_fft)               # s~(k)
    s = np.fft.ifft(S, axis=1) * np.sqrt(n_fft)                # IFFT, unit power
    g = cp if use_cp else 0
    tx = np.hstack([s[:, n_fft - g:], s]) if g else s          # add CP
    sigma2 = 10 ** (-snr_db / 10)                              # P = 1 per sample
    Y = np.empty(S.shape, dtype=complex)
    for i in range(nsym):
        h = taps_fn()
        # linear convolution of the whole stream would carry ISI from the previous
        # symbol; emulate it by prepending the tail of the previous symbol
        prev = tx[i - 1] if i > 0 else np.zeros(tx.shape[1], complex)
        x = np.hstack([prev, tx[i]])
        y = np.convolve(x, h)[len(prev):len(prev) + tx.shape[1]]
        y += np.sqrt(sigma2 / 2) * (RNG.standard_normal(y.size) + 1j * RNG.standard_normal(y.size))
        y = y[g:]                                              # remove CP
        Hk = np.fft.fft(h, n_fft)                              # h~(k)
        Y[i] = np.fft.fft(y) / np.sqrt(n_fft) / Hk             # FFT + ZF FDE
    bits_hat = demodulate(Y.ravel(), mod)
    return np.mean(bits_hat != bits)


def exp_pdp_taps(L=8, decay=0.6):
    pdp = decay ** np.arange(L)
    pdp /= pdp.sum()

    def draw():
        return np.sqrt(pdp / 2) * (RNG.standard_normal(L) + 1j * RNG.standard_normal(L))
    return draw


# --------------------------------------------------------------------------
# Part A: ISI in single carrier vs. OFDM on a two-path channel (slide 9)
# --------------------------------------------------------------------------
def part_a(nsym):
    snr_db = np.arange(0, 31, 2)
    plt.figure(figsize=(7, 5))
    for zeta, c in zip((0.0, 0.1, 0.5), ("C0", "C2", "C3")):
        # single carrier BPSK, h = [1, sqrt(zeta)] with delay = 1 symbol
        h1 = np.sqrt(zeta)
        n = 200_000
        b = RNG.integers(0, 2, n)
        x = 1 - 2.0 * b
        ber_sc = []
        for s in snr_db:
            sig2 = 10 ** (-s / 10)
            y = x + h1 * np.concatenate([[0], x[:-1]]) + np.sqrt(sig2 / 2) * RNG.standard_normal(n)  # Re{complex noise}
            ber_sc.append(np.mean((y < 0) != b))
        gamma_i = 1.0 / (zeta + 10 ** (-snr_db / 10))            # SINR (slide 9)
        plt.semilogy(snr_db, 0.5 * erfc(np.sqrt(gamma_i)), c + "-",
                     label=f"single carrier, SINR formula (slide 9), zeta={zeta}")
        plt.semilogy(snr_db, np.maximum(ber_sc, 1e-7), c + "o", ms=4,
                     label=f"single carrier sim, zeta={zeta}")
        # OFDM with CP on the same (fixed) channel
        if zeta > 0:
            ber_o = [ofdm_link("BPSK", s, nsym=nsym // 4, taps_fn=lambda: np.array([1.0, h1]))
                     for s in snr_db]
            plt.semilogy(snr_db, np.maximum(ber_o, 1e-7), c + "--x", ms=5,
                         label=f"OFDM+CP+FDE sim, zeta={zeta}")
    plt.ylim(1e-6, 1)
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("SNR = P/sigma^2 [dB]")
    plt.ylabel("Bit error rate")
    plt.title("Part A: ISI error floor (single carrier) vs OFDM")
    plt.legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "partA_isi_vs_ofdm.png"), dpi=150)
    print("Part A done -> results/partA_isi_vs_ofdm.png")


# --------------------------------------------------------------------------
# Part B: OFDM average BER over frequency-selective Rayleigh fading (slide 18)
# --------------------------------------------------------------------------
def part_b(nsym):
    snr_db = np.arange(0, 31, 3)
    taps = exp_pdp_taps(L=8, decay=0.6)     # TODO (students): try L=2, L=16
    plt.figure(figsize=(7, 5))
    for mod, c in zip(MODS, ("C3", "C2", "C0", "C1")):
        ber = [ofdm_link(mod, s, nsym=nsym, taps_fn=taps) for s in snr_db]
        g = 10 ** (snr_db / 10)
        plt.semilogy(snr_db, ber_rayleigh_theory(g, mod), c + "-", label=f"{mod} theory (Rayleigh)")
        plt.semilogy(snr_db, np.maximum(ber, 1e-7), c + "o", ms=4, label=f"{mod} OFDM sim")
        plt.semilogy(snr_db, ber_awgn_theory(g, mod), c + ":", lw=1)
        print(f"  {mod:6s}: simulated BER @ 21 dB = {ber[list(snr_db).index(21)]:.2e}, "
              f"theory = {ber_rayleigh_theory(10**2.1, mod):.2e}")
    plt.ylim(1e-6, 1)
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel("Average SNR per subcarrier [dB]")
    plt.ylabel("Bit error rate")
    plt.title("Part B: OFDM over frequency-selective Rayleigh (dotted = AWGN)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "partB_ofdm_rayleigh_ber.png"), dpi=150)
    print("Part B done -> results/partB_ofdm_rayleigh_ber.png")


# --------------------------------------------------------------------------
# Part C: Average channel capacity vs. bandwidth (slides 10 and 19)
# --------------------------------------------------------------------------
def part_c(nreal=400):
    B = np.linspace(0.05e6, 10e6, 120)       # bandwidth [Hz]
    snr_ref_db, B_ref = 20.0, 100e3          # SNR = 20 dB @ 100 kHz (lecture)
    dtau = 1e-6                              # two-path delay 1 us (lecture)
    gamma_ref = 10 ** (snr_ref_db / 10)
    P_over_N0 = gamma_ref * B_ref            # P/N0 is fixed; sigma^2 = B*N0

    c_ideal = B * np.log2(1 + gamma_ref)     # noise power does not grow with B
    c_awgn = B * np.log2(1 + P_over_N0 / B)  # band-limited noise sigma^2 = B N0

    c_sc, c_ofdm = np.zeros_like(B), np.zeros_like(B)
    for _ in range(nreal):
        h0, h1 = (RNG.standard_normal(2) + 1j * RNG.standard_normal(2)) / 2  # E|h|^2 = 1/2 each
        for j, b in enumerate(B):
            gamma = P_over_N0 / b
            Ts = 1 / b
            # single carrier with sinc pulse g(t), receive SINR (slide 10)
            i = np.arange(-50, 51)
            g = np.sinc((i * Ts - dtau) / Ts)       # g(iTs - dtau)
            sig = abs(h0 * 1.0 + h1 * np.sinc(-dtau / Ts)) ** 2
            isi = np.sum(abs(h1 * g[i != 0]) ** 2)
            c_sc[j] += b * np.log2(1 + sig / (isi + 1 / gamma))
            # OFDM: K subcarriers, delta_f << 1/dtau, C = sum df log2(1+gamma_k)
            K = 256
            df = b / K
            k = np.arange(K) - K // 2
            hk = h0 + h1 * np.exp(-2j * np.pi * k * df * dtau)
            c_ofdm[j] += np.sum(df * np.log2(1 + gamma * abs(hk) ** 2))
    c_sc /= nreal
    c_ofdm /= nreal

    plt.figure(figsize=(7, 5))
    plt.plot(B / 1e6, c_ideal / 1e6, "k--", label="Ideal noise (sigma^2 fixed)")
    plt.plot(B / 1e6, c_awgn / 1e6, "b-", label="Band-limited noise, flat channel")
    plt.plot(B / 1e6, c_sc / 1e6, "r-", label="Single carrier with ISI (2-path Rayleigh)")
    plt.plot(B / 1e6, c_ofdm / 1e6, "g-", label="OFDM, 2-path Rayleigh")
    plt.ylim(0, 10)
    plt.grid(alpha=0.3)
    plt.xlabel("Bandwidth [MHz]")
    plt.ylabel("Average channel capacity [Mbps]")
    plt.title("Part C: capacity vs bandwidth (SNR=20 dB @100 kHz, dtau=1 us)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "partC_capacity_vs_bw.png"), dpi=150)
    for b_mhz in (1, 5, 10):
        j = np.argmin(abs(B - b_mhz * 1e6))
        print(f"  B={b_mhz:2d} MHz: AWGN {c_awgn[j]/1e6:.2f}  SC+ISI {c_sc[j]/1e6:.2f}  "
              f"OFDM {c_ofdm[j]/1e6:.2f} Mbps")
    print("Part C done -> results/partC_capacity_vs_bw.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--nsym", type=int, default=1500, help="OFDM symbols per SNR point")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.part in ("A", "all"):
        part_a(args.nsym)
    if args.part in ("B", "all"):
        part_b(args.nsym)
    if args.part in ("C", "all"):
        part_c()
