#!/usr/bin/env python3
"""
Lab 5 (Session 2) - OFDM over the air with two HackRF Pro:
frame generator, channel simulator and offline OFDM receiver.

    gen  : build one OFDM frame (preamble + data) -> ofdm_tx_frames.cfile/.json
           The TX flowgraph (lab5_hackrf_iq_tx.grc) transmits this file in a loop.
    sim  : pass the TX file through a software channel (multipath, CFO, noise,
           random delay) -> rx_capture.cfile.  Use it to test the receiver
           before you have the hardware.
    rx   : offline OFDM receiver for a capture made with lab5_hackrf_iq_rx.grc:
           Schmidl & Cox timing/CFO -> fine timing (LTS) -> CP removal -> FFT ->
           channel estimation -> one-tap FDE -> pilot phase tracking -> demod.
           Prints BER, CFO, per-subcarrier SNR, channel capacity and saves plots.

Frame structure (fs = sample rate, N = FFT size, CP = cyclic prefix):

  | gap (zeros) | CP | STS (N) | 2CP | LTS (N) | LTS (N) | CP | DATA 1 | ... | CP | DATA nsym |

  STS : only even subcarriers used -> two identical halves of N/2 samples
        (Schmidl & Cox timing metric + fractional CFO)
  LTS : known BPSK on all used subcarriers, sent twice (channel estimate h~(k)
        and noise estimate)
  DATA: 48 data + 4 pilot subcarriers (N=64), like IEEE 802.11a/g

Examples:
    python3 ofdm_lab5.py gen --mod qpsk --fs 4e6
    python3 ofdm_lab5.py sim --snr 25 --cfo 3000 --taps "1,0,0,0.5j"
    python3 ofdm_lab5.py rx  --file rx_capture.cfile
    python3 ofdm_lab5.py gen --cp 0 --emulate-taps "1,0,0,0,0,0,0,0,0.6"   # ISI test
"""
import argparse
import json
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Common definitions
# ----------------------------------------------------------------------------
MODS = {"bpsk": 1, "qpsk": 2, "16qam": 4, "64qam": 6}


def carriers(N):
    """Used / pilot / data subcarrier indices (FFT bin numbers 0..N-1)."""
    half = int(round(26 * N / 64))
    used = [k for k in range(-half, half + 1) if k != 0]
    pil = [int(round(p * N / 64)) for p in (-21, -7, 7, 21)]
    data = [k for k in used if k not in pil]
    b = lambda ks: np.array([k % N for k in ks])
    return b(used), b(pil), b(data)


def _pam(m):
    lv = np.arange(-(m - 1), m, 2, dtype=float)
    gray = np.arange(m) ^ (np.arange(m) >> 1)
    t = np.empty(m)
    t[gray] = lv
    return t


def qam_mod(bits, k):
    if k == 1:
        return (1 - 2.0 * bits).astype(complex)
    b = bits.reshape(-1, k)
    h = k // 2
    w = 2 ** np.arange(h - 1, -1, -1)
    pam = _pam(2 ** h)
    s = pam[b[:, :h] @ w] + 1j * pam[b[:, h:] @ w]
    return s / np.sqrt(2 * (2 ** k - 1) / 3)


def qam_demod(sym, k):
    if k == 1:
        return (sym.real < 0).astype(np.uint8)
    h = k // 2
    m = 2 ** h
    pam = _pam(m)
    order = np.argsort(pam)          # gray codes of the sorted levels
    x = sym * np.sqrt(2 * (2 ** k - 1) / 3)

    def dim(v):
        idx = np.clip(np.round((v + (m - 1)) / 2), 0, m - 1).astype(int)
        g = order[idx]
        return ((g[:, None] >> np.arange(h - 1, -1, -1)) & 1).astype(np.uint8)
    return np.hstack([dim(x.real), dim(x.imag)]).ravel()


def known_sequences(N, seed):
    rng = np.random.default_rng(seed)
    used, pil, data = carriers(N)
    # STS: QPSK on even subcarriers only (x sqrt(2) keeps the power)
    S = np.zeros(N, complex)
    even = np.array([k for k in used if (k if k < N // 2 else k - N) % 2 == 0])
    S[even] = np.sqrt(2) * np.exp(1j * np.pi / 2 * rng.integers(0, 4, even.size) + 1j * np.pi / 4)
    # LTS: BPSK on all used subcarriers
    L = np.zeros(N, complex)
    L[used] = 1 - 2.0 * rng.integers(0, 2, used.size)
    pilots = np.array([1, 1, 1, -1], dtype=complex)
    return S, L, pilots


def frame_bits(meta):
    rng = np.random.default_rng(meta["seed"] + 1)
    _, _, data = carriers(meta["N"])
    return rng.integers(0, 2, meta["nsym"] * data.size * MODS[meta["mod"]]).astype(np.uint8)


def add_cp(x, cp):
    return np.concatenate([x[-cp:], x]) if cp > 0 else x


# ----------------------------------------------------------------------------
# gen
# ----------------------------------------------------------------------------
def cmd_gen(a):
    N, CP = a.fft, a.cp
    used, pil, data = carriers(N)
    S, L, pilots = known_sequences(N, a.seed)
    meta = dict(N=N, CP=CP, nsym=a.nsym, mod=a.mod, seed=a.seed, fs=a.fs,
                gap=a.gap, emulate_taps=a.emulate_taps, rms=a.rms)
    bits = frame_bits(meta)
    syms = qam_mod(bits, MODS[a.mod]).reshape(a.nsym, data.size)
    ifft = lambda X: np.fft.ifft(X) * np.sqrt(N)      # OFDM modulation (IFFT)
    parts = [np.zeros(a.gap, complex),
             add_cp(ifft(S), CP if CP > 0 else N // 4),          # STS keeps a CP for the plateau
             add_cp(np.concatenate([ifft(L), ifft(L)]), max(2 * CP, N // 4))]
    for i in range(a.nsym):
        X = np.zeros(N, complex)
        X[data] = syms[i]
        X[pil] = pilots * (1 if i % 2 == 0 else -1)              # simple pilot polarity
        parts.append(add_cp(ifft(X), CP))
    x = np.concatenate(parts)
    if a.emulate_taps:
        h = np.array([complex(t) for t in a.emulate_taps.split(",")])
        x = np.convolve(x, h)[:x.size]
    sig = x[a.gap:]
    x *= a.rms / np.sqrt(np.mean(abs(sig) ** 2))
    peak = np.max(abs(x))
    if peak > 0.95:
        x *= 0.95 / peak
        print(f"warning: peak limited, rms now {np.sqrt(np.mean(abs(x[a.gap:])**2)):.3f}")
    x.astype(np.complex64).tofile(a.out)
    json.dump(meta, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    Tsym = (N + CP) / a.fs
    print(f"wrote {a.out}: {x.size} samples = {x.size / a.fs * 1e3:.2f} ms per frame")
    print(f"  N={N}, CP={CP} ({CP / a.fs * 1e6:.1f} us), subcarrier spacing df={a.fs / N / 1e3:.2f} kHz,"
          f" occupied BW ~{used.size * a.fs / N / 1e6:.2f} MHz")
    print(f"  {data.size} data + {pil.size} pilot subcarriers, {a.mod.upper()},"
          f" PHY rate = {data.size * MODS[a.mod] / Tsym / 1e6:.3f} Mbps, PAPR = "
          f"{20 * np.log10(peak / a.rms):.1f} dB")


# ----------------------------------------------------------------------------
# sim (software channel)
# ----------------------------------------------------------------------------
def cmd_sim(a):
    meta = json.load(open(os.path.splitext(a.tx)[0] + ".json"))
    fs = meta["fs"]
    x = np.fromfile(a.tx, np.complex64).astype(complex)
    rng = np.random.default_rng(a.seed)
    x = np.tile(x, a.frames)
    x = np.concatenate([np.zeros(rng.integers(100, 3000), complex), x])  # unknown start
    h = np.array([complex(t) for t in a.taps.split(",")])
    y = np.convolve(x, h)[:x.size]
    n = np.arange(y.size)
    y *= np.exp(2j * np.pi * a.cfo / fs * n + 1j * rng.uniform(0, 2 * np.pi))
    p = np.mean(abs(y[abs(y) > 1e-6]) ** 2)
    sig2 = p * 10 ** (-a.snr / 10)
    y += np.sqrt(sig2 / 2) * (rng.standard_normal(y.size) + 1j * rng.standard_normal(y.size))
    y += a.dc
    y.astype(np.complex64).tofile(a.out)
    print(f"wrote {a.out}: {y.size} samples, taps={h}, CFO={a.cfo} Hz, SNR={a.snr} dB")


# ----------------------------------------------------------------------------
# rx (offline receiver)
# ----------------------------------------------------------------------------
def schmidl_cox(r, N):
    """Schmidl & Cox timing metric with half-symbol length L = N/2.
    P(d) = sum_m r*(d+m) r(d+m+L);  M(d) = |P(d)|^2 / (R1(d) R2(d))
    (normalising by the energy of both halves keeps 0 <= M <= 1 and avoids
    false peaks at the end of a burst)."""
    L = N // 2
    c = np.conj(r[:-L]) * r[L:]
    P = np.convolve(c, np.ones(L), "valid")                      # P(d)
    e = np.convolve(abs(r) ** 2, np.ones(L), "valid")
    R1, R2 = e[:-L], e[L:]                                       # energy of 1st / 2nd half
    n = min(P.size, R1.size)
    P, R1, R2 = P[:n], R1[:n], R2[:n]
    M = abs(P) ** 2 / np.maximum(R1 * R2, 1e-30)
    return M, P, np.minimum(R1, R2)


def cmd_rx(a):
    meta = json.load(open(a.meta))
    N, CP, nsym, k = meta["N"], meta["CP"], meta["nsym"], MODS[meta["mod"]]
    fs = a.fs if a.fs else meta["fs"]
    used, pil, data = carriers(N)
    S, Lk, pilots = known_sequences(N, meta["seed"])
    bits_ref = frame_bits(meta)
    cp_sts = CP if CP > 0 else N // 4
    cp_lts = max(2 * CP, N // 4)
    lts_t = np.fft.ifft(Lk) * np.sqrt(N)
    frame_len = cp_sts + N + cp_lts + 2 * N + nsym * (N + CP)

    r = np.fromfile(a.file, np.complex64).astype(complex)
    if a.max_samples:
        r = r[:a.max_samples]
    r -= np.mean(r)                                              # remove DC offset
    print(f"capture: {r.size} samples ({r.size / fs:.2f} s at {fs / 1e6:.2f} MS/s), "
          f"rms={np.sqrt(np.mean(abs(r) ** 2)):.4f}")
    M, P, Rd = schmidl_cox(r, N)
    # energy gate: a constant (DC) or silent input also gives M ~ 1, so only accept
    # positions whose energy is clearly above the noise floor (5th percentile)
    floor = np.percentile(Rd, 5)
    gate = Rd > a.gate * floor

    # --- coarse frame detection: regions where M > threshold -----------------
    above = np.flatnonzero((M > a.thr) & gate)
    starts = []
    if above.size:
        groups = np.split(above, np.flatnonzero(np.diff(above) > N) + 1)
        for g in groups:
            if g.size >= max(2, cp_sts // 4):
                d = g[np.argmax(M[g])]
                if not starts or d - starts[-1] > frame_len // 2:
                    starts.append(d)
    print(f"Schmidl & Cox: {len(starts)} frame candidates (threshold M > {a.thr})")
    if not starts:
        print("No frames found: check frequency, gains, sample rate, or lower --thr")
        sys.exit(1)

    res = dict(ber=[], cfo=[], H=[], snr_lts=[], evm=[], raw=[], eq=[], nerr=0, nbit=0)
    n = np.arange(frame_len + 4 * N)
    for d in starts[: a.max_frames]:
        if d + frame_len + 2 * N >= r.size:
            break
        eps = np.angle(P[d]) / np.pi                                 # fractional CFO [subcarriers]
        seg = r[d - 2 * N if d >= 2 * N else 0: d + frame_len + 2 * N]
        off = d - (d - 2 * N if d >= 2 * N else 0)
        nn = np.arange(seg.size)
        seg = seg * np.exp(-2j * np.pi * eps * nn / N)
        # --- fine timing + integer CFO, jointly ------------------------------
        # The STS phase only gives the CFO modulo 2 subcarriers, so we try the
        # integer candidates q and keep the one whose LTS cross-correlation
        # peak is highest; the peak position is the fine timing.
        exp_lts = off + N + cp_lts                                    # rough guess (d is inside STS CP plateau)
        lo, hi = max(exp_lts - cp_sts - N // 4, 0), exp_lts + cp_sts + N // 4
        best = None
        for q in range(-4, 5):
            sq = seg[lo:hi + N] * np.exp(-2j * np.pi * q * nn[lo:hi + N] / N)
            xc = abs(np.correlate(sq, lts_t, "valid"))
            if best is None or xc.max() > best[0]:
                best = (xc.max(), q, lo + int(np.argmax(xc)))
        _, best, t_peak = best
        t1 = t_peak - a.backoff                                       # start of LTS #1 body
        if best:
            seg = seg * np.exp(-2j * np.pi * best * nn / N)
            eps += best
        # --- fine CFO from the two identical LTS symbols --------------------
        l1, l2 = seg[t1:t1 + N], seg[t1 + N:t1 + 2 * N]
        eps2 = np.angle(np.vdot(l1, l2)) / (2 * np.pi)
        seg = seg * np.exp(-2j * np.pi * eps2 * nn / N)
        eps += eps2
        res["cfo"].append(eps * fs / N)
        # --- channel estimation (lecture: y~(k) = h~(k) s~(k) + n~(k)) -------
        Y1 = np.fft.fft(seg[t1:t1 + N]) / np.sqrt(N)
        Y2 = np.fft.fft(seg[t1 + N:t1 + 2 * N]) / np.sqrt(N)
        H = np.zeros(N, complex)
        H[used] = (Y1[used] + Y2[used]) / 2 / Lk[used]
        noise = np.mean(abs(Y1[used] - Y2[used]) ** 2) / 2
        res["H"].append(H)
        res["snr_lts"].append(abs(H[used]) ** 2 / max(noise, 1e-15))
        # --- data symbols ---------------------------------------------------
        t = t1 + 2 * N
        Z = np.zeros((nsym, N), complex)
        R = np.zeros((nsym, N), complex)
        for i in range(nsym):
            st = t + i * (N + CP) + CP                               # remove CP
            Yd = np.fft.fft(seg[st:st + N]) / np.sqrt(N)             # FFT
            R[i] = Yd
            Zi = Yd / np.where(H == 0, 1, H) if not a.no_eq else Yd.copy()   # one-tap FDE
            if not a.no_cpe:                                         # common phase error from pilots
                ref = pilots * (1 if i % 2 == 0 else -1)
                Zi *= np.exp(-1j * np.angle(np.vdot(ref, Zi[pil])))
            Z[i] = Zi
        sd = Z[:, data]
        bits_hat = qam_demod(sd.ravel(), k)
        errs = int(np.sum(bits_hat != bits_ref))
        res["ber"].append(errs / bits_ref.size)
        res["nerr"] += errs
        res["nbit"] += bits_ref.size
        ref_sym = qam_mod(bits_ref, k).reshape(nsym, data.size)
        res["evm"].append(np.mean(abs(sd - ref_sym) ** 2, axis=0))
        res["raw"].append(R[:, data].ravel())
        res["eq"].append(sd.ravel())

    nf = len(res["ber"])
    if nf == 0:
        print("Frames were detected but none could be processed (capture too short?)")
        sys.exit(1)
    ber = res["nerr"] / res["nbit"]
    cfo = np.array(res["cfo"])
    evm = np.mean(res["evm"], axis=0)
    snr_evm = 1 / evm                                               # per data subcarrier
    snr_lts = np.mean(res["snr_lts"], axis=0)
    df = fs / N
    cap = np.sum(df * np.log2(1 + snr_lts))                          # C = sum df log2(1+gamma_k)
    rate = data.size * k / ((N + CP) / fs)
    good = np.mean(np.array(res["ber"]) == 0)
    print(f"processed {nf} frames")
    print(f"  CFO        : {np.mean(cfo):+.1f} Hz (std {np.std(cfo):.1f} Hz)"
          + (f" = {np.mean(cfo) / a.fc * 1e6:+.3f} ppm at {a.fc / 1e6:.1f} MHz" if a.fc else ""))
    print(f"  mean SNR   : {10 * np.log10(np.mean(snr_evm)):.1f} dB (EVM),"
          f" {10 * np.log10(np.mean(snr_lts)):.1f} dB (LTS)")
    print(f"  min/max SNR over subcarriers: {10 * np.log10(snr_evm.min()):.1f} / "
          f"{10 * np.log10(snr_evm.max()):.1f} dB")
    print(f"  BER        : {ber:.3e}  ({res['nerr']} / {res['nbit']} bits), error-free frames {good * 100:.0f} %")
    print(f"  PHY rate   : {rate / 1e6:.3f} Mbps ({meta['mod'].upper()}, {data.size} data subcarriers)")
    print(f"  Capacity   : {cap / 1e6:.3f} Mbps  (sum over {used.size} used subcarriers, df={df / 1e3:.2f} kHz)")

    # ---------------------------- plots ------------------------------------
    os.makedirs(a.outdir, exist_ok=True)
    fig, ax = plt.subplots(3, 2, figsize=(12, 12))
    m_show = M[: min(M.size, starts[min(3, len(starts) - 1)] + frame_len)]
    ax[0, 0].plot(np.arange(m_show.size) / fs * 1e3, m_show)
    ax[0, 0].axhline(a.thr, color="r", ls="--", lw=0.8)
    ax[0, 0].set(title="Schmidl & Cox timing metric M(d)", xlabel="time [ms]", ylabel="M(d)")
    f = np.fft.fftshift(np.fft.fftfreq(1024, 1 / fs))
    seg = r[: (r.size // 1024) * 1024][: 1024 * 400].reshape(-1, 1024)
    psd = np.fft.fftshift(np.mean(abs(np.fft.fft(seg * np.hanning(1024), axis=1)) ** 2, axis=0))
    ax[0, 1].plot(f / 1e6, 10 * np.log10(psd / psd.max()))
    ax[0, 1].set(title="Received power spectrum", xlabel="frequency offset [MHz]", ylabel="dB", ylim=(-60, 3))
    Hp = np.mean(np.abs(res["H"]) ** 2, axis=0)            # average |h~(k)|^2 over frames
    H0 = res["H"][0]                                         # phase of the first frame
    kk = np.array([k if k < N // 2 else k - N for k in used])
    o = np.argsort(kk)
    ax[1, 0].plot(kk[o] * df / 1e3, 10 * np.log10(Hp[used][o] / np.max(Hp[used])), "b.-", label="|h~(k)| [dB]")
    ax2 = ax[1, 0].twinx()
    ax2.plot(kk[o] * df / 1e3, np.unwrap(np.angle(H0[used][o])), "g.", ms=3)
    ax2.set_ylabel("phase of h~(k), frame 0 [rad]", color="g")
    ax[1, 0].set(title="Estimated channel frequency response h~(k)", xlabel="subcarrier frequency [kHz]",
                 ylabel="|h~(k)| [dB]")
    raw = np.concatenate(res["raw"])
    eq = np.concatenate(res["eq"])
    ax[1, 1].plot(raw.real / np.sqrt(np.mean(abs(raw) ** 2)), raw.imag / np.sqrt(np.mean(abs(raw) ** 2)),
                  ".", ms=1, alpha=0.3, color="gray", label="before FDE")
    ax[1, 1].plot(eq.real, eq.imag, ".", ms=2, alpha=0.5, label="after FDE + CPE")
    ax[1, 1].set(title=f"Constellation ({meta['mod'].upper()})", xlim=(-2, 2), ylim=(-2, 2), aspect="equal")
    ax[1, 1].legend(loc="upper right", fontsize=8, markerscale=6)
    kd = np.array([k if k < N // 2 else k - N for k in data])
    od = np.argsort(kd)
    ax[2, 0].plot(kd[od] * df / 1e3, 10 * np.log10(snr_evm[od]), "r.-", label="SNR from EVM (data)")
    ax[2, 0].plot(kk[o] * df / 1e3, 10 * np.log10(snr_lts[o]), "k.", ms=3, label="SNR from LTS")
    ax[2, 0].set(title="SNR per subcarrier gamma_k", xlabel="subcarrier frequency [kHz]", ylabel="dB")
    ax[2, 0].legend(fontsize=8)
    ax[2, 1].semilogy(np.maximum(res["ber"], 1e-6), "o-", ms=3)
    ax[2, 1].set(title=f"BER per frame (mean {ber:.2e})", xlabel="frame #", ylabel="BER", ylim=(1e-6, 1))
    for x in ax.ravel():
        x.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(a.outdir, a.tag + "rx_report.png")
    fig.savefig(out, dpi=130)
    summary = dict(frames=nf, ber=ber, cfo_hz=float(np.mean(cfo)), snr_db=float(10 * np.log10(np.mean(snr_evm))),
                   capacity_mbps=cap / 1e6, phy_rate_mbps=rate / 1e6, error_free_frames=float(good),
                   snr_per_subcarrier_db=(10 * np.log10(snr_evm[od])).round(2).tolist(),
                   subcarrier_khz=(kd[od] * df / 1e3).round(2).tolist())
    json.dump(summary, open(os.path.join(a.outdir, a.tag + "rx_summary.json"), "w"), indent=1)
    print(f"plots -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gen", help="generate the OFDM TX frame file")
    g.add_argument("--fft", type=int, default=64)
    g.add_argument("--cp", type=int, default=16)
    g.add_argument("--nsym", type=int, default=20, help="data OFDM symbols per frame")
    g.add_argument("--mod", choices=MODS, default="qpsk")
    g.add_argument("--fs", type=float, default=4e6, help="sample rate (must match the GRC flowgraphs)")
    g.add_argument("--gap", type=int, default=800, help="zeros between frames")
    g.add_argument("--rms", type=float, default=0.25, help="baseband RMS amplitude")
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--emulate-taps", default="", help='extra multipath applied in TX, e.g. "1,0,0,0.5j"')
    g.add_argument("--out", default="ofdm_tx_frames.cfile")
    s = sp.add_parser("sim", help="software channel: TX file -> simulated capture")
    s.add_argument("--tx", default="ofdm_tx_frames.cfile")
    s.add_argument("--out", default="rx_capture.cfile")
    s.add_argument("--frames", type=int, default=40)
    s.add_argument("--snr", type=float, default=25)
    s.add_argument("--cfo", type=float, default=2000, help="carrier frequency offset [Hz]")
    s.add_argument("--taps", default="1", help='channel impulse response, e.g. "1,0,0.4-0.3j"')
    s.add_argument("--dc", type=complex, default=0.0)
    s.add_argument("--seed", type=int, default=1)
    r = sp.add_parser("rx", help="offline OFDM receiver")
    r.add_argument("--file", default="rx_capture.cfile")
    r.add_argument("--meta", default="ofdm_tx_frames.json")
    r.add_argument("--fs", type=float, default=0, help="override sample rate")
    r.add_argument("--fc", type=float, default=0, help="carrier frequency [Hz] (only for the ppm printout)")
    r.add_argument("--thr", type=float, default=0.6, help="Schmidl & Cox threshold")
    r.add_argument("--gate", type=float, default=4.0, help="energy gate above the noise floor (linear)")
    r.add_argument("--backoff", type=int, default=2, help="start FFT window this many samples inside the CP")
    r.add_argument("--no-eq", action="store_true", help="disable the frequency-domain equalizer")
    r.add_argument("--no-cpe", action="store_true", help="disable pilot phase tracking")
    r.add_argument("--max-frames", type=int, default=200)
    r.add_argument("--max-samples", type=int, default=0)
    r.add_argument("--outdir", default="results")
    r.add_argument("--tag", default="", help="prefix for output file names")
    a = ap.parse_args()
    {"gen": cmd_gen, "sim": cmd_sim, "rx": cmd_rx}[a.cmd](a)


if __name__ == "__main__":
    main()
