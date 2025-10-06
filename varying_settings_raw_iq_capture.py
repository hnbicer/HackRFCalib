#!/usr/bin/env python3
"""
Minimal HackRF IQ sweep with SigGen control, simple validity checks, and optional Welch PSD.

Edit CONFIG below, then run:  python hackrf_iq_sweep_minimal.py
Outputs:
  /<OUTDIR>/HackRFOne_<serial16>_<timestamp>/
      summary.csv                     ← one row per (freq, rate, LNA, VGA, AMP)
      sr_10Msps/L16_V16_A0/iq_f2450MHz_1048576S.npz  ← IQ + metadata (+ optional PSD)
      (optional) sr_.../plots/fXXXXMHz.png           ← PSD with tone marker (if MAKE_PLOT)

Calibration logic:
  SigGen frequency = tuned center + TONE_OFFSET_KHZ
  - Integrate Welch PSD around the tone bin to get tone power in dBFS.
  - expected_input_dbm = SIGGEN_POWER_DBM - CABLE_LOSS_DB
  - offset_db = expected_input_dbm - tone_power_dbfs  (use this to map dBFS → dBm)
"""

import csv
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import SoapySDR
from SoapySDR import *

# ─────────────────────────────────────────────────────────────────────────────
# User config (EDIT THESE)
# ─────────────────────────────────────────────────────────────────────────────
CONFIG = {
    # Sweep grid
    "FREQ_START_MHZ": 100,
    "FREQ_STOP_MHZ":  6000,
    "FREQ_STEP_MHZ":  100,

    # Settings grids
    "SAMPLE_RATES": [1e6],         # e.g., [10e6, 4e6]
    "LNA_LIST":     [0,8,16],           # e.g., [0, 16, 24, 32, 40]
    "VGA_LIST":     [20],           # e.g., [0, 16, 32, 48]
    "AMP_LIST":     [0],         # 0=off, >0=on

    # Capture
    "SAMPLES":      1_048_576,      # complex samples per point
    "READ_BLOCK":   262_144,        # internal read chunk
    "TIMEOUT_US":   300_000,        # per read
    "SETTLE_S":     0.6,            # after retune
    "WARMUP_READS": 1,              # discard reads after retune
    "RETRIES":      3,

    # SigGen (RF ON/OFF is handled automatically)
    "SIGGEN_IP":        "10.173.170.235",
    "SIGGEN_POWER_DBM": -30.0,      # output at SigGen
    "CABLE_LOSS_DB":     0.0,       # loss from SigGen to SDR input

    # Put the tone slightly off DC to avoid DC spur
    "TONE_OFFSET_KHZ": 100.0,       # SigGen = center + this offset

    # Welch PSD (optional)
    "USE_WELCH":  True,
    "NPERSEG":    8192,
    "NOVERLAP":   None,             # None → NPERSEG//2
    "TONE_RBW_HZ": 5000.0,          # integrate ±RBW/2 around the tone

    # Output & optional PNGs
    "OUTDIR":    ".",
    "MAKE_PLOT": False,             # True = save a PSD PNG per capture
}

# ─────────────────────────────────────────────────────────────────────────────
# SigGen control
# ─────────────────────────────────────────────────────────────────────────────
from SigGen import SigGen  # assumes your SigGen class is importable

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def find_single_hackrf_serial():
    devices = SoapySDR.Device.enumerate()
    hackrfs = [d for d in devices if "driver" in d and d["driver"] == "hackrf"]
    if not hackrfs:
        raise RuntimeError("No HackRF devices found.")
    if len(hackrfs) > 1:
        raise RuntimeError("Multiple HackRF devices found; connect only one.")
    info = hackrfs[0]
    serial_full = info["serial"]
    if not serial_full or len(serial_full) < 16:
        raise RuntimeError("Invalid or missing HackRF serial.")
    return serial_full[-16:], info

def _set_amp(dev, on):
    if isinstance(on, (int, float)):
        on = (on > 0)
    dev.setGain(SOAPY_SDR_RX, 0, "AMP", 1 if on else 0)

def setup_stream(dev, sample_rate):
    dev.setSampleRate(SOAPY_SDR_RX, 0, float(sample_rate))
    rx = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
    dev.activateStream(rx)
    return rx

def teardown_stream(dev, rx):
    try: dev.deactivateStream(rx)
    except Exception: pass
    try: dev.closeStream(rx)
    except Exception: pass

def capture_iq(dev, rx, want_samples, read_block, timeout_us, warmup_reads, retries):
    buf = np.empty(read_block, np.complex64)

    # Warm-up reads (no retries)
    for _ in range(max(0, warmup_reads)):
        try:
            dev.readStream(rx, [buf], read_block, timeoutUs=timeout_us)
        except Exception:
            pass  # Ignore warmup errors

    chunks = []
    total = 0
    overflows = 0
    attempts_left = retries

    while total < want_samples:
        try:
            sr = dev.readStream(rx, [buf], read_block, timeoutUs=timeout_us)
            if sr.ret < 0:
                raise RuntimeError(f"SoapySDR read error: {sr.ret}")
            if sr.flags & SOAPY_SDR_OVERFLOW:
                overflows += 1
            if sr.ret > 0:
                take = min(sr.ret, want_samples - total)
                chunks.append(buf[:take].copy())
                total += take
            else:
                time.sleep(0.01)

        except Exception as e:
            if attempts_left > 0:
                attempts_left -= 1
                print(f"[Retrying] Exception occurred: {e}. Retries left: {attempts_left}")
                time.sleep(0.05)
                continue
            else:
                raise RuntimeError(f"Exceeded maximum retries. Last error: {e}")

    return (np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]), overflows

def quick_metrics(iq: np.ndarray):
    iq = np.asarray(iq, dtype=np.complex64)
    nan_inf = not np.isfinite(iq.view(np.float32)).all()
    mean_i, mean_q = float(np.mean(iq.real)), float(np.mean(iq.imag))
    rms2 = float(np.mean((iq.real**2 + iq.imag**2)))
    pwr_dbfs = 10.0 * np.log10(max(rms2, 1e-30))
    clip = np.mean((np.abs(iq.real) >= 0.999) | (np.abs(iq.imag) >= 0.999))
    return dict(n=int(iq.size), nan_inf=bool(nan_inf), mean_i=mean_i, mean_q=mean_q,
                rms=np.sqrt(max(rms2, 0.0)), pwr_dbfs=float(pwr_dbfs), clip_frac=float(clip))

def welch_psd(iq: np.ndarray, fs: float, nperseg: int, noverlap: int | None):
    try:
        from scipy.signal import welch
    except Exception:
        return None, None
    if noverlap is None:
        noverlap = nperseg // 2
    f, Pxx = welch(iq, fs=fs, nperseg=nperseg, noverlap=noverlap,
                   return_onesided=False, scaling="density")
    Pxx = np.maximum(Pxx, 1e-30)
    Pxx_dbfs_per_hz = 10.0 * np.log10(Pxx)
    return f.astype(np.float32), Pxx_dbfs_per_hz.astype(np.float32)

def integrate_tone_power_dbfs(f_hz: np.ndarray, psd_dbfs_per_hz: np.ndarray,
                              tone_hz: float, rbw_hz: float) -> float:
    """Integrate PSD around tone within ±rbw/2 (two-sided), return power in dBFS."""
    if f_hz is None or psd_dbfs_per_hz is None or f_hz.size == 0:
        return np.nan
    # convert dBFS/Hz → linear FS^2/Hz
    S = 10.0 ** (psd_dbfs_per_hz / 10.0)
    # pick bins inside the window
    lo, hi = tone_hz - rbw_hz/2, tone_hz + rbw_hz/2
    mask = (f_hz >= lo) & (f_hz <= hi)
    if not np.any(mask):
        # fallback: take nearest bin
        idx = np.argmin(np.abs(f_hz - tone_hz))
        mask = np.zeros_like(f_hz, dtype=bool); mask[idx] = True
        # approximate RBW by local bin width
        if f_hz.size > 1:
            df = float(np.median(np.diff(np.sort(f_hz))))
        else:
            df = rbw_hz
        p_lin = S[mask].sum() * df
    else:
        # integrate with trapezoid in case of nonuniform bins
        f_sel, S_sel = f_hz[mask], S[mask]
        sort = np.argsort(f_sel)
        f_sel, S_sel = f_sel[sort], S_sel[sort]
        p_lin = np.trapz(S_sel, f_sel)
    return 10.0 * np.log10(max(p_lin, 1e-30))

def maybe_plot_psd(out_png: Path, f_hz, psd_dbfs_per_hz, tone_hz):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    plt.figure(figsize=(8, 4.5))
    plt.plot(f_hz/1e3, psd_dbfs_per_hz)
    plt.axvline(tone_hz/1e3, linestyle="--")
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("PSD (dBFS/Hz)")
    plt.title("Welch PSD")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main(C):
    # Discover device
    serial16, dev_kwargs = find_single_hackrf_serial()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    root = ensure_dir(Path(C["OUTDIR"]) / f"HackRFOne_{serial16}_{ts}")
    csv_path = root / "summary.csv"
    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "serial16","timestamp","freq_MHz","rate_Sps","LNA_dB","VGA_dB","AMP_on",
            "nsamp","rms_dbfs","clip_frac","nan_inf","overflows",
            "tone_offset_kHz","tone_rbw_Hz",
            "tone_power_dbfs","expected_input_dbm","offset_db",
            "iq_npz_relpath","psd_png_relpath"
        ])

    # SigGen setup
    sg = SigGen(C["SIGGEN_IP"])
    sg.RF(True)
    sg.set(power=C["SIGGEN_POWER_DBM"])

    # Device
    dev = SoapySDR.Device(dev_kwargs)
    dev.setAntenna(SOAPY_SDR_RX, 0, dev.listAntennas(SOAPY_SDR_RX, 0)[0])

    try:
        for rate in C["SAMPLE_RATES"]:
            rx = setup_stream(dev, rate)
            try:
                rate_dir = ensure_dir(root / f"sr_{int(rate/1e6)}Msps")
                plot_dir = ensure_dir(rate_dir / "plots") if C["MAKE_PLOT"] else None

                for lna in C["LNA_LIST"]:
                    dev.setGain(SOAPY_SDR_RX, 0, "LNA", int(lna))
                    for vga in C["VGA_LIST"]:
                        dev.setGain(SOAPY_SDR_RX, 0, "VGA", int(vga))
                        for amp in C["AMP_LIST"]:
                            _set_amp(dev, amp)
                            amp_on = 1 if amp > 0 else 0
                            set_dir = ensure_dir(rate_dir / f"L{lna}_V{vga}_A{amp_on}")

                            for f_mhz in range(C["FREQ_START_MHZ"], C["FREQ_STOP_MHZ"] + 1, C["FREQ_STEP_MHZ"]):
                                # Tune SDR center
                                dev.setFrequency(SOAPY_SDR_RX, 0, f_mhz * 1e6)
                                # Set SigGen tone at center + offset
                                f_sig = f_mhz * 1e6 + C["TONE_OFFSET_KHZ"] * 1e3
                                sg.set(freq=f_sig / 1e6)  # SigGen expects MHz
                                time.sleep(C["SETTLE_S"])

                                # capture (with basic retry loop)
                                last_err = None
                                for attempt in range(1, C["RETRIES"] + 1):
                                    try:
                                        iq, over = capture_iq(
                                            dev, rx, C["SAMPLES"], C["READ_BLOCK"], C["TIMEOUT_US"],
                                            C["WARMUP_READS"], C["RETRIES"]
                                        )
                                        break
                                    except RuntimeError as e:
                                        last_err = e
                                        if attempt == C["RETRIES"]:
                                            raise
                                        time.sleep(0.1)

                                mets = quick_metrics(iq)

                                # Welch PSD & tone integration
                                tone_dbfs = np.nan
                                f_psd = Pxx = None
                                if C["USE_WELCH"]:
                                    f_psd, Pxx = welch_psd(iq, fs=rate, nperseg=C["NPERSEG"], noverlap=C["NOVERLAP"])
                                    if f_psd is not None:
                                        tone_dbfs = integrate_tone_power_dbfs(
                                            f_psd, Pxx, tone_hz=C["TONE_OFFSET_KHZ"] * 1e3, rbw_hz=C["TONE_RBW_HZ"]
                                        )

                                expected_input_dbm = C["SIGGEN_POWER_DBM"] - C["CABLE_LOSS_DB"]
                                offset_db = expected_input_dbm - tone_dbfs if np.isfinite(tone_dbfs) else np.nan

                                # Save NPZ
                                npz_path = set_dir / f"iq_f{f_mhz}MHz_{C['SAMPLES']}S.npz"
                                np.savez_compressed(
                                    npz_path,
                                    iq=iq.astype(np.complex64),
                                    meta=np.array([{
                                        "serial16": serial16,
                                        "timestamp": ts,
                                        "freq_MHz": int(f_mhz),
                                        "rate_Sps": float(rate),
                                        "LNA_dB": int(lna),
                                        "VGA_dB": int(vga),
                                        "AMP_on": int(amp_on),
                                        "nsamp": int(mets["n"]),
                                        "rms_dbfs": float(mets["pwr_dbfs"]),
                                        "clip_frac": float(mets["clip_frac"]),
                                        "nan_inf": bool(mets["nan_inf"]),
                                        "overflows": int(over),
                                        "tone_offset_kHz": float(C["TONE_OFFSET_KHZ"]),
                                        "tone_rbw_Hz": float(C["TONE_RBW_HZ"]),
                                        "tone_power_dbfs": float(tone_dbfs),
                                        "expected_input_dbm": float(expected_input_dbm),
                                        "offset_db": float(offset_db),
                                    }], dtype=object),
                                    psd_freq=f_psd if f_psd is not None else np.array([], dtype=np.float32),
                                    psd_dbfs_per_hz=Pxx if Pxx is not None else np.array([], dtype=np.float32),
                                )

                                # Optional plot
                                png_rel = ""
                                if C["MAKE_PLOT"] and (f_psd is not None):
                                    png_path = (plot_dir / f"psd_f{f_mhz}MHz.png")
                                    maybe_plot_psd(png_path, f_psd, Pxx, tone_hz=C["TONE_OFFSET_KHZ"] * 1e3)
                                    png_rel = png_path.relative_to(root).as_posix()

                                # CSV append
                                with open(csv_path, "a", newline="") as fcsv:
                                    writer = csv.writer(fcsv)
                                    writer.writerow([
                                        serial16, ts, int(f_mhz), float(rate), int(lna), int(vga), int(amp_on),
                                        int(mets["n"]), float(mets["pwr_dbfs"]), float(mets["clip_frac"]), int(mets["nan_inf"]),
                                        int(over), float(C["TONE_OFFSET_KHZ"]), float(C["TONE_RBW_HZ"]),
                                        float(tone_dbfs), float(expected_input_dbm), float(offset_db),
                                        (npz_path.relative_to(root).as_posix()),
                                        png_rel
                                    ])

                                print(f"[OK] f={f_mhz:>5} MHz | sr={rate/1e6:>4.1f} Msps | "
                                      f"L{lna:>2} V{vga:>2} A{amp_on} | "
                                      f"RMS={mets['pwr_dbfs']:>6.2f} dBFS | "
                                      f"Tone={tone_dbfs:>6.2f} dBFS | Cal(ofs)={offset_db:>6.2f} dB | "
                                      f"clip={mets['clip_frac']*100:>4.1f}% | over={over}")

            finally:
                teardown_stream(dev, rx)
    finally:
        # cleanup SigGen
        try:
            sg.RF(False)
            sg.close()
        except Exception:
            pass

if __name__ == "__main__":
    main(CONFIG)
