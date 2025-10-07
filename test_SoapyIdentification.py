#!/usr/bin/env python3
# list_sdrs_soapy.py
import sys
import SoapySDR
from SoapySDR import *  # SOAPY_SDR_*

FRIENDLY = {
    "rtlsdr": "RTL-SDR",
    "hackrf": "HackRF",
    "lime": "LimeSDR",
    "airspy": "Airspy",
    "bladerf": "bladeRF",
    "uhd": "USRP (UHD)",
    "plutosdr": "ADI PlutoSDR",
    "soapyremote": "Soapy Remote",
}

def summarize_kwargs(kwargs):
    g = kwargs.get
    driver = g("driver", "unknown")
    label  = g("label", "")
    serial = g("serial", g("serial_number", ""))
    manu   = g("manufacturer", "")
    prod   = g("product", "")
    hw     = g("hardware", "")
    idx    = g("index", "")
    name   = FRIENDLY.get(driver, driver)
    bits = []
    if manu or prod: bits.append(f"{manu} {prod}".strip())
    if serial:       bits.append(f"SN:{serial}")
    if hw:           bits.append(hw)
    if idx != "":    bits.append(f"idx:{idx}")
    tail = "  •  ".join([b for b in bits if b])
    if not tail and label:
        tail = label
    return name, driver, tail

def list_devices():
    devs = SoapySDR.Device.enumerate()
    if not devs:
        print("No SDR devices found via SoapySDR.")
        return []

    print(f"Found {len(devs)} device(s):\n")
    for i, kw in enumerate(devs):
        name, driver, tail = summarize_kwargs(kw)
        print(f"[{i}] {name}  (driver='{driver}')")
        if tail:
            print(f"    {tail}")
        try:
            dev = SoapySDR.Device(kw)
            rx_ch = dev.getNumChannels(SOAPY_SDR_RX)
            tx_ch = dev.getNumChannels(SOAPY_SDR_TX)
            antennas = dev.listAntennas(SOAPY_SDR_RX, 0) if rx_ch else []
            print(f"    RX chans: {rx_ch}  |  TX chans: {tx_ch}  |  RX0 antennas: {antennas}")
            dev.unmake(dev)  # explicit close
        except Exception as e:
            print(f"    (Could not open to query: {e})")
        print()
    return devs

def any_connected(driver_substr: str) -> bool:
    """Case-insensitive substring match on driver name (e.g., 'rtlsdr', 'hackrf')."""
    for kw in SoapySDR.Device.enumerate():
        if driver_substr.lower() in kw.get("driver", "").lower():
            return True
    return False

if __name__ == "__main__":
    # List everything
    devs = list_devices()

    print("Quick checks:")
    print(f"  Any RTL-SDR connected?  {'YES' if any_connected('rtlsdr') else 'NO'}")
    print(f"  Any HackRF connected?   {'YES' if any_connected('hackrf') else 'NO'}")

    if len(sys.argv) > 1:
        drv = sys.argv[1]
        print(f"  • Any '{drv}' connected?  {'YES' if any_connected(drv) else 'NO'}")
