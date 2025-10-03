import SoapySDR
from SoapySDR import *  # SOAPY_SDR_ constants
import numpy as np

sdr = SoapySDR.Device("driver=hackrf") 


# Setup stream
sample_rate = 10e6
center_freq = 100e6  # Tune to FM radio as example
gain = 20

sdr.setSampleRate(SOAPY_SDR_RX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_RX, 0, center_freq)
sdr.setGain(SOAPY_SDR_RX, 0, gain)

rxStream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rxStream)

buff = np.array([0]*1024, np.complex64)
sr = sdr.readStream(rxStream, [buff], len(buff))

print(f"Read {sr.ret} samples, First few: {buff[:5]}")

# Cleanup
sdr.deactivateStream(rxStream)
sdr.closeStream(rxStream)
