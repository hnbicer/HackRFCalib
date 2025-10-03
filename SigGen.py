# !pip install pyvisa
# !pip install pyvisa-py
import pyvisa
import random
import time

class SigGen:

    def __init__(self, ip):
        print(f"CONNECTING TO IP: {ip}")
        self.N5172B = pyvisa.ResourceManager().open_resource(f'TCPIP0::{ip}::inst0::INSTR')
        self.N5172B.write(":SOURce:DM:STATE 0")  # Set modulation to OFF

        self._frequency = None  # MHz
        self._power = None  # dBm
        self._RF_on = None

    def set(self, freq=None, power=None):

        if freq:
            self._frequency = freq * 1e6
            self.N5172B.write(f":SOURce:FREQuency:CW {freq * 1e6} Hz")  # Set frequency
        if power:
            self._power = power
            self.N5172B.write(f":SOURce:POWer:LEVel:IMMediate:AMPLitude {power} DBM")  # Set Power

    def RF(self, on):
        if on:
            self.N5172B.write(":OUTPUT:STATe 1")  # Set RF ON
        elif not on:
            self.N5172B.write(":OUTPUT:STATe 0")  # Set RF OFF

    def close(self):
        self.N5172B.close()
    def __exit__(self):
        self.N5172B.close()


if __name__ == "__main__":
    ip = '10.173.170.235'
    #ip = '10.173.170.235'
    sig_gen = SigGen(ip=ip)
    sig_gen.RF(on=True)
    sig_gen.set(freq=2000, power=-30)
    count = 0
