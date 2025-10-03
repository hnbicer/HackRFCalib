import SoapySDR

def list_devices():
    results = SoapySDR.Device.enumerate()
    if not results:
        print("No SDR devices found.")
    for i, result in enumerate(results):
        print(f"Device {i}:")
        for key, value in result.items():
            print(f"  {key} = {value}")

list_devices()
