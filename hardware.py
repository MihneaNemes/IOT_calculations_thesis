import board
import busio
import RPi.GPIO as GPIO
import time
import statistics
from adafruit_ads1x15.ads1115 import ADS1115
from adafruit_ads1x15.analog_in import AnalogIn
from picamera2 import Picamera2

class CameraManager:
    
    def __init__(self):
        self.camera = None
        self.is_initialized = False
        self.initialize()
    
    def initialize(self):
        try:
            self.camera = Picamera2()
            self.camera.configure(self.camera.create_still_configuration())
            self.camera.start()
            self.is_initialized = True
        except Exception as e:
            print(f"Camera Error: {e}")
            self.is_initialized = False
    
    def capture_array(self):
        if not self.is_initialized or not self.camera:
            raise RuntimeError("Camera not initialized")
        return self.camera.capture_array()
    
    def shutdown(self):
        if self.camera:
            try:
                self.camera.stop()
                self.camera.close()
            except Exception as e:
                print(f"Error closing camera: {e}")


class SensorManager:
    def __init__(self):
        self.is_ready = False
        self.PWM_PIN = 18
        self.V_IN = 3.3
        self.R_TOP = 1000.0
        self.R1 = 4700.0
        self.R2 = 4700.0
        self.CALIBRATION_RATIO = 0.003079

        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.PWM_PIN, GPIO.OUT)
            self.pwm = GPIO.PWM(self.PWM_PIN, 10000)
            i2c = busio.I2C(board.SCL, board.SDA)
            self.ads = ADS1115(i2c)
            self.ads.gain = 1
            self.chan = AnalogIn(self.ads, 0)
            self.is_ready = True
        except Exception as e:
            print(f"Failed to initialize BIA sensor: {e}")
            self.is_ready = False

    def get_impedance(self, num_readings=50, delay=0.005):
        if not self.is_ready:
            print("Sensor not ready!")
            return None
        try:
            print("Starting PWM...")
            self.pwm.start(50)
            print("PWM started, waiting to stabilize...")
            time.sleep(1.0)

            _ = self.chan.voltage 
            time.sleep(0.01)    

            samples = []
            for i in range(num_readings):
                v = self.chan.voltage
                samples.append(v)
                time.sleep(delay)

            self.pwm.stop()

            valid_samples = [v for v in samples if v > 0.01]
            if len(valid_samples) < 5:
                print(f"Not enough valid readings: {len(valid_samples)}/{num_readings}")
                return None

            v_out = statistics.mean(valid_samples)
            print(f"Sensor Output: {round(v_out, 3)}V (from {len(valid_samples)} valid samples)")

            v_electrode2 = v_out * 2.0
            current = v_electrode2 / (self.R1 + self.R2)
            if current <= 0.000001:
                print("Current too low - check electrode contact")
                return None

            r_total = self.V_IN / current
            r_raw = r_total - self.R_TOP - self.R1 - self.R2

            r_body = r_raw * self.CALIBRATION_RATIO
            r_body = r_body

            print(f"R_raw: {r_raw:.0f}Ω → R_body (calibrated): {r_body:.1f}Ω")
            return r_body

        except Exception as e:
            print(f"Error reading impedance: {e}")
            self.pwm.stop()
            return None
