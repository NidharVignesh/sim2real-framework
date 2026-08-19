from machine import Pin, I2C
import utime
import math

# ---------------------------------------------------------------------------
# Low-level MPU6050 driver: direct I2C register access, no external library.
# ---------------------------------------------------------------------------

MPU6050_ADDR = 0x68
PWR_MGMT_1 = 0x6B
ACCEL_XOUT_H = 0x3B  # first of 14 contiguous bytes: accel(6) temp(2) gyro(6)

ACCEL_SCALE = 16384.0  # LSB per g, at +/-2g range (sensor default)
GYRO_SCALE = 131.0     # LSB per deg/s, at +/-250 dps range (sensor default)
# If you reconfigure the sensor's range registers elsewhere, update these
# two constants to match, or the physical-unit conversion below is wrong.


def _to_signed16(high, low):
    val = (high << 8) | low
    if val >= 0x8000:
        val -= 0x10000
    return val


def init_mpu6050(i2c, addr=MPU6050_ADDR):
    # MPU6050 boots in sleep mode - clear the sleep bit to wake it
    i2c.writeto(addr, bytes([PWR_MGMT_1 , 0x00]))

def get_mpu6050_data(i2c, addr=MPU6050_ADDR):
    raw = i2c.readfrom_mem(addr, ACCEL_XOUT_H, 14)

    ax = _to_signed16(raw[0], raw[1]) / ACCEL_SCALE
    ay = _to_signed16(raw[2], raw[3]) / ACCEL_SCALE
    az = _to_signed16(raw[4], raw[5]) / ACCEL_SCALE

    temp_raw = _to_signed16(raw[6], raw[7])
    temp = temp_raw / 340.0 + 36.53

    gx = _to_signed16(raw[8], raw[9]) / GYRO_SCALE
    gy = _to_signed16(raw[10], raw[11]) / GYRO_SCALE
    gz = _to_signed16(raw[12], raw[13]) / GYRO_SCALE

    return {
        'accel': {'x': ax, 'y': ay, 'z': az},
        'gyro': {'x': gx, 'y': gy, 'z': gz},
        'temp': temp,
    }


# ---------------------------------------------------------------------------
# High-level wrapper: tilt angles + complementary-filtered pitch/roll.
# ---------------------------------------------------------------------------

class mpu6050:
    def __init__(self, axis='roll', scl=22, sda=21, freq=400000, gyro_scale=1.0):
        """
        gyro_scale: multiplier applied to gyro readings before they're
        treated as deg/s. Defaults to 1.0 since get_mpu6050_data() above
        already returns deg/s. Only change this if you swap in a
        different driver with different output units.
        """
        self.axis = axis
        self.gyro_scale = gyro_scale
        self.i2c = I2C(1, scl=Pin(scl), sda=Pin(sda), freq=freq)
        init_mpu6050(self.i2c)
        self.pitch = 0
        self.roll = 0
        self.prev_time = utime.ticks_ms()

    def calculate_tilt_angles(self, accel_data):
        x, y, z = accel_data['x'], accel_data['y'], accel_data['z']

        # rotation about X axis (roll), using Y-Z plane
        tilt_x = math.atan2(y, math.sqrt(x * x + z * z)) * 180 / math.pi
        # rotation about Y axis (pitch), using X-Z plane
        tilt_y = math.atan2(-x, math.sqrt(y * y + z * z)) * 180 / math.pi
        # NOTE: tilt_z (rotation about Z / yaw) can't actually be derived
        # from accelerometer data alone - gravity doesn't change with
        # rotation about its own axis. Kept for reference/logging only;
        # never used as a correction reference in the filter below.
        tilt_z = math.atan2(z, math.sqrt(x * x + y * y)) * 180 / math.pi

        return tilt_x, tilt_y, tilt_z

    def complementary_filter(self, angle, gyro_rate, accel_angle, dt, alpha=0.98):
        """
        Single-axis complementary filter.
        angle:       previous filtered estimate (deg)
        gyro_rate:   current gyro rate for THIS axis (deg/s, after gyro_scale)
        accel_angle: current accelerometer-derived angle for THIS axis (deg)
        dt:          time since last update (s)
        """
        gyro_estimate = angle + gyro_rate * dt
        return alpha * gyro_estimate + (1 - alpha) * accel_angle

    def read(self):
        data = get_mpu6050_data(self.i2c)
        curr_time = utime.ticks_ms()
        dt = (curr_time - self.prev_time) / 1000

        tilt_x, tilt_y, tilt_z = self.calculate_tilt_angles(data['accel'])

        gx = data['gyro']['x'] * self.gyro_scale
        gy = data['gyro']['y'] * self.gyro_scale

        # roll <-> X-axis gyro rate <-> tilt_x accel reference
        roll = self.complementary_filter(self.roll, gx, tilt_x, dt)
        # pitch <-> Y-axis gyro rate <-> tilt_y accel reference
        pitch = self.complementary_filter(self.pitch, gy, tilt_y, dt)

        self.prev_time = curr_time
        self.pitch = pitch
        self.roll = roll

        if self.axis == 'pitch':
            return pitch
        elif self.axis == 'roll':
            return roll
        elif self.axis == 'tilt_x':
            return tilt_x
        elif self.axis == 'tilt_y':
            return tilt_y
        elif self.axis == 'tilt_z':
            return tilt_z
        elif self.axis == 'ax':
            return data['accel']['x']
        elif self.axis == 'ay':
            return data['accel']['y']
        elif self.axis == 'az':
            return data['accel']['z']
        elif self.axis == 'gx':
            return data['gyro']['x']
        elif self.axis == 'gy':
            return data['gyro']['y']
        elif self.axis == 'gz':
            return data['gyro']['z']
        elif self.axis == 'temp':
            return data['temp']
        elif self.axis == 'all':
            return {
                'pitch': pitch,
                'roll': roll,
                'tilt_x': tilt_x,
                'tilt_y': tilt_y,
                'tilt_z': tilt_z,
                'accel': data['accel'],
                'gyro': data['gyro'],
                'temp': data['temp']
            }
        else:
            raise ValueError("Invalid axis specified. Choose from 'pitch', 'roll', 'tilt_x', 'tilt_y', 'tilt_z', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'temp', or 'all'.")
