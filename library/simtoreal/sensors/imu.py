from machine import Pin, I2C
import utime
import math
from mpu6050 import init_mpu6050, get_mpu6050_data

class MPU6050:
    def __init__(self,axis='roll',scl=21,sda=22,freq=400000):
        self.axis = axis
        self.i2c = I2C(1, scl=Pin(scl), sda=Pin(sda), freq=freq)
        init_mpu6050(self.i2c)
        self.pitch = 0
        self.roll = 0
        self.prev_time = utime.ticks_ms()
 
    def calculate_tilt_angles(self, accel_data):
        x, y, z = accel_data['x'], accel_data['y'], accel_data['z']
    
        tilt_x = math.atan2(y, math.sqrt(x * x + z * z)) * 180 / math.pi
        tilt_y = math.atan2(-x, math.sqrt(y * y + z * z)) * 180 / math.pi
        tilt_z = math.atan2(z, math.sqrt(x * x + y * y)) * 180 / math.pi
    
        return tilt_x, tilt_y, tilt_z
    
    def complementary_filter(self, pitch, roll, gyro_data, dt, alpha=0.98):
        pitch += gyro_data['x'] * dt
        roll -= gyro_data['y'] * dt
    
        pitch = alpha * pitch + (1 - alpha) * math.atan2(gyro_data['y'], math.sqrt(gyro_data['x'] * gyro_data['x'] + gyro_data['z'] * gyro_data['z'])) * 180 / math.pi
        roll = alpha * roll + (1 - alpha) * math.atan2(-gyro_data['x'], math.sqrt(gyro_data['y'] * gyro_data['y'] + gyro_data['z'] * gyro_data['z'])) * 180 / math.pi
    
        return pitch, roll
    

    
    def read(self):
        data = get_mpu6050_data(self.i2c)
        curr_time = utime.ticks_ms()
        dt = (curr_time - self.prev_time) / 1000
    
        tilt_x, tilt_y, tilt_z = self.calculate_tilt_angles(data['accel'])
        pitch, roll = self.complementary_filter(self.pitch, self.roll, data['gyro'], dt)
    
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