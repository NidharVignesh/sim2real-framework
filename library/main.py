import utime
from policy_network import policy_forward
from imu import mpu6050
from servo import servo

PERIOD_MS = 20  # control.rate_hz = 50

# observations
sensor=mpu6050(axis='all',scl=22,sda=21)
sensor.calibrate()
# obs2 = mpu6050(axis='az', scl=22, sda=21)

# actions
act0 = servo(pin=18)
act1 = servo(pin=19)

# current value of each action, in the policy's units
action_values = [0.0, 0.0]

def _clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def observe():
    reading = sensor.read()
    return [
        reading['pitch'] * 0.017453292519943295,
        reading['roll'] * 0.017453292519943295,
#         obs2.read(),
        action_values[0],
        action_values[1],
    ]

def apply_actions(actions):
    action_values[0] = _clip(action_values[0] + _clip(actions[0], -1.0, 1.0) * 0.08, -1.55, 1.55)
    act0.write(action_values[0] * 57.29577951308232 + 90.0)
    action_values[1] = _clip(action_values[1] + _clip(actions[1], -1.0, 1.0) * 0.08, -1.55, 1.55)
    act1.write(action_values[1] * 57.29577951308232 + 90.0)

def init_actuators():
    # send every actuator to its initial value before the loop starts
    act0.write(action_values[0] * 57.29577951308232 + 90.0)
    act1.write(action_values[1] * 57.29577951308232 + 90.0)

def policy():
    obs = observe()
    print(obs)
    actions = policy_forward(obs)
    apply_actions(actions)
    print(actions)

if __name__ == '__main__':
    init_actuators()
    while True:
        start = utime.ticks_ms()
        policy()
        spare = PERIOD_MS - utime.ticks_diff(utime.ticks_ms(), start)
        if spare > 0:
            utime.sleep_ms(spare)

