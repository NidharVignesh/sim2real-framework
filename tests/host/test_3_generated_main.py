"""Test 3 - Does the main.py that simtoreal generates run the policy the way it was trained?

Generates main.py from library/simtoreal/config.yaml with the library, then runs it on the PC
against fake ESP32 hardware (machine.Pin / I2C / PWM) and checks, one by one:
  A. pin numbers are passed as integers  (the real ESP32 rejects Pin('22') with ValueError)
  B. observe() returns as many values as the policy has inputs
  C. roll/pitch are in radians, like the simulator
  D. the network output is turned into servo angles like in training
     (target += 0.08 * action, clipped to +-1.55 rad, then servo angle = 90 + degrees(target))
  E. the loop runs at a fixed 50 Hz, like training
  F. the generator also works for a different, made-up robot (it is a general library)

Run from the repo root:  python tests/host/test_3_generated_main.py
"""

import argparse
import importlib
import math
import re
import shutil
import sys
import types

import common

WORK = common.BUILD / "generated_main"
CONFIG = common.REPO / "library" / "simtoreal" / "config.yaml"
TRUE_ROLL_DEG = 30.0  # the fake IMU is tilted 30 deg about X
SERVO_RESOLUTION_DEG = 180 / (123 - 26)  # servo.py uses 10-bit duty 26..123


class FakeClock:
    def __init__(self):
        self.ms = 0

    def ticks_ms(self):
        self.ms += 10
        return self.ms

    def sleep_ms(self, _):
        pass


def install_fake_hardware(strict_pins: bool, servo_log: list):
    clock = FakeClock()
    machine = types.ModuleType("machine")

    class Pin:
        def __init__(self, pin, *args, **kwargs):
            if strict_pins and not isinstance(pin, int):
                raise ValueError("invalid pin")  # what MicroPython 1.28 on the ESP32 really does
            self.pin = int(pin)

    class I2C:
        def __init__(self, *args, **kwargs):
            pass

        def writeto(self, addr, data):
            pass

        def readfrom_mem(self, addr, reg, n):
            ay = math.sin(math.radians(TRUE_ROLL_DEG))
            az = math.cos(math.radians(TRUE_ROLL_DEG))
            words = [0, int(ay * 16384), int(az * 16384), 0, 0, 0, 0]
            out = bytearray()
            for w in words:
                w &= 0xFFFF
                out += bytes([w >> 8, w & 0xFF])
            return bytes(out)

    class PWM:
        def __init__(self, pin, *args, **kwargs):
            self.pin = pin.pin

        def freq(self, f=None):
            pass

        def duty(self, d):
            servo_log.append((self.pin, d))

    machine.Pin, machine.I2C, machine.PWM = Pin, I2C, PWM
    utime = types.ModuleType("utime")
    utime.ticks_ms, utime.sleep_ms = clock.ticks_ms, clock.sleep_ms
    sys.modules["machine"], sys.modules["utime"] = machine, utime


def import_generated_main():
    for name in ("main", "imu", "servo", "policy_network"):
        sys.modules.pop(name, None)
    return importlib.import_module("main")


def duty_to_angle(duty):
    # inverse of servo.py's mapping (u10 duty 26..123 <-> 0..180 deg)
    return (duty - 26) * 180 / (123 - 26)


def main():
    from simtoreal.interface import generate_interface

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG), help="YAML config to generate main.py from")
    parser.add_argument("--result-name", default="host_test3_generated_main")
    args = parser.parse_args()
    config = common.Path(args.config).resolve()

    artifacts = common.build_artifacts()
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    generate_interface(str(config), str(WORK / "main.py"))
    shutil.copy(artifacts["py"], WORK / "policy_network.py")
    shutil.copy(common.REPO / "library/simtoreal/sensors/imu.py", WORK / "imu.py")
    shutil.copy(common.REPO / "library/simtoreal/actuators/servo.py", WORK / "servo.py")
    sys.path.insert(0, str(WORK))
    source = (WORK / "main.py").read_text()
    checks = {}

    # A. integer pins
    quoted = re.findall(r"(scl|sda|pin)='([^']*)'", source)
    servo_log = []
    install_fake_hardware(strict_pins=True, servo_log=servo_log)
    try:
        import_generated_main()
        import_error = None
    except Exception as e:  # noqa: BLE001 - we want to report whatever the board would hit
        import_error = f"{type(e).__name__}: {e}"
    checks["A_pins_are_integers"] = {
        "passed": not quoted and import_error is None,
        "quoted_pin_arguments": [f"{k}='{v}'" for k, v in quoted],
        "error_on_esp32_style_import": import_error,
        "note": "Verified on the real board: Pin('22') -> ValueError: invalid pin",
    }

    # Continue with lenient pins so the remaining problems can be found too.
    install_fake_hardware(strict_pins=False, servo_log=servo_log)
    gen = import_generated_main()
    policy_inputs = gen.policy_forward.__globals__["POLICY_INPUT_SIZE"]

    # B. observation length
    for _ in range(1000):  # let the complementary filter converge to the fake 30 deg tilt
        obs = gen.observe()
    checks["B_observation_length_matches_policy"] = {
        "passed": len(obs) == policy_inputs,
        "observe_returns": len(obs),
        "policy_expects": policy_inputs,
        "expected_order": ["roll_rad", "pitch_rad", "az_g", "target1_rad", "target2_rad"],
    }
    try:
        gen.policy()
        policy_error = None
    except Exception as e:  # noqa: BLE001
        policy_error = f"{type(e).__name__}: {e}"
    checks["B_observation_length_matches_policy"]["error_when_policy_runs"] = policy_error

    # C. units
    roll_value = obs[0]
    checks["C_roll_in_radians"] = {
        "passed": abs(roll_value - math.radians(TRUE_ROLL_DEG)) < 0.05,
        "true_roll_rad": math.radians(TRUE_ROLL_DEG),
        "observe_roll_value": roll_value,
        "looks_like_degrees": abs(roll_value - TRUE_ROLL_DEG) < 1.0,
    }

    # D. action handling: give the generated apply_actions() a real network output, starting from
    #    the initial set-points, and compare the servo angles with what the training environment does.
    full_obs = [math.radians(TRUE_ROLL_DEG), 0.0, math.cos(math.radians(TRUE_ROLL_DEG)), 0.0, 0.0]
    action = gen.policy_forward(full_obs)
    if hasattr(gen, "action_values"):
        gen.action_values[:] = [0.0] * len(gen.action_values)
    commanded = []
    actuators = sorted(n for n in vars(gen) if re.fullmatch(r"act\d+", n))
    for n in actuators:  # record exactly what main.py sends to each actuator's write()
        getattr(gen, n).write = lambda value: commanded.append(round(float(value), 4))
    gen.apply_actions(action)
    trained = [90 + math.degrees(max(-1.55, min(1.55, 0.08 * max(-1.0, min(1.0, a))))) for a in action]
    tolerance = 1e-3
    checks["D_actions_integrated_like_training"] = {
        "passed": len(commanded) == len(trained)
        and all(abs(c - e) <= tolerance for c, e in zip(commanded, trained)),
        "network_output": [float(a) for a in action],
        "servo_angle_commanded_by_generated_main_deg": commanded,
        "servo_angle_expected_after_first_step_deg": [round(e, 2) for e in trained],
        "tolerance_deg": tolerance,
        "note": "Values are what main.py passes to write(); servo.py then rounds to its 10-bit PWM duty "
                "(one step = %.2f deg)." % SERVO_RESOLUTION_DEG,
    }

    # E. fixed control rate, like training (50 Hz)
    period = getattr(gen, "PERIOD_MS", None)
    checks["E_fixed_control_rate"] = {
        "passed": period == 20 and "utime.sleep_ms(spare)" in source,
        "period_ms_in_generated_main": period,
        "expected_period_ms": 20,
    }

    # F. the generator is not specific to this robot: a made-up wheeled robot with an absolute-mode
    #    motor, a custom sensor and no control rate must still give valid code, and the custom
    #    placeholder must stop with a clear error instead of a silent wrong value.
    other = WORK / "other_robot.yaml"
    other.write_text(
        "observations:\n"
        "  - {name: wheel_speed, type: encoder, class: Encoder, pin_a: 4, pin_b: 5, scale: 0.01}\n"
        "  - {name: distance, type: custom, class: custom}\n"
        "actions:\n"
        "  - {name: motor, type: motor, class: Motor, pin: 12, min: -1.0, max: 1.0, scale: 100.0}\n"
    )
    other_main = WORK / "other_robot_main.py"
    generate_interface(str(other), str(other_main))
    other_src = other_main.read_text()
    try:
        compile(other_src, str(other_main), "exec")
        syntax_error = None
    except SyntaxError as e:
        syntax_error = str(e)
    namespace = {"__name__": "other_robot_main"}
    fake_encoder = types.ModuleType("encoder")
    fake_encoder.Encoder = lambda **kw: types.SimpleNamespace(read=lambda: 250.0)
    fake_motor = types.ModuleType("motor")
    motor_log = []
    fake_motor.Motor = lambda **kw: types.SimpleNamespace(write=motor_log.append)
    fake_policy = types.ModuleType("policy_network")
    fake_policy.policy_forward = lambda obs: [3.0]
    saved = {k: sys.modules.get(k) for k in ("encoder", "motor", "policy_network")}
    sys.modules.update({"encoder": fake_encoder, "motor": fake_motor, "policy_network": fake_policy})
    try:
        exec(other_src, namespace)
        try:
            namespace["observe"]()
            todo_error = None
        except NotImplementedError as e:
            todo_error = str(e)
        namespace["apply_actions"]([3.0])
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    checks["F_generic_other_robot"] = {
        "passed": syntax_error is None and todo_error is not None and motor_log == [100.0]
        and "utime" not in other_src,
        "syntax_error": syntax_error,
        "custom_placeholder_error": todo_error,
        "motor_command_for_action_3.0": motor_log,
        "expected_motor_command": [100.0],
        "note": "absolute mode: clip(3.0, -1, 1) * scale 100 = 100",
    }

    n_failed = sum(1 for c in checks.values() if c["passed"] is not True)
    print("\nChecks on the library-generated main.py:")
    for name, c in checks.items():
        print(f"  {'PASS' if c['passed'] else 'FAIL'}  {name}")
    print(f"\n{n_failed} of {len(checks)} checks failed")
    common.save_result(args.result_name, {
        "config": str(config.relative_to(common.REPO)),
        "config_yaml": config.read_text(),
        "generated_main_py": source,
        "checks": checks,
        "checks_failed": n_failed,
    })


if __name__ == "__main__":
    main()
