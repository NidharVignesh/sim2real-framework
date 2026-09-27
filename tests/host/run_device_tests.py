"""Upload the test files to the ESP32, run the board tests, and save their results on the PC.

Uses mpremote (the official MicroPython command-line tool; `pip install mpremote`).
Close Thonny first - only one program can use the USB serial port at a time.

Run from the repo root:
    python tests/host/run_device_tests.py                    # all tests, /dev/ttyUSB0
    python tests/host/run_device_tests.py --tests d1         # just the policy test
    python tests/host/run_device_tests.py --port COM5        # Windows port name
"""

import argparse
import datetime
import json
import subprocess
import sys

import common
import make_device_files

TESTS = {
    "d1": "test_d1_policy.py",
    "d2": "test_d2_sensor.py",
    "d3": "test_d3_control_loop.py",
    "d4": "test_d4_generated_main.py",
}
BOARD_DIR = "/sim2real_tests"


def mpremote(port, *args, timeout=120):
    cmd = [sys.executable, "-m", "mpremote", "connect", port, *args]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def board_sizes(port) -> dict:
    proc = mpremote(port, "fs", "ls", f":{BOARD_DIR}")
    sizes = {}
    for line in proc.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            sizes[parts[1]] = int(parts[0])
    return sizes


def upload(port, attempts=3):
    files = sorted(make_device_files.UPLOAD.iterdir())
    mpremote(port, "fs", "mkdir", f":{BOARD_DIR}")  # fails harmlessly if the folder already exists
    for f in files:
        # Serial transfers of large files occasionally glitch, so copy one file at a time and
        # confirm the size on the board before moving on.
        for attempt in range(1, attempts + 1):
            mpremote(port, "fs", "cp", str(f), f":{BOARD_DIR}/{f.name}", timeout=300)
            if board_sizes(port).get(f.name) == f.stat().st_size:
                print(f"  uploaded {f.name} ({f.stat().st_size} bytes)")
                break
            print(f"  {f.name}: size mismatch on board, retrying ({attempt}/{attempts})")
        else:
            raise RuntimeError(f"Could not upload {f.name} after {attempts} attempts")
    print(f"Uploaded {len(files)} files to {BOARD_DIR} on the board")


def run_test(port, key):
    script = make_device_files.UPLOAD / TESTS[key]
    print(f"\n===== Running {TESTS[key]} on the board =====")
    cmd = [sys.executable, "-m", "mpremote", "connect", port, "run", str(script)]
    # errors="replace": a board reset (e.g. brownout from the servos) emits non-UTF-8 bytes.
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    result = None
    for line in proc.stdout:
        line = line.rstrip()
        if line.startswith("RESULT_JSON:"):
            result = json.loads(line[len("RESULT_JSON:"):])
        else:
            print("  board> " + line)
    proc.wait()
    if result is None:
        print(f"  !! {TESTS[key]} did not report a result (exit code {proc.returncode})")
        return None
    result["pc_timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    result["git_commit"] = common.git_commit()
    common.RESULTS.mkdir(parents=True, exist_ok=True)
    out = common.RESULTS / f"{result['test']}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"  Saved results -> {out.relative_to(common.REPO)}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--tests", nargs="+", choices=list(TESTS), default=list(TESTS))
    parser.add_argument("--skip-build", action="store_true", help="Reuse tests/build/device_upload as is")
    parser.add_argument("--skip-upload", action="store_true", help="Files are already on the board")
    args = parser.parse_args()

    if not args.skip_build:
        make_device_files.main()
    if not args.skip_upload:
        upload(args.port)
    for key in args.tests:
        run_test(args.port, key)


if __name__ == "__main__":
    main()
