# MicroPython helper: save a test's results as JSON on the board AND print them for the PC.
import json
import os
import time

RESULTS_DIR = "/sim2real_results"


def stats(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    ordered = sorted(values)
    return {
        "n": n,
        "mean": mean,
        "std": var ** 0.5,
        "min": ordered[0],
        "max": ordered[-1],
        "p99": ordered[min(n - 1, int(0.99 * n))],
    }


def save(name, data):
    data["test"] = name
    data["firmware"] = os.uname().version
    data["board"] = os.uname().machine
    data["board_uptime_ms"] = time.ticks_ms()
    try:
        os.mkdir(RESULTS_DIR)
    except OSError:
        pass
    text = json.dumps(data)
    with open(RESULTS_DIR + "/" + name + ".json", "w") as f:
        f.write(text)
    print("Saved on board: " + RESULTS_DIR + "/" + name + ".json")
    # The PC runner (tests/host/run_device_tests.py) looks for this exact prefix.
    print("RESULT_JSON:" + text)
