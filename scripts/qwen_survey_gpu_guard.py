#!/usr/bin/env python3
"""Reserve actual CUDA/MIG UUIDs and check free memory without CUDA contexts."""

import argparse
from contextlib import ExitStack, contextmanager
import ctypes
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid


def driver_uuids():
    driver = ctypes.CDLL("libcuda.so.1")
    if driver.cuInit(0) != 0:
        raise RuntimeError("CUDA driver initialization failed")
    count = ctypes.c_int()
    if driver.cuDeviceGetCount(ctypes.byref(count)) != 0:
        raise RuntimeError("Cannot enumerate selected CUDA devices")

    class CUuuid(ctypes.Structure):
        _fields_ = [("data", ctypes.c_ubyte * 16)]

    result = []
    get_uuid = getattr(driver, "cuDeviceGetUuid_v2", driver.cuDeviceGetUuid)
    for index in range(count.value):
        device = ctypes.c_int()
        raw = CUuuid()
        if (
            driver.cuDeviceGet(ctypes.byref(device), index) != 0
            or get_uuid(ctypes.byref(raw), device) != 0
        ):
            raise RuntimeError("Cannot resolve selected CUDA UUID")
        result.append(str(uuid.UUID(bytes=bytes(raw.data))))
    return result


def resolve_devices(selection):
    # A fresh process is necessary: CUDA visibility is captured at driver init.
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=selection)
    output = subprocess.check_output(
        [sys.executable, str(Path(__file__).resolve()), "--resolve"], env=env, text=True
    )
    result = json.loads(output)
    if len(result) != 2 or len(set(result)) != 2:
        raise RuntimeError(
            f"Select two distinct accessible CUDA/MIG devices; {selection!r} resolves to {result}"
        )
    return result


def memory_info(device_uuid):
    lib = ctypes.CDLL("libnvidia-ml.so.1")
    if lib.nvmlInit_v2() != 0:
        raise RuntimeError("NVML initialization failed")

    class Memory(ctypes.Structure):
        _fields_ = [
            ("total", ctypes.c_ulonglong),
            ("free", ctypes.c_ulonglong),
            ("used", ctypes.c_ulonglong),
        ]

    try:
        handle = ctypes.c_void_p()
        for prefix in ("MIG-", "GPU-"):
            if (
                lib.nvmlDeviceGetHandleByUUID(
                    (prefix + device_uuid).encode(), ctypes.byref(handle)
                )
                == 0
            ):
                break
        else:
            raise RuntimeError(f"Cannot query NVML handle for {device_uuid}")
        memory = Memory()
        status = lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory))
        if status != 0:
            raise RuntimeError(
                f"NVML memory query failed for {device_uuid}: status {status}"
            )
        return {key: getattr(memory, key) / 2**30 for key in ("total", "free", "used")}
    finally:
        lib.nvmlShutdown()


@contextmanager
def reserve_devices(devices, lock_root, min_free_gib=8.0):
    if len(devices) != 2 or len(set(devices)) != 2:
        raise RuntimeError("Two distinct device UUIDs are required")
    lock_root = Path(lock_root)
    lock_root.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for device in sorted(devices):
            handle = stack.enter_context((lock_root / f"{device}.lock").open("a+"))
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.seek(0)
                owner = handle.read().strip()
                raise RuntimeError(
                    f"GPU/MIG slice {device} is already reserved ({owner}). Wait or select a different pair."
                ) from None
            handle.seek(0)
            handle.truncate()
            handle.write(json.dumps({"guard_pid": os.getpid(), "devices": devices}))
            handle.flush()
        memory = {device: memory_info(device) for device in devices}
        for device, info in memory.items():
            if info["free"] < min_free_gib:
                raise RuntimeError(
                    f'GPU/MIG slice {device} has {info["used"]:.2f} GiB used and {info["free"]:.2f} GiB free; this run requires at least {min_free_gib:g} GiB free. Wait or select an unused pair.'
                )
        yield memory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resolve", action="store_true")
    p.add_argument("--devices")
    p.add_argument(
        "--lock-root",
        type=Path,
        default=Path("/scratch/.tmp-gsm/aion_output/gpu_locks") / str(os.getuid()),
    )
    p.add_argument("--min-free-gib", type=float, default=8.0)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if args.resolve:
        print(json.dumps(driver_uuids()))
        return 0
    if not args.devices or args.min_free_gib < 0:
        p.error("--devices and a nonnegative --min-free-gib are required")
    devices = resolve_devices(args.devices)
    with reserve_devices(devices, args.lock_root, args.min_free_gib) as memory:
        print(
            "Reserved CUDA/MIG slices: "
            + ", ".join(
                f'{device} ({memory[device]["free"]:.2f} GiB free)'
                for device in devices
            ),
            flush=True,
        )
        if args.check_only:
            return 0
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            p.error("A launcher command is required")
        child = subprocess.Popen(command, env=dict(os.environ, SURVEY_GPU_GUARDED="1"))

        def forward(signum, _frame):
            if child.poll() is None:
                child.send_signal(signum)

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, forward)
        return child.wait()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.SubprocessError) as error:
        print(f"GPU assignment rejected: {error}", file=sys.stderr)
        raise SystemExit(2)
