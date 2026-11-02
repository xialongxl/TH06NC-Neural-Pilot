from __future__ import annotations

import math
import time
from dataclasses import replace

import pytest

from th06ai.config import SafetyConfig
from th06ai.gpu import AsyncTelemetry, GpuSample, SafetyPolicy, SafetyTrip


def healthy(**kwargs):
    sample = GpuSample(time.monotonic(), "GPU-test", "test", 48, None, (0, 0), (0, 0),
                       15, 215, 2 * 2**30, 12 * 2**30, 20, 1, 0xE8,
                       {"remapped_corrected": 0, "windows_driver_errors": 0})
    return replace(sample, **kwargs)


def test_normal_zero_rpm_and_full_utilization_are_not_failures():
    policy = SafetyPolicy(SafetyConfig())
    policy.check(healthy(utilization=100, clocks_reasons=0x4, power_w=215))


@pytest.mark.parametrize("changes", [
    {"temperature_c": 70}, {"temperature_c": 60},
    {"temperature_c": 55, "fan_percent": (20, 20), "fan_rpm": (0, 1000)},
    {"memory_temperature_c": 80}, {"temperature_c": math.nan},
    {"memory_used": int(12 * 2**30 * .91)}, {"power_w": 223},
    {"clocks_reasons": 0x8}, {"clocks_reasons": 0x20}, {"clocks_reasons": 0x40},
    {"clocks_reasons": 0x80}, {"fan_percent": ()}, {"memory_total": 0},
    {"power_limit_w": 0}, {"monotonic": time.monotonic() - 61},
    {"errors": {"windows_driver_errors": 1}},
])
def test_any_detectable_fault_stops_on_first_sample(changes):
    with pytest.raises(SafetyTrip):
        SafetyPolicy(SafetyConfig()).check(healthy(**changes))


@pytest.mark.parametrize("age", [1.0, 10.0, 59.99, 60.0])
def test_monitor_accepts_delayed_data_until_sixty_seconds(age):
    SafetyPolicy(SafetyConfig()).check(healthy(monotonic=100.0), now=100.0 + age)


def test_trip_is_latched_after_temperature_recovers():
    policy = SafetyPolicy(SafetyConfig())
    with pytest.raises(SafetyTrip):
        policy.check(healthy(temperature_c=70))
    with pytest.raises(SafetyTrip):
        policy.check(healthy())


def test_gpu_identity_and_error_sensor_loss_stop():
    for changes in ({"uuid": "GPU-other"}, {"errors": {}}):
        policy = SafetyPolicy(SafetyConfig())
        policy.check(healthy())
        with pytest.raises(SafetyTrip):
            policy.check(healthy(**changes))


def test_monitor_errors_are_not_retried_silently():
    def broken():
        raise OSError("driver lost")
    stream = AsyncTelemetry(broken)
    stream.thread.join(timeout=1)
    with pytest.raises(SafetyTrip, match="driver lost"):
        stream.poll(.75)


def test_hung_native_query_trips_deadline():
    import threading
    done = threading.Event()
    def hanging():
        done.wait(1)
        raise RuntimeError("done")
    stream = AsyncTelemetry(hanging)
    try:
        timeout = SafetyConfig().telemetry_timeout_seconds
        stream.pending_since = time.monotonic() - 10
        assert stream.poll(timeout) is None
        stream.pending_since = time.monotonic() - 61
        with pytest.raises(SafetyTrip, match="超时"):
            stream.poll(timeout)
    finally:
        done.set()
        stream.close()


def test_cannot_raise_or_disable_protection_in_config():
    with pytest.raises(ValueError):
        SafetyConfig(temperature_limit_c=90)
    with pytest.raises(ValueError):
        SafetyConfig(poll_seconds=10)
