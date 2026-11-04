from __future__ import annotations
from .i18n import msg

import math
import os
import queue
import threading
import time
from ctypes import byref
from dataclasses import asdict, dataclass
from typing import Callable

from .config import SafetyConfig


class SafetyTrip(RuntimeError):
    """A latched stop; never automatically retry or resume training."""


@dataclass(frozen=True)
class GpuSample:
    monotonic: float
    uuid: str
    name: str
    temperature_c: float
    memory_temperature_c: float | None
    fan_percent: tuple[float, ...]
    fan_rpm: tuple[float, ...] | None
    power_w: float
    power_limit_w: float
    memory_used: int
    memory_total: int
    utilization: float
    clocks_reasons: int
    hardware_fault_mask: int
    errors: dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)


class SafetyPolicy:
    def __init__(self, config: SafetyConfig):
        self.config = config
        self.uuid: str | None = None
        self.first_errors: dict[str, int] | None = None
        self.latched: str | None = None

    def trip(self, message: str):
        self.latched = self.latched or message
        raise SafetyTrip(self.latched)

    def check(self, s: GpuSample, now: float | None = None) -> None:
        if self.latched:
            raise SafetyTrip(self.latched)
        c = self.config
        now = time.monotonic() if now is None else now
        numbers = (s.monotonic, s.temperature_c, s.power_w, s.power_limit_w,
                   s.memory_used, s.memory_total, s.utilization, *s.fan_percent)
        if any(not math.isfinite(n) for n in numbers):
            self.trip(msg('backend.gpu_telemetry_returned_an_invalid_value'))
        if not -0.05 <= now - s.monotonic <= c.telemetry_timeout_seconds:
            self.trip(msg('backend.gpu_telemetry_is_stale_or_its_timestamp'))
        if self.uuid is not None and self.uuid != s.uuid:
            self.trip(msg('backend.the_monitored_gpu_identity_changed'))
        if not s.uuid or not s.name:
            self.trip(msg('backend.could_not_verify_gpu_identity'))
        if not (0 < s.temperature_c < 120 and 0 <= s.utilization <= 100):
            self.trip(msg('backend.invalid_gpu_temperature_or_utilization'))
        if not s.fan_percent or any(not 0 <= x <= 200 for x in s.fan_percent):
            self.trip(msg('backend.fan_telemetry_is_unavailable_or_invalid'))
        if not (0 <= s.memory_used <= s.memory_total and s.memory_total > 0):
            self.trip(msg('backend.invalid_vram_reading'))
        if not 0 <= s.power_w or s.power_limit_w <= 0:
            self.trip(msg('backend.invalid_power_reading'))
        if s.temperature_c >= c.temperature_limit_c:
            self.trip(msg('backend.gpu_core_temperature_reached_the_stop_threshold', f'{s.temperature_c:.0f}', f'{c.temperature_limit_c:.0f}'))
        if s.memory_temperature_c is not None:
            if not math.isfinite(s.memory_temperature_c) or not 0 < s.memory_temperature_c < 140:
                self.trip(msg('backend.invalid_vram_temperature'))
            if s.memory_temperature_c >= c.memory_temperature_limit_c:
                self.trip(msg('backend.vram_temperature_reached_the_stop_threshold_0', f'{s.memory_temperature_c:.0f}'))
        if s.memory_used / s.memory_total >= c.vram_fraction_limit:
            self.trip(msg('backend.vram_usage_reached_the_0_stop_threshold', f'{c.vram_fraction_limit:.0%}'))
        if s.power_w > s.power_limit_w * c.power_limit_ratio:
            self.trip(msg('backend.abnormal_power_draw_0_w_1_w', f'{s.power_w:.1f}', f'{s.power_limit_w:.1f}'))
        if s.clocks_reasons & s.hardware_fault_mask:
            self.trip(msg('backend.hardware_throttling_thermal_protection_or_power_braking', f'{s.clocks_reasons:x}'))
        if s.temperature_c >= c.fan_required_at_c and any(x == 0 for x in s.fan_percent):
            self.trip(msg('backend.a_fan_still_has_a_zero_target', f'{s.temperature_c:.0f}'))
        if s.fan_rpm is not None:
            if len(s.fan_rpm) != len(s.fan_percent):
                self.trip(msg('backend.the_number_of_fan_sensors_changed'))
            if any(not math.isfinite(x) or x < 0 for x in s.fan_rpm):
                self.trip(msg('backend.invalid_fan_rpm_reading'))
            if any(p > 0 and rpm == 0 for p, rpm in zip(s.fan_percent, s.fan_rpm)):
                self.trip(msg('backend.a_fan_reports_zero_rpm_despite_a'))
        if self.first_errors is not None and set(s.errors) != set(self.first_errors):
            self.trip(msg('backend.hardware_error_counter_telemetry_was_lost'))
        for name, count in s.errors.items():
            if count < 0 or count > 0:
                self.trip(msg('backend.gpu_hardware_error_or_pending_repair_0', name, count))
        self.uuid = s.uuid
        self.first_errors = dict(s.errors)


class NvmlReader:
    """Only NVML getters. Unsupported optional sensors are explicitly advertised.

    A sensor that worked at startup may never silently turn into 'unsupported'.
    Fan percentages are requested speeds, not proof of physical rotation.
    """
    def __init__(self, gpu_index: int = 0):
        import pynvml as nv
        self.nv = nv
        nv.nvmlInit()
        self.handle = nv.nvmlDeviceGetHandleByIndex(gpu_index)
        self.name = self._text(nv.nvmlDeviceGetName(self.handle))
        self.uuid = self._text(nv.nvmlDeviceGetUUID(self.handle))
        self.driver = self._text(nv.nvmlSystemGetDriverVersion())
        self.capabilities: dict[str, bool] = {}
        self.fan_count = self._optional("fan_count", lambda: nv.nvmlDeviceGetNumFans(self.handle))
        self.fan_count = int(self.fan_count or 1)
        self.driver_events = None
        if os.name == "nt":
            from .driver_events import DriverEvents
            self.driver_events = DriverEvents()
        self.fault_mask = sum(getattr(nv, name, default) for name, default in (
            ("nvmlClocksEventReasonHwSlowdown", 0x8),
            ("nvmlClocksEventReasonSwThermalSlowdown", 0x20),
            ("nvmlClocksEventReasonHwThermalSlowdown", 0x40),
            ("nvmlClocksEventReasonHwPowerBrakeSlowdown", 0x80),
        ))

    @staticmethod
    def _text(x) -> str:
        return x.decode() if isinstance(x, bytes) else str(x)

    def _optional(self, name: str, fn: Callable):
        if self.capabilities.get(name) is False:
            return None
        nv = self.nv
        try:
            result = fn()
        except (nv.NVMLError_NotSupported, nv.NVMLError_FunctionNotFound, AttributeError):
            if self.capabilities.get(name):
                raise SafetyTrip(msg('backend.a_previously_available_sensor_stopped_reporting_0', name))
            self.capabilities[name] = False
            return None
        self.capabilities[name] = True
        return result

    def _memory_temp(self):
        nv = self.nv
        field = nv.nvmlDeviceGetFieldValues(self.handle, [nv.NVML_FI_DEV_MEMORY_TEMP])[0]
        if field.nvmlReturn:
            raise nv.NVMLError(field.nvmlReturn)
        return float(field.value.uiVal)

    def _fan_rpm(self):
        nv = self.nv
        result = []
        for index in range(self.fan_count):
            # The public Python wrapper hardcodes fan 0. Use the same documented
            # NVML getter/structure to read each fan, never a control/set API.
            reading = nv.c_nvmlFanSpeedInfo_t()
            reading.version = nv.nvmlFanSpeedInfo_v1
            reading.fan = index
            fn = nv._nvmlGetFunctionPointer("nvmlDeviceGetFanSpeedRPM")
            nv._nvmlCheckReturn(fn(self.handle, byref(reading)))
            result.append(float(reading.speed))
        return tuple(result)

    def sample(self) -> GpuSample:
        nv, h = self.nv, self.handle
        start = time.monotonic()
        uuid = self._text(nv.nvmlDeviceGetUUID(h))
        temp = float(nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU))
        mem = nv.nvmlDeviceGetMemoryInfo(h)
        fan = tuple(float(nv.nvmlDeviceGetFanSpeed_v2(h, i)) for i in range(self.fan_count))
        power = nv.nvmlDeviceGetPowerUsage(h) / 1000
        limit = nv.nvmlDeviceGetEnforcedPowerLimit(h) / 1000
        utilization = float(nv.nvmlDeviceGetUtilizationRates(h).gpu)
        reasons = int(nv.nvmlDeviceGetCurrentClocksEventReasons(h))
        memory_temp = self._optional("memory_temperature", self._memory_temp)
        rpm = self._optional("fan_rpm", self._fan_rpm)
        errors = {}
        remapped = self._optional("remapped_rows", lambda: nv.nvmlDeviceGetRemappedRows(h))
        if remapped is not None:
            corrected, uncorrected, pending, failure = remapped
            errors.update(remapped_corrected=int(corrected), remapped_uncorrected=int(uncorrected),
                          remap_pending=int(pending), remap_failure=int(failure))
        for label, error_type in (("ecc_corrected", nv.NVML_MEMORY_ERROR_TYPE_CORRECTED),
                                  ("ecc_uncorrected", nv.NVML_MEMORY_ERROR_TYPE_UNCORRECTED)):
            value = self._optional(label, lambda t=error_type: nv.nvmlDeviceGetTotalEccErrors(
                h, t, nv.NVML_VOLATILE_ECC))
            if value is not None:
                errors[label] = int(value)
        retired = self._optional("retired_pages_pending", lambda: nv.nvmlDeviceGetRetiredPagesPendingStatus(h))
        if retired is not None:
            errors["retired_pages_pending"] = int(retired)
        if self.driver_events is not None:
            faults = self.driver_events.poll()
            errors["windows_driver_errors"] = len(faults)
        return GpuSample(start, uuid, self.name, temp, memory_temp, fan, rpm, power,
                         limit, int(mem.used), int(mem.total), utilization, reasons,
                         self.fault_mask, errors)

    def close(self):
        if self.driver_events:
            self.driver_events.close()
        self.nv.nvmlShutdown()


class AsyncTelemetry:
    """Keep native-driver calls off the watchdog loop, with a strict deadline."""
    def __init__(self, factory: Callable[[], NvmlReader]):
        self.events: queue.Queue = queue.Queue(maxsize=2)
        self.request: queue.Queue = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=self._run, args=(factory,), daemon=True, name="nvml-reader")
        self.thread.start()
        self.pending_since: float | None = time.monotonic()

    def _put(self, item):
        self.events.put(item)

    def _run(self, factory):
        reader = None
        try:
            reader = factory()
            self._put(("sample", reader.sample(), {"driver": reader.driver, "capabilities": reader.capabilities.copy()}))
            while self.request.get():
                self._put(("sample", reader.sample(), {"driver": reader.driver, "capabilities": reader.capabilities.copy()}))
        except Exception as exc:
            self._put(("error", f"{type(exc).__name__}: {exc}", {}))
        finally:
            if reader:
                try:
                    reader.close()
                except Exception:
                    pass

    def poll(self, timeout_seconds: float):
        try:
            kind, value, metadata = self.events.get_nowait()
        except queue.Empty:
            if self.pending_since is not None and time.monotonic() - self.pending_since > timeout_seconds:
                raise SafetyTrip(msg('backend.nvml_polling_timed_out_gpu_health_could'))
            return None
        self.pending_since = None
        if kind == "error":
            raise SafetyTrip(msg('backend.gpu_telemetry_read_failed_0', value))
        return value, metadata

    def next(self):
        if self.pending_since is None:
            self.pending_since = time.monotonic()
            self.request.put_nowait(True)

    def close(self):
        try:
            self.request.put_nowait(False)
        except queue.Full:
            pass
