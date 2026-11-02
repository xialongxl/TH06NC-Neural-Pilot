"""Read-only capture regressions; a changing tick must never trigger STOP."""
import struct

import numpy as np
import pytest

from th06ai import native


class UpdatingGameMemory:
    base = 0x140000000

    def __init__(self, advancing=True):
        self.tick = 100
        self.advancing = advancing
        self.world = bytearray(native.WORLD_SIZE)
        self.actors = bytearray(native.ACTORS_SIZE)
        self.bulk_reads = self.header_reads = 0
        self.fail = False
        self.put(native.BULLET_NODE+8, "Q", self.base+native.BULLET_UPDATE)
        self.put(native.BULLET_NODE+0x38, "Q", self.base+native.BULLET_MANAGER)
        self.put(native.PLAYER+0x7730, "ff", 192, 384)
        self.put(native.PLAYER+0x774C, "f", 2)
        self.put(native.BOUNDS, "ffff", 8, 16, 368, 416)
        self.put(native.POWER, "H", 16)
        self.put(native.LIVES_BOMBS, "bb", 2, 3)

    def put(self, address, fmt, *values):
        struct.pack_into("<"+fmt, self.world, address-native.WORLD_START, *values)

    def read(self, address, size):
        if address == native.INPUT_WORD and size == 4:
            return struct.pack('<I', 0x84)
        assert address == native.BULLET_MANAGER and size == 8
        self.header_reads += 1
        value = struct.pack("<ii", self.tick-1, self.tick)
        self.tick += int(self.advancing)
        return value

    def read_into(self, address, target):
        self.bulk_reads += 1
        if self.fail:
            raise RuntimeError("simulated ReadProcessMemory failure")
        if address == native.WORLD_START:
            self.put(native.BULLET_MANAGER, "ii", self.tick-1, self.tick)
            target[:] = self.world
        else:
            assert address == native.ACTORS_START
            target[:] = self.actors
        self.tick += int(self.advancing)


def reader_for(memory):
    reader = native.GameReader.__new__(native.GameReader)
    reader.process = memory
    reader._dtype = native.BULLET_DTYPE
    return reader


def test_empty_opening_with_every_capture_crossing_ticks_never_retries_or_stops():
    memory = UpdatingGameMemory()
    reader = reader_for(memory)
    for _ in range(100):
        frame = reader.read()
        assert frame.stage_active and len(frame.bullets) == 0
        assert reader.snapshot_diagnostics["crossed_tick"] is True
        assert reader.snapshot_diagnostics["attempts"] == 1
        assert frame.tick == reader.snapshot_diagnostics["sample_tick"]
        assert frame.resources.power == 16
    assert memory.bulk_reads == memory.header_reads == 200


def test_visible_bullets_are_returned_even_when_capture_crosses_a_tick():
    memory = UpdatingGameMemory()
    memory.put(native.BULLET_MANAGER+8+0x30, "ff", 204, 380)
    memory.put(native.BULLET_MANAGER+8+0x44, "H", 1)
    memory.put(native.BULLET_MANAGER+8+0x5F4, "ff", 4, 8)
    reader = reader_for(memory)
    old = reader.read()
    buffers = id(reader._world), id(reader._actors)
    memory.put(native.BULLET_MANAGER+8+0x30, "ff", 220, 384)
    new = reader.read()
    assert new.bullets[0, 1] == 220 and old.bullets[0, 1] == 204
    np.testing.assert_array_equal(new.bullet_sizes[0], [4, 8])
    assert buffers == (id(reader._world), id(reader._actors))


def test_paused_snapshot_and_live_snapshot_use_the_same_decoder():
    memory = UpdatingGameMemory(advancing=False)
    memory.put(native.PAUSED, "B", 1)
    reader = reader_for(memory)
    frame = reader.read()
    assert frame.paused and frame.stage_active and frame.tick == 100
    assert reader.snapshot_diagnostics["crossed_tick"] is False


def test_actual_read_failure_is_not_replaced_with_a_cached_observation():
    memory = UpdatingGameMemory()
    reader = reader_for(memory)
    reader.read()
    memory.fail = True
    with pytest.raises(RuntimeError, match="ReadProcessMemory failure"):
        reader.read()


def test_hardware_guard_checkpoint_still_interrupts_before_reading():
    memory = UpdatingGameMemory()
    reader = reader_for(memory)
    def stopped():
        raise RuntimeError("hardware guard stopped")
    reader.checkpoint = stopped
    with pytest.raises(RuntimeError, match="hardware guard stopped"):
        reader.read()
    assert memory.bulk_reads == memory.header_reads == 0


def test_invalid_native_structure_still_fails_instead_of_training_on_garbage():
    memory = UpdatingGameMemory()
    memory.put(native.BULLET_MANAGER+8+0x44, "H", 99)
    with pytest.raises(RuntimeError, match="弹幕池结构"):
        reader_for(memory).read()


def test_reusable_read_buffer_boundary_copies_kernel_data_and_rejects_out_of_module(monkeypatch):
    import ctypes as ct
    from th06ai import win32
    process = win32.ReadOnlyProcess.__new__(win32.ReadOnlyProcess)
    process.base, process.size, process.handle = 0x1000, 100, 1
    calls = []
    def read_memory(handle, address, target, size, count):
        calls.append((address, size))
        ct.memmove(target, b"abcd", size)
        count._obj.value = size
        return True
    monkeypatch.setattr(win32, "read_memory", read_memory)
    buffer = bytearray(4)
    process.read_into(12, buffer)
    assert buffer == b"abcd" and calls == [(0x100c, 4)]
    assert process.read(12, 4) == b"abcd"
    with pytest.raises(ValueError, match="主模块"):
        process.read_into(99, buffer)
    assert len(calls) == 2
