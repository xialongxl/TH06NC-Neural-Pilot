"""Optimization parity and concurrency; no game, CUDA or OS keyboard input."""
import ctypes as ct
from dataclasses import replace
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from th06ai import bullet_state, model_store, win32
from th06ai.acceleration import extension, RustKeyController
from th06ai.bullet_state import BulletStateCache
from th06ai.config import Config, DiagnosticsConfig
from th06ai.console_service import ConsoleService
from th06ai.diagnostics import SessionDiagnostics
from th06ai.files import atomic_json
from th06ai.input_timing import InputTiming
from th06ai.native import GameResources
from th06ai.observations import ObservationBuilder, action_keys, movement_action
from th06ai.rewards import RewardTracker, item_potential
from test_rewards import frame
from test_web_console import model


def test_menu_q_has_vk_scan_and_serial_native_bit_without_real_input(monkeypatch):
    packets = []
    def send(count, events, size):
        packets.extend((events[i].data.ki.wVk, events[i].data.ki.wScan, events[i].data.ki.dwFlags) for i in range(count))
        return count
    monkeypatch.setattr(win32, 'send_input', send)
    keys = win32.KeyController(SimpleNamespace(focused=True))
    keys.set({'q'}); keys.release()
    assert packets == [(0x51,0x10,0),(0x51,0x10,2)]
    if extension() is None:pytest.skip('optional ABI v3 extension not built')
    native = RustKeyController(SimpleNamespace(pid=0,hwnd=0,focused=True),dry_run=True)
    assert native.set({'q'})['events'] == 1 and native._native.held == 1<<9
    assert native.set({'r'})['events'] == 2 and native._native.held == 1<<8
    native.release()


def test_return_to_title_uses_real_q_adapter_with_dry_run_os_boundary(monkeypatch):
    if extension() is None:pytest.skip('optional ABI v3 extension not built')
    from th06ai import environment
    env=environment.TouhouEnv.__new__(environment.TouhouEnv)
    env.keys=RustKeyController(SimpleNamespace(pid=0,hwnd=0,focused=True),dry_run=True)
    screens=iter([replace(frame(),phase='paused',paused=True,menu_ready=True),replace(frame(),phase='title')])
    env.reader=SimpleNamespace(read=lambda:next(screens),process=SimpleNamespace(read=lambda *_:b'\0'*4))
    env.pause=lambda:None;env.lease=lambda:None;env._ensure_transaction_focus=lambda:None
    env._wait=lambda _:None;env.status=lambda **_:None
    monkeypatch.setattr(environment,'get_async_key',lambda _:0)
    env._return_to_title()
    assert not env.keys.held and env.keys._native.held==0


def test_resource_reachability_uses_verified_self_speed_and_unknown_keeps_baseline():
    current = replace(frame(), resources=GameResources(0,16,3,2,0),
        items=np.array([[0,300,420,0,0,1]],np.float32), movement_speeds=(4.,2.))
    baseline = item_potential(current,0,0)
    assert baseline > 0
    assert item_potential(replace(current,movement_speeds=(5.,2.5)),0,0) == pytest.approx(baseline*1.25)
    for speed in (None,(float('nan'),2.),(0.,0.)):
        assert item_potential(replace(current,movement_speeds=speed),0,0) == baseline


def test_shared_bullet_state_preserves_history_and_does_not_reuse_same_tick_new_arrays(monkeypatch):
    old = frame(tick=100,bullets=[(1,200,300)])
    current = frame(tick=102,bullets=[(1,200,304)])
    calls = []
    compute = bullet_state.visible_bullet_state
    def counted(*args):
        calls.append(args)
        return compute(*args)
    monkeypatch.setattr(bullet_state,'visible_bullet_state',counted)
    cache = BulletStateCache()
    builder = ObservationBuilder(feature_cache=cache)
    tracker = RewardTracker(feature_cache=cache);tracker.reset(old)
    builder.previous = replace(old)
    tracker.step(current,0,False);obs = builder.build(current)
    assert len(calls) == 1
    first = cache.get(current,replace(old))
    changed = replace(current,bullets=np.array([[1,200,308]],np.float32))
    second = cache.get(changed,old)
    assert second is not first and len(calls) == 2
    assert not np.array_equal(second.velocity,first.velocity)
    changed.bullets[0,2] += 2
    assert cache.get(changed,old) is not second
    assert np.isfinite(obs['full_bullets']).all()


def test_cached_key_mapping_is_identical_and_callers_cannot_mutate_templates():
    for action in range(36):
        keys=action_keys(action)
        assert movement_action(keys)==action%18
        keys.clear()
        assert 'z' in action_keys(action)
    assert movement_action({'q'}) is None


def test_display_catalog_reuses_metadata_but_invalidates_changed_entries(tmp_path, monkeypatch):
    first = model(tmp_path)
    catalog = model_store.CheckpointCatalog(tmp_path/'models')
    inspect = model_store.inspect_checkpoint; calls=[]
    def counted(path,**kwargs):calls.append(path);return inspect(path,**kwargs)
    monkeypatch.setattr(model_store,'inspect_checkpoint',counted)
    catalog.scan();catalog.scan()
    assert calls == [first]
    assert catalog.scan()==model_store.list_checkpoints(tmp_path/'models')
    calls.clear()
    meta = json.loads(first.with_suffix('.json').read_text(encoding='utf-8'))
    atomic_json(first.with_suffix('.json'),dict(meta,steps=2048))
    assert catalog.scan()[0].metadata['steps']==2048 and len(calls)==1
    first.unlink()
    assert catalog.scan()==[]


def test_background_catalog_does_not_hold_control_lock(tmp_path,monkeypatch):
    model(tmp_path)
    service=ConsoleService(Config('unused.exe'),root=tmp_path,preview=True)
    started=threading.Event();finish=threading.Event();published=threading.Event()
    scan=service._catalog_cache.scan
    def blocked():
        started.set();assert finish.wait(3)
        return scan()
    monkeypatch.setattr(service._catalog_cache,'scan',blocked)
    service._catalog_at=-float('inf')
    before=service.refresh_models()
    assert started.wait(2)
    try:
        def control_lock():
            with service._lock:published.set()
        thread=threading.Thread(target=control_lock);thread.start()
        assert published.wait(1), 'Catalog I/O blocked the control lock'
        assert service.models()==service._model_view and service._catalog is before
        thread.join()
    finally:
        finish.set()
        service._catalog_thread.join(3)


def test_diagnostic_process_preserves_arrays_json_nulls_and_drains(tmp_path):
    diagnostics=SessionDiagnostics(tmp_path,'play')
    info={'action':4,'position':{'x':192,'y':384},
        'decision_enemy_bodies':np.array([[1,200,300,float('nan'),4,0]]),
        'decision_lasers':np.empty((0,8)), 'input_latency':{'pre_policy_work':{'callback_ms':.2}}}
    diagnostics.record(info,123)
    assert diagnostics.record_trace('observations-test.npz',{'values':np.array([1,2]),'metadata':np.array('test')})
    diagnostics.close(timeout=5)
    summary=json.loads((tmp_path/'diagnostics_summary.json').read_text(encoding='utf-8'))
    assert summary['complete'] and summary['written_actions']==1 and summary['written_traces']==1
    assert summary['errors']==[]
    import csv
    with (tmp_path/'actions.csv').open(encoding='utf-8',newline='') as stream:row=next(csv.DictReader(stream))
    assert json.loads(row['decision_enemy_bodies'])[0][3] is None
    assert json.loads(row['input_pre_policy_work'])=={'callback_ms':.2}
    with np.load(tmp_path/'observations-test.npz',allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['values'],[1,2])
    diagnostics.close(timeout=1)


def test_timing_work_is_bound_to_exact_observation_not_leaked_across_refresh():
    timing=InputTiming();current=frame()
    timing.record_work(current,callback_ms=.3)
    dispatch={'started_ns':1,'completed_ns':2,'send_ms':0.,'events':0}
    timing.begin(current,None,{'z'},{'z'},dispatch)
    assert timing.latest['pre_policy_work']=={'callback_ms':.3}
    timing.record_work(current,buffer_add_ms=.2)
    timing.begin(replace(current),None,{'z'},{'z'},dispatch)
    assert timing.latest['pre_policy_work']=={}


def test_native_owned_scatter_and_indirect_reads_use_only_this_test_process():
    mod=extension()
    if mod is None:pytest.skip('optional ABI v3 extension not built')
    assert mod.API_VERSION==3
    memory=ct.create_string_buffer(128);base=ct.addressof(memory)
    ct.memmove(base+64,b'owned-state',11)
    ct.c_uint64.from_buffer(memory).value=base+64
    process=win32.ReadOnlyProcess.__new__(win32.ReadOnlyProcess)
    process.base,process.size=base,len(memory)
    process.handle=win32.open_process(0x10|0x1000,False,os.getpid())
    process._native_reader=mod
    assert process.handle
    try:
        blocks=process.read_many(((64,5),(69,6)))
        assert blocks==[b'owned',b'-state']
        assert process.read_indirect(0,0,11)==b'owned-state'
        ct.memmove(base+64,b'changed',7)
        assert blocks==[b'owned',b'-state']
        with pytest.raises(ValueError):process.read_many(((127,2),))
        ct.c_uint64.from_buffer(memory).value=0
        assert process.read_indirect(0,0,11) is None
        assert process.last_batch_timing['native_compute_ms']>=0
    finally:process.close()


def test_default_capture_keeps_reusable_bulk_buffers_and_explicit_modes():
    if extension() is None:pytest.skip('optional ABI v3 extension not built')
    process=win32.ReadOnlyProcess.__new__(win32.ReadOnlyProcess)
    process.configure_reads('auto')
    assert process._native_reader is not None and not process.native_batch
    process.configure_reads('rust')
    assert process.native_batch
    process.configure_reads('python')
    assert process._native_reader is None and not process.native_batch
    with pytest.raises(ValueError):process.configure_reads('invalid')


def test_batch_capture_keeps_cross_tick_and_empty_opening_semantics():
    from test_state_capture import UpdatingGameMemory, reader_for
    class BatchMemory(UpdatingGameMemory):
        native_batch=True
        def read_many(self, requests):
            blocks=[]
            for address,size in requests:
                if size==8:blocks.append(self.read(address,size))
                else:
                    buffer=bytearray(size);self.read_into(address,buffer);blocks.append(bytes(buffer))
            return blocks
    plain=reader_for(UpdatingGameMemory());batch=reader_for(BatchMemory())
    for _ in range(12):
        a,b=plain.read(),batch.read()
        assert a.tick==b.tick and a.resources==b.resources and len(b.bullets)==0
        assert batch.snapshot_diagnostics['crossed_tick']
        assert b.capture_metrics['capture_backend']=='rust'


def test_diagnostic_backpressure_is_explicit_and_nonblocking():
    import queue
    from th06ai.diagnostic_writer import DiagnosticWriter
    writer=DiagnosticWriter.__new__(DiagnosticWriter)
    writer.lock=threading.Lock();writer.jobs=queue.Queue(maxsize=1)
    writer.closed=writer.failed=False;writer.pending_traces=writer.dropped=0;writer.errors=[]
    assert writer.submit(('event',{}))
    assert not writer.submit(('death','fixture'),critical=True)
    assert writer.dropped==1 and 'critical' in writer.errors[0]


def test_diagnostic_helper_startup_failure_is_reported_without_failing_control(tmp_path,monkeypatch):
    from th06ai import diagnostic_writer
    def denied(*args,**kwargs):raise PermissionError('fixture launch denied')
    monkeypatch.setattr(diagnostic_writer.subprocess,'Popen',denied)
    writer=diagnostic_writer.DiagnosticWriter(tmp_path)
    assert not writer.submit(('event',{})) and not writer.close()
    assert 'fixture launch denied' in writer.status()['errors'][0]


@pytest.mark.parametrize('kwargs',[{'trajectory':'invalid'},{'compression_level':10},
                                  {'checkpoint_every_rollouts':0},{'trajectory_every_rollouts':True}])
def test_invalid_diagnostics_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):DiagnosticsConfig(**kwargs)
