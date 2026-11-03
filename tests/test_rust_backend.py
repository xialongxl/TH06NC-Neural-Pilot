from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import numpy as np
import pytest
from th06ai.acceleration import extension,rust_sweep,RustKeyController
from th06ai.planning import swept_segments

pytestmark=pytest.mark.skipif(extension() is None,reason='optional Rust extension not built')


def test_collision_sweeps_resolve_the_loaded_extension_once(monkeypatch):
    from th06ai import acceleration
    original=acceleration.importlib.import_module
    lookups=[]
    def counted(name):
        lookups.append(name)
        return original(name)
    extension.cache_clear()
    try:
        monkeypatch.setattr(acceleration.importlib,'import_module',counted)
        for _ in range(8):
            result=rust_sweep(np.array([[100,100]],np.float32),np.array([[2,0]],np.float32),
                2,0,np.array([8,16]),np.array([376,432]),np.empty((0,2)),np.empty((0,2)),np.empty(0),np.empty(0))
            np.testing.assert_array_equal(result[0],[[104,100]])
        assert lookups in (['th06ai.native_runtime._native_backend'],
                           ['th06ai.native_runtime._native_backend','th06ai._native_backend'])
    finally:extension.cache_clear()

@pytest.mark.parametrize('seed',range(12))
def test_continuous_collision_parity_with_walls_enemies_and_full_bullet_pool(seed):
    rng=np.random.default_rng(seed)
    n,m,k=41,640,12
    low=np.array([8,16],np.float32);high=np.array([376,432],np.float32)
    starts=rng.uniform(low,high,(n,2)).astype(np.float32)
    starts[:2]=[low,high]
    speed=rng.uniform(-5,5,(n,2)).astype(np.float32)
    duration=rng.uniform(.1,32,n).astype(np.float32)
    tick=rng.uniform(0,24,n).astype(np.float32)
    positions=rng.uniform(-40,480,(m,2)).astype(np.float32)
    velocity=rng.uniform(-5,5,(m,2)).astype(np.float32)
    radii=rng.uniform(2,14,m).astype(np.float32)
    error=rng.uniform(0,.3,m).astype(np.float32)
    bodies=SimpleNamespace(positions=rng.uniform(0,448,(k,2)).astype(np.float32),
        velocity=rng.uniform(-1,1,(k,2)).astype(np.float32),half_extents=rng.uniform(4,24,(k,2)).astype(np.float32),
        uncertainty=np.zeros(k,np.float32),player_radius=2.,angles=rng.uniform(-np.pi,np.pi,k).astype(np.float32))
    args=(starts,speed,duration,tick,low,high,positions,velocity,radii,error,bodies)
    expected=swept_segments(*args);actual=rust_sweep(*args)
    for a,b in zip(expected,actual):np.testing.assert_allclose(a,b,atol=.003,rtol=3e-5)

def test_native_validates_owned_buffers_and_bad_shapes():
    mod=extension()
    with pytest.raises(ValueError):mod.sweep_bytes(b'bad',1,0,0,0.)
    with pytest.raises(ValueError):mod.sweep_bytes(np.zeros(10,np.float32).tobytes(),999999,0,0,0.)
    with pytest.raises(ValueError):mod.sweep_bytes(np.full(10,np.nan,np.float32).tobytes(),1,0,0,0.)

def test_parallel_owned_collision_snapshots():
    with ThreadPoolExecutor(4) as pool:list(pool.map(test_continuous_collision_parity_with_walls_enemies_and_full_bullet_pool,range(8)))

def test_concurrent_native_input_is_serialized_without_sending_real_keys():
    c=RustKeyController(SimpleNamespace(pid=0,hwnd=0,focused=True),dry_run=True)
    def change(i):
        for j in range(200):c.set({'z','shift'} if (i+j)%2 else {'left'})
    with ThreadPoolExecutor(6) as pool:list(pool.map(change,range(6)))
    c.release()
    assert c.held==set() and c._native.held==0

def test_native_foreground_race_uses_same_cooperative_pause_signal_as_python():
    from th06ai.win32 import GameFocusLost
    c=RustKeyController(SimpleNamespace(pid=0,hwnd=0,focused=True),dry_run=True)
    def lost(_):raise RuntimeError('game is not foreground')
    c._native=SimpleNamespace(set=lost)
    with pytest.raises(GameFocusLost):c.set({'z'})
    assert not c.held
