"""Optional acceleration; importing this module never touches game input."""
from __future__ import annotations
from .i18n import msg
import importlib
from functools import lru_cache
import threading
import time

KEY_ORDER = ('up', 'down', 'left', 'right', 'shift', 'z', 'x', 'esc', 'r', 'q')

@lru_cache(maxsize=1)
def extension():
    # The loaded PyO3 binary belongs to this process; replacing it already
    # requires a restart. Resolve it once, not at every geometry sweep.
    try:
        module=importlib.import_module('th06ai.native_runtime._native_backend')
    except (ImportError, OSError):
        try:module=importlib.import_module('th06ai._native_backend')
        except (ImportError, OSError):return None
    return module if getattr(module,'API_VERSION',None)==3 else None

def backend_status():
    mod=extension()
    return {'available':mod is not None,'api_version':getattr(mod,'API_VERSION',None)}


@lru_cache(maxsize=1)
def search_extension():
    current=extension()
    if current is not None and getattr(current, 'SEARCH_API_VERSION', None)==1:
        return current
    try:
        mod=importlib.import_module('th06ai.native_planner._native_backend')
    except (ImportError,OSError):
        return None
    return mod if getattr(mod,'SEARCH_API_VERSION',None)==1 else None


def rust_plan(start,pending,low,high,velocities,directions,control,bounds,items,weights,target,
              bullets,velocity,radii,uncertainty,bodies,known,body_known,
              delay,late,horizon,durations,delays,complete,beam,branches,fast,age):
    """One owned snapshot and one detached call for the entire search."""
    import numpy as np
    mod=search_extension()
    if mod is None:raise RuntimeError(msg('backend.the_rust_search_extension_is_unavailable'))
    m,k,q=len(bullets),len(bodies.positions),len(items)
    data=np.empty(88+7*m+9*k+5*q,dtype='<f8')
    data[:8]=(*start,*pending,*low,*high)
    data[8:44]=velocities.ravel();data[44:80]=directions.ravel()
    data[80:87]=(control,bounds[0]*2+bounds[2],bounds[1]*2+bounds[3],
                 target is not None,*(target if target is not None else (0.,0.,0.)))
    offset=87
    circles=data[offset:offset+7*m].reshape(m,7);offset+=7*m
    circles[:,:2]=bullets;circles[:,2:4]=velocity
    circles[:,4]=radii;circles[:,5]=uncertainty;circles[:,6]=known
    contacts=data[offset:offset+9*k].reshape(k,9);offset+=9*k
    contacts[:,:2]=bodies.positions;contacts[:,2:4]=bodies.velocity
    contacts[:,4:6]=bodies.half_extents;contacts[:,6]=bodies.uncertainty
    contacts[:,7]=bodies.angles if bodies.angles is not None else 0.;contacts[:,8]=body_known
    goals=data[offset:offset+5*q].reshape(q,5);offset+=5*q
    goals[:,:2]=items[:,1:3];goals[:,2:4]=items[:,4:6];goals[:,4]=weights
    data[offset]=bodies.player_radius
    packed,meta=mod.plan_bytes(data.tobytes(),m,k,q,float(delay),float(late),float(horizon),
                              durations,delays,bool(complete),int(beam),int(branches),float(fast),float(age))
    output=np.frombuffer(packed,dtype='<f4')
    npath=len(durations)+2;offset=18*8
    features=output[:offset].reshape(18,8).copy()
    paths=output[offset:offset+18*npath*2].reshape(18,npath,2).copy();offset+=18*npath*2
    minimum=output[offset:offset+18].copy();offset+=18
    collision=output[offset:offset+18].copy();offset+=18
    navigation=output[offset:offset+18*9].reshape(18,9).copy();offset+=18*9
    decision=output[offset:offset+8].copy()
    return features,paths,minimum,collision,navigation,decision,meta

def rust_sweep(starts, velocities, duration, start_tick, low, high, bullets, bullet_velocity,
               radii, uncertainty, bodies=None):
    import numpy as np
    mod=extension()
    if mod is None:raise RuntimeError(msg('backend.build_the_rust_extension_or_select_the'))
    starts=np.asarray(starts,np.float32).reshape(-1,2)
    n,m=len(starts),len(bullets)
    circles=np.column_stack((bullets,bullet_velocity,radii,uncertainty)).astype('<f4')
    k=len(bodies.positions) if bodies is not None else 0
    angles=getattr(bodies,'angles',None)
    contacts=(np.column_stack((bodies.positions,bodies.velocity,bodies.half_extents,bodies.uncertainty,
                              np.zeros(k) if angles is None else angles)).astype('<f4')
              if k else np.empty((0,8),np.float32))
    # tobytes creates an owned, immutable snapshot before Rust releases the GIL.
    data=np.concatenate((starts.ravel(),np.asarray(velocities,np.float32).ravel(),
             np.broadcast_to(duration,(n,)),np.broadcast_to(start_tick,(n,)),np.asarray(low),np.asarray(high),
             circles.ravel(),contacts.ravel())).astype('<f4').tobytes()
    result=np.frombuffer(mod.sweep_bytes(data,n,m,k,float(bodies.player_radius) if k else 0.),dtype='<f4').reshape(n,5)
    return result[:,:2].copy(),result[:,2].copy(),result[:,3].copy(),result[:,4].copy()

class RustKeyController:
    """Serializes Python state and the native mutex; no callbacks under native lock."""
    def __init__(self,window,*,dry_run=False):
        mod=extension()
        if mod is None:raise RuntimeError(msg('backend.the_rust_input_extension_has_not_been'))
        self.window=window
        self._native=mod.NativeInput(window.pid,window.hwnd,dry_run)
        self._lock=threading.RLock()
        self.held=set()

    def set(self,keys):
        from .win32 import GameFocusLost
        keys=set(keys)
        if not keys<=set(KEY_ORDER):raise ValueError(msg('backend.unsupported_game_key'))
        with self._lock:
            if not self.window.focused:
                self.release()
                raise GameFocusLost(msg('backend.the_game_lost_focus_and_key_input'))
            started=time.perf_counter_ns()
            if keys==self.held:
                return {'started_ns':started,'completed_ns':started,'events':0,'send_ms':0.,'backend':'rust'}
            try:
                _,events=self._native.set(sum(1<<i for i,k in enumerate(KEY_ORDER) if k in keys))
            except Exception as error:
                self.held.clear()
                if 'game is not foreground' in str(error):
                    raise GameFocusLost(msg('backend.the_game_lost_focus_and_key_input')) from None
                raise
            completed=time.perf_counter_ns()
            self.held=keys
            return {'started_ns':started,'completed_ns':completed,'events':events,'send_ms':(completed-started)/1e6,'backend':'rust'}

    def release(self):
        with self._lock:
            try:return self._native.set(0)
            finally:self.held.clear()

    def reconcile(self,name):
        from .win32 import get_async_key,VIRTUAL_KEYS
        with self._lock:
            if name in self.held and not get_async_key(VIRTUAL_KEYS[name]) & 0x8000:
                remaining=self.held-{name}
                self.set(remaining)

def key_controller(window,backend='python'):
    if backend=='rust' or backend=='auto' and extension() is not None:
        return RustKeyController(window)
    from .win32 import KeyController
    return KeyController(window)
