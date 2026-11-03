"""Bounded raw snapshot ring and asynchronous diagnostic I/O; no control policy."""
from __future__ import annotations
from collections import deque
import csv
import json
from pathlib import Path
import threading
import time
import numpy as np
from .files import _json_default, atomic_json
from .log_rows import action_row, episode_row, serialized_row
from .diagnostic_writer import DiagnosticWriter, write_npz


class SessionDiagnostics:
    def __init__(self, run_dir, mode='train', *, max_frames=180, queue_size=256):
        self.root=Path(run_dir)
        self.root.mkdir(parents=True,exist_ok=True)
        self.mode=mode
        self.frames=deque(maxlen=max_frames)
        self.controls=deque(maxlen=512)
        self.lock=threading.Lock()
        self.deaths=0
        self.written_deaths=0
        self.writer=DiagnosticWriter(self.root, capacity=queue_size)

    def observe(self, frame):
        with self.lock:
            if self.frames and (self.frames[-1].stage_id!=frame.stage_id or frame.tick<self.frames[-1].tick):
                self.frames.clear();self.controls.clear()
            if self.frames and (self.frames[-1].tick,self.frames[-1].state,self.frames[-1].phase)==(frame.tick,frame.state,frame.phase):
                self.frames[-1]=frame
            else:self.frames.append(frame)
            while self.frames and frame.tick-self.frames[0].tick>180:self.frames.popleft()

    def observe_control(self, sample):
        with self.lock:self.controls.append(sample)

    def event(self, **values):
        self._put(('event',{'monotonic_ns':time.perf_counter_ns(),**values}))

    def record(self, info, steps):
        if info.get('died'):
            self.deaths+=1
            death_id=f'death-{self.deaths:04d}-step-{int(steps):09d}'
            info['death_id']=death_id
            with self.lock:frames,controls=list(self.frames),list(self.controls)
            self._put(('death',death_id,frames,controls,dict(info)),critical=True)
        if self.mode!='train':
            self.record_action(info, steps)
            if info.get('died') or info.get('episode_end_reason'):
                self._put(('episode',episode_row(info,steps)),critical=True)

    def _put(self, job, critical=False):
        return self.writer.submit(job, critical=critical)

    def record_action(self, info, steps):
        return self._put(('action_info', dict(info), int(steps)))

    def record_trace(self, filename, payload, compression_level=1):
        return self._put(('trace', filename, payload, compression_level))

    def _csv(self,name,row):
        path=self.root/name
        new=not path.exists()
        with path.open('a',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(row))
            if new:writer.writeheader()
            writer.writerow(serialized_row(row))

    @staticmethod
    def _ragged(rows,width):
        arrays=[np.asarray(row,np.float32).reshape(-1,width) for row in rows]
        offsets=np.r_[0,np.cumsum([len(a) for a in arrays])].astype(np.int64)
        return np.concatenate(arrays) if arrays else np.empty((0,width),np.float32),offsets

    def _death(self,death_id,frames,controls,info):
        info = dict(info)
        for key in ('decision_enemy_bodies', 'next_enemy_bodies'):
            value = info.get(key)
            if isinstance(value, np.ndarray):info[key] = np.where(np.isfinite(value), value, None).tolist()
        bullets=[]
        for f in frames:
            sizes=f.bullet_sizes if f.bullet_sizes is not None else np.full((len(f.bullets),2),np.nan)
            states=f.bullet_states if getattr(f,'bullet_states',None) is not None else np.full(len(f.bullets),-1)
            bullets.append(np.column_stack((f.bullets,sizes,states)))
        payload={
            'frames':np.asarray([[f.tick,f.stage_id or 0,f.x,f.y,f.state,int(f.paused),int(f.focused),
                  f.player_radius if f.player_radius is not None else np.nan,f.input_bits if f.input_bits is not None else -1,
                  f.resources.power if f.resources else -1,f.resources.bombs if f.resources else -1,
                  f.resources.lives if f.resources else -1] for f in frames],np.float64).reshape(-1,12),
            'capture_started_ns':np.asarray([f.capture_started_ns for f in frames],np.int64),
            'capture_finished_ns':np.asarray([f.capture_finished_ns for f in frames],np.int64),
            'input_observed_ns':np.asarray([f.input_observed_ns for f in frames],np.int64),
            'phases':np.asarray([f.phase for f in frames]),
            'controls':np.asarray([[f.tick,f.stage_id or 0,f.x,f.y,f.state,int(f.paused),
                   f.input_bits if f.input_bits is not None else -1] for f in controls],np.float64).reshape(-1,7),
            'control_capture_started_ns':np.asarray([f.capture_started_ns for f in controls],np.int64),
            'control_capture_finished_ns':np.asarray([f.capture_finished_ns for f in controls],np.int64),
        }
        for name,rows,width in (('bullets',bullets,6),('enemy_bodies',[f.enemy_bodies for f in frames],6),
                               ('lasers',[f.lasers for f in frames],8),('items',[f.items for f in frames],6)):
            payload[name],payload[name+'_offsets']=self._ragged(rows,width)
        payload['metadata']=np.array(json.dumps({'version':1,'death_id':death_id,
             'native_damage_source_id':None,'source':'observed raw snapshots, not a native hit-source label',
             'bullet_fields':['pool_slot','x','y','width','height','state'],
             'frame_fields':['tick','stage_id','x','y','player_state','paused','focused','player_radius','input_bits','power','bombs','lives'],
             'control_fields':['tick','stage_id','x','y','player_state','paused','input_bits'],
             'bounds':[list(f.bounds) for f in frames],'phase_evidence':[f.phase_evidence for f in frames],
             'decision':info},ensure_ascii=False,default=_json_default))
        write_npz(self.root/(death_id+'.npz'), payload)
        self.written_deaths+=1

    def status(self):
        return {'deaths':self.deaths, **self.writer.status()}

    def close(self,timeout=3.):
        complete=self.writer.close(timeout)
        atomic_json(self.root/'diagnostics_summary.json',{**self.status(),'complete':complete})
