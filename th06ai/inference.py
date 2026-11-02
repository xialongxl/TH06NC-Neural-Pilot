"""Batch-one inference buffers and optional CUDA graphs. No game or input handles."""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import threading
import time
import numpy as np
import torch


@dataclass
class Decision:
    actions: np.ndarray
    values: torch.Tensor
    log_probs: torch.Tensor
    bomb_probability: float
    timing: dict


class PolicyRunner:
    """One serialized policy owner; training and graph replay share parameter storage."""
    def __init__(self, policy, backend='auto', *, profile=False):
        if backend not in ('auto', 'eager', 'cuda_graph'):
            raise ValueError('未知推理后端')
        self.policy, self.requested = policy, backend
        self.device = policy.device
        if backend == 'cuda_graph' and self.device.type != 'cuda':
            raise ValueError('CUDA Graph 需要 CUDA 模型')
        self.lock = threading.RLock()
        cuda = self.device.type == 'cuda'
        self.host = {k: torch.empty((1, *space.shape), dtype=torch.float32, pin_memory=cuda)
                     for k,space in policy.observation_space.spaces.items()}
        self.views = {k:v.numpy() for k,v in self.host.items()}
        self.inputs = {k:torch.empty_like(v, device=self.device) if cuda else v for k,v in self.host.items()}
        self.graphs, self.outputs = {}, {}
        self.ready = False
        self.engine = 'cuda_eager' if cuda else 'cpu_eager'
        self.fallback_reason = None
        self.timings = deque(maxlen=256)
        self.profile = bool(profile)
        self.events = ({mode:(torch.cuda.Event(enable_timing=True,external=True),
                              torch.cuda.Event(enable_timing=True,external=True)) for mode in (False,True)}
                       if profile and cuda else {})

    def _fill(self, observation):
        for key,target in self.views.items():
            source = np.asarray(observation[key], np.float32)
            if source.shape == target.shape[1:]:source = source[None]
            if source.shape != target.shape or not np.isfinite(source).all():
                raise ValueError(f'推理观测 {key} 形状或数值异常')
            np.copyto(target, source)

    def _forward(self, deterministic):
        actions, values, log_probs = self.policy(self.inputs, deterministic=deterministic)
        bomb = getattr(self.policy, 'last_bomb_probability', torch.zeros_like(log_probs))
        return torch.stack((actions.reshape(-1).float(),values.reshape(-1),log_probs.reshape(-1),bomb.reshape(-1)),dim=-1)

    def _transfer(self):
        if self.device.type == 'cuda':
            for key,value in self.inputs.items():value.copy_(self.host[key], non_blocking=True)

    def _submit(self, deterministic):
        events = self.events.get(bool(deterministic))
        if events:events[0].record()
        self._transfer()
        output = self._forward(deterministic)
        if events:events[1].record()
        return output

    def warmup(self, observation=None, lease=lambda: None):
        with self.lock, torch.no_grad():
            if self.ready:return
            observation = observation or {k:np.zeros(v.shape, np.float32) for k,v in self.views.items()}
            self._fill(observation)
            self.policy.set_training_mode(False)
            previous = getattr(self.policy, '_fast_inference', False)
            self.policy._fast_inference = True
            try:
                if self.device.type == 'cuda' and self.requested != 'eager':
                    stream = torch.cuda.Stream(device=self.device)
                    stream.wait_stream(torch.cuda.current_stream(self.device))
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            lease();self._submit(False)
                    torch.cuda.current_stream(self.device).wait_stream(stream)
                    torch.cuda.synchronize(self.device)
                    for deterministic in (False, True):
                        lease()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            output = self._submit(deterministic)
                        self.graphs[deterministic], self.outputs[deterministic] = graph, output
                    self.engine = 'cuda_graph'
                else:
                    self._submit(False)
                self.ready = True
            except RuntimeError as error:
                text = str(error).lower()
                # A capture limitation falls back to CUDA eager, never CPU.
                # Device loss, OOM and illegal access remain real failures.
                if self.requested == 'auto' and any(x in text for x in ('capture', 'capturing', 'graph')) and not any(
                        x in text for x in ('out of memory','illegal memory','device-side assert','device lost')):
                    self.graphs.clear();self.outputs.clear()
                    self.fallback_reason = str(error)
                    self.engine = 'cuda_eager'
                    self._submit(False)
                    self.ready = True
                else:raise
            finally:
                self.policy._fast_inference = previous

    def predict(self, observation, deterministic=False, lease=lambda: None):
        with self.lock, torch.no_grad():
            if not self.ready:self.warmup(observation, lease)
            lease()
            started = time.perf_counter_ns()
            self._fill(observation)
            filled = time.perf_counter_ns()
            self.policy.set_training_mode(False)
            previous = getattr(self.policy, '_fast_inference', False)
            self.policy._fast_inference = True
            try:
                if self.graphs:
                    self.graphs[bool(deterministic)].replay()
                    output = self.outputs[bool(deterministic)]
                else:
                    output = self._submit(deterministic)
                submitted = time.perf_counter_ns()
                # One small D2H transfer for action/value/log-prob/B probability.
                packed = output.detach().cpu().numpy().copy()
            finally:self.policy._fast_inference = previous
            if not np.isfinite(packed).all():raise RuntimeError('神经网络推理结果包含无效数值')
            finished = time.perf_counter_ns()
            events = self.events.get(bool(deterministic))
            timing = {'engine':self.engine,'prepare_ms':(filled-started)/1e6,
                      'execute_ms':(finished-filled)/1e6,'total_ms':(finished-started)/1e6,
                      'submit_ms':(submitted-filled)/1e6,'result_wait_copy_ms':(finished-submitted)/1e6,
                      'gpu_stream_ms':events[0].elapsed_time(events[1]) if events else None,
                      'fallback_reason':self.fallback_reason}
            self.timings.append(timing['total_ms'])
            return Decision(packed[:,0].astype(np.int64),
                            torch.from_numpy(packed[:,1:2].copy()),
                            torch.from_numpy(packed[:,2].copy()),float(packed[0,3]),timing)

    def diagnostics(self):
        samples = np.asarray(self.timings)
        return {'engine':self.engine,'samples':len(samples),'fallback_reason':self.fallback_reason,
                'p50_ms':float(np.percentile(samples,50)) if len(samples) else None,
                'p95_ms':float(np.percentile(samples,95)) if len(samples) else None}
