"""Batch-one inference buffers and optional CUDA graphs. No game or input handles."""
from __future__ import annotations
from .i18n import msg
from .config import INFERENCE_READBACK_MODES, READBACK_AB_STEPS
from .thread_profile import thread_cycle_reader
from collections import deque
from dataclasses import dataclass
from random import Random
import threading
import time
import numpy as np
import torch

_CONFIGURED_READBACK = object()


@dataclass
class Decision:
    actions: np.ndarray
    values: torch.Tensor
    log_probs: torch.Tensor
    bomb_probability: float
    timing: dict


class LiveReadbackTrial:
    """One method per action. Randomized pair order uses a separate CPU RNG."""
    def __init__(self):
        order_rng = Random(0)
        self._reverse = [bool(order_rng.getrandbits(1)) for _ in range(READBACK_AB_STEPS // 2)]
        self._completion = torch.cuda.Event(blocking=True)
        self._cycles = thread_cycle_reader()

    def predict(self, runner, observation, index, lease):
        pair, position = divmod(index, 2)
        use_event = self._reverse[pair] != bool(position)
        decision = runner.predict(observation, lease=lease,
                                  readback_event=self._completion if use_event else None,
                                  readback_counter=self._cycles)
        decision.timing['readback_ab'] = {'pair': pair, 'position': position,
                                         'method': 'blocking_event' if use_event else 'synchronous'}
        return decision


class PolicyRunner:
    """One serialized policy owner; training and graph replay share parameter storage."""
    def __init__(self, policy, backend='auto', *, profile=False, readback='auto'):
        if backend not in ('auto', 'eager', 'cuda_graph'):
            raise ValueError(msg('backend.unknown_inference_backend'))
        if readback not in INFERENCE_READBACK_MODES:
            raise ValueError(msg('backend.inference_readback_must_be_auto_or_synchronous'))
        self.policy, self.requested = policy, backend
        self.requested_readback = readback
        self.device = policy.device
        if backend == 'cuda_graph' and self.device.type != 'cuda':
            raise ValueError(msg('backend.cuda_graph_requires_a_cuda_model'))
        self.lock = threading.RLock()
        cuda = self.device.type == 'cuda'
        shapes = {k:(1,*space.shape) for k,space in policy.observation_space.spaces.items()}
        count = sum(int(np.prod(shape)) for shape in shapes.values())
        self.host_flat = torch.empty(count, dtype=torch.float32, pin_memory=cuda)
        self.input_flat = torch.empty_like(self.host_flat, device=self.device) if cuda else self.host_flat
        self.host, self.inputs = {}, {}
        offset = 0
        for key, shape in shapes.items():
            size = int(np.prod(shape))
            self.host[key] = self.host_flat[offset:offset+size].view(shape)
            self.inputs[key] = self.input_flat[offset:offset+size].view(shape)
            offset += size
        self.views = {k:v.numpy() for k,v in self.host.items()}
        self.result_host = torch.empty((1,4), dtype=torch.float32, pin_memory=cuda)
        self.result_view = self.result_host.numpy()
        self.graphs, self.outputs = {}, {}
        self.ready = False
        self.readback_trial = None
        self._default_readback_event = None
        self.engine = 'cuda_eager' if cuda else 'cpu_eager'
        self.fallback_reason = None
        self.timings = deque(maxlen=256)
        self.profile = bool(profile)
        self._readback_cycles = thread_cycle_reader() if profile and cuda else None
        self.events = ({mode:(torch.cuda.Event(enable_timing=True,external=True),
                              torch.cuda.Event(enable_timing=True,external=True)) for mode in (False,True)}
                       if profile and cuda else {})

    def _fill(self, observation):
        for key,target in self.views.items():
            source = np.asarray(observation[key], np.float32)
            if source.shape == target.shape[1:]:source = source[None]
            if source.shape != target.shape or not np.isfinite(source).all():
                raise ValueError(msg('backend.invalid_shape_or_values_for_inference_observation', key))
            np.copyto(target, source)

    def _forward(self, deterministic):
        actions, values, log_probs = self.policy(self.inputs, deterministic=deterministic)
        bomb = getattr(self.policy, 'last_bomb_probability', torch.zeros_like(log_probs))
        return torch.stack((actions.reshape(-1).float(),values.reshape(-1),log_probs.reshape(-1),bomb.reshape(-1)),dim=-1)

    def _transfer(self):
        if self.device.type == 'cuda':
            self.input_flat.copy_(self.host_flat, non_blocking=True)

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
                else:raise
            finally:
                self.policy._fast_inference = previous
            if self.engine == 'cuda_graph' and self.requested_readback == 'auto':
                self._default_readback_event = torch.cuda.Event(blocking=True)
            self.ready = True

    def _read_result(self, output, completion):
        self.result_host.copy_(output.detach(), non_blocking=completion is not None)
        if completion is not None:
            # The pinned CPU view is safe to consume only after the D2H copy.
            completion.record(torch.cuda.current_stream(self.device))
            completion.synchronize()
        return self.result_view

    def predict(self, observation, deterministic=False, lease=lambda: None, *, readback_event=_CONFIGURED_READBACK,
                readback_counter=None):
        if readback_event is not _CONFIGURED_READBACK and readback_event is not None and self.device.type != 'cuda':
            raise ValueError(msg('backend.cuda_readback_requires_a_cuda_model'))
        with self.lock, torch.no_grad():
            if not self.ready:self.warmup(observation, lease)
            completion = self._default_readback_event if readback_event is _CONFIGURED_READBACK else readback_event
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
                counter = self._readback_cycles if readback_counter is None else readback_counter
                cycles_before = counter() if counter is not None else None
                packed = self._read_result(output, completion)
                cycles_after = counter() if counter is not None else None
            finally:self.policy._fast_inference = previous
            if not np.isfinite(packed).all():raise RuntimeError(msg('backend.the_policy_returned_non_finite_values'))
            finished = time.perf_counter_ns()
            events = self.events.get(bool(deterministic))
            timing = {'engine':self.engine,'prepare_ms':(filled-started)/1e6,
                      'execute_ms':(finished-filled)/1e6,'total_ms':(finished-started)/1e6,
                      'submit_ms':(submitted-filled)/1e6,'result_wait_copy_ms':(finished-submitted)/1e6,
                      'gpu_stream_ms':events[0].elapsed_time(events[1]) if events else None,
                      'fallback_reason':self.fallback_reason,
                      'readback':'blocking_event' if completion is not None else 'synchronous'}
            if self.profile or readback_counter is not None:
                timing['readback_thread_cycles'] = (cycles_after - cycles_before
                    if cycles_before is not None and cycles_after is not None and cycles_after >= cycles_before else None)
            self.timings.append(timing['total_ms'])
            return Decision(packed[:,0].astype(np.int64),
                            torch.from_numpy(packed[:,1:2].copy()),
                            torch.from_numpy(packed[:,2].copy()),float(packed[0,3]),timing)

    def diagnostics(self):
        samples = np.asarray(self.timings)
        return {'engine':self.engine,'samples':len(samples),'fallback_reason':self.fallback_reason,
                'readback':('blocking_event' if self._default_readback_event is not None else 'synchronous')
                           if self.ready else None,
                'p50_ms':float(np.percentile(samples,50)) if len(samples) else None,
                'p95_ms':float(np.percentile(samples,95)) if len(samples) else None}
