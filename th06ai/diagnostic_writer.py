"""Bounded, CPU-only diagnostic process. Private pipes; never opens a game/GPU."""
from __future__ import annotations

import json
import os
from pathlib import Path
import pickle
import queue
import subprocess
import sys
import threading
import time
import zipfile

from .battle_view import BattleViewPublisher, publish_battle_view


def write_npz(path, payload, compression_level=1):
    import numpy as np
    path = Path(path)
    if path.exists():
        return
    pending = path.with_name(path.stem + f'.{os.getpid()}.pending.npz')
    try:
        method = zipfile.ZIP_DEFLATED if compression_level else zipfile.ZIP_STORED
        with zipfile.ZipFile(pending, 'w', compression=method, compresslevel=compression_level or None,
                             allowZip64=True) as archive:
            for name, value in payload.items():
                with archive.open(name + '.npy', 'w', force_zip64=True) as stream:
                    np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


class DiagnosticWriter:
    """Only a bounded enqueue is on the control thread; copying/encoding is off it."""
    def __init__(self, root, *, capacity=256):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs = queue.Queue(maxsize=capacity)
        self.lock = threading.Lock()
        self.errors = []
        self.dropped = 0
        self.counts = {'written_deaths': 0, 'written_actions': 0, 'written_traces': 0}
        self.pending_traces = 0
        self.closed = self.failed = False
        self.close_result = None
        self.sender_stop = threading.Event()
        self.process = self.sender = self.receiver = None
        self.battle_view = BattleViewPublisher(self.root)
        try:
            with (self.root / 'diagnostic_worker.log').open('ab') as log:
                self.process = subprocess.Popen([sys.executable, '-m', 'th06ai.diagnostic_writer', str(self.root)],
                    cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), bufsize=65536,
                    env=dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1'))
        except OSError as error:
            self.failed = True
            self._error('diagnostic startup: '+str(error))
            return
        self.sender = threading.Thread(target=self._send, name='diagnostic-pipe', daemon=True)
        self.receiver = threading.Thread(target=self._receive, name='diagnostic-results', daemon=True)
        self.sender.start(); self.receiver.start()

    def _error(self, message):
        with self.lock:
            self.errors = [*self.errors[-7:], str(message)]

    def submit(self, job, *, critical=False):
        with self.lock:
            if self.closed or self.failed or job[0] == 'trace' and self.pending_traces >= 2:
                self.dropped += 1
                if critical:self.errors = [*self.errors[-7:], 'critical diagnostic record rejected']
                return False
            try:
                self.jobs.put_nowait(job)
            except queue.Full:
                self.dropped += 1
                if critical:self.errors = [*self.errors[-7:], 'critical diagnostic backlog: '+str(job[:2])]
                return False
            if job[0] == 'trace':self.pending_traces += 1
            return True

    def _send(self):
        try:
            while not self.sender_stop.is_set():
                queued = True
                try:
                    job = self.jobs.get_nowait()
                except queue.Empty:
                    try:
                        job = self.battle_view.next_job()
                    except (OSError, ValueError, TypeError, AttributeError, IndexError):
                        job = None
                    if job is not None and not self.jobs.empty():
                        self.battle_view.in_flight = False
                        job = None
                    if job is None:
                        try:job = self.jobs.get(timeout=.05)
                        except queue.Empty:continue
                    else:
                        queued = False
                try:
                    # Only this process creates the pipe and its writer. Never
                    # deserialize pickle from HTTP, model metadata or disk here.
                    pickle.dump(job, self.process.stdin, protocol=5)
                    self.process.stdin.flush()
                finally:
                    if queued:self.jobs.task_done()
                if job[0] == 'close':break
        except (OSError, ValueError, pickle.PickleError) as error:
            self.failed = True
            self._error('diagnostic pipe: '+str(error))
        finally:
            self.process.stdin.close()

    def _receive(self):
        try:
            for line in self.process.stdout:
                message = json.loads(line)
                if message.get('kind') == 'battle_view':
                    self.battle_view.in_flight = False
                    continue
                with self.lock:
                    self.counts.update(message.get('counts', {}))
                    if message.get('kind') == 'trace':self.pending_traces = max(0, self.pending_traces-1)
                if message.get('error'):self._error(message['error'])
        except (OSError, ValueError) as error:
            self._error('diagnostic result: '+str(error))
        finally:
            self.process.stdout.close()
            if not self.closed:
                self.failed = True
                self._error('diagnostic writer exited unexpectedly')

    def status(self):
        with self.lock:
            return {**self.counts, 'queued': self.jobs.qsize(), 'pending_traces': self.pending_traces,
                    'dropped_records': self.dropped, 'errors': list(self.errors)}

    def close(self, timeout=3.):
        if self.close_result is not None:
            return self.close_result
        if self.process is None:
            self.closed = True
            self.close_result = False
            return False
        deadline = time.monotonic()+timeout
        with self.lock:self.closed = True
        try:
            if not self.failed:
                self.jobs.put(('close',), timeout=max(.001, deadline-time.monotonic()))
                self.sender.join(max(0, deadline-time.monotonic()))
                self.process.wait(timeout=max(.001, deadline-time.monotonic()))
            else:
                self.sender_stop.set()
                self.sender.join(max(0, deadline-time.monotonic()))
        except (queue.Full, subprocess.TimeoutExpired):
            self._error('diagnostic writer did not drain before close deadline')
        complete = self.process.poll() == 0 and not self.sender.is_alive()
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=1.)
        self.sender_stop.set()
        self.sender.join(.1)
        self.receiver.join(max(.1, deadline-time.monotonic()))
        self.close_result = complete
        return self.close_result


def writer_main(root):
    # Imports stay in this CPU-only child, not the sender thread.
    from .diagnostics import SessionDiagnostics
    from .log_rows import action_row
    from .files import _json_default
    sink = SessionDiagnostics.__new__(SessionDiagnostics)
    sink.root, sink.written_deaths = Path(root), 0
    counts = {'written_deaths': 0, 'written_actions': 0, 'written_traces': 0}
    while True:
        try:job = pickle.load(sys.stdin.buffer)
        except EOFError:break
        kind, error = job[0], None
        if kind == 'close':break
        if kind == 'battle_view':
            try:
                publish_battle_view(sink.root, *job[1:])
            except (OSError, ValueError, TypeError, IndexError):
                pass  # Best-effort live frame, independent of retained diagnostics.
            print('{"kind":"battle_view"}', flush=True)
            continue
        started = time.perf_counter_ns()
        try:
            if kind == 'death':
                sink._death(*job[1:]); counts['written_deaths'] += 1
            elif kind == 'action':
                sink._csv('actions.csv', job[1]); counts['written_actions'] += 1
            elif kind == 'action_info':
                sink._csv('actions.csv', action_row(job[1], job[2])); counts['written_actions'] += 1
            elif kind == 'episode':sink._csv('episodes.csv', job[1])
            elif kind == 'trace':
                write_npz(sink.root / job[1], job[2], job[3]); counts['written_traces'] += 1
            elif kind == 'event':
                with (sink.root/'sampling_events.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(job[1], ensure_ascii=False, default=_json_default)+'\n')
            else:raise ValueError('Unknown diagnostic job')
        except Exception as exc:error = str(exc)
        counts['last_'+kind+'_ms'] = (time.perf_counter_ns()-started)/1e6
        print(json.dumps({'kind': kind, 'counts': counts, 'error': error}), flush=True)


if __name__ == '__main__':
    writer_main(sys.argv[1])
