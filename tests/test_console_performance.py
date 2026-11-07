from fastapi.testclient import TestClient

from th06ai import console_performance
from th06ai.config import Config
from th06ai.console_performance import CpuMonitor, CpuTimes
from th06ai.web_console import create_app


def test_cpu_samples_are_shared_and_report_the_busiest_logical_processor():
    now = [0.0]
    readings = iter((CpuTimes(2, (100, 400), ((50, 200), (50, 200))),
                     CpuTimes(2, (150, 600), ((100, 300), (50, 300)))))
    monitor = CpuMonitor(read=lambda: next(readings), clock=lambda: now[0])
    first = monitor.snapshot()
    assert first['status'] == 'warming' and first['usage_percent'] is None
    now[0] = 0.5
    assert monitor.snapshot() is first
    now[0] = 1.0
    live = monitor.snapshot()
    assert live == {'status': 'live', 'usage_percent': 75.0, 'busiest_core_percent': 100.0,
                    'logical_processors': 2, 'history': (75.0,)}
    assert monitor.snapshot() is live


def test_cpu_gap_rebaselines_and_failed_read_does_not_reuse_old_usage():
    now = [0.0]
    readings = iter((CpuTimes(4, (0, 0)), CpuTimes(4, (40, 80)), CpuTimes(4, (60, 160)),
                     CpuTimes(4, (140, 240)), None))
    monitor = CpuMonitor(read=lambda: next(readings), clock=lambda: now[0])
    assert monitor.snapshot()['status'] == 'warming'
    now[0] = 1.0
    assert monitor.snapshot()['usage_percent'] == 50.0
    now[0] = 7.0
    gap = monitor.snapshot()
    assert gap['status'] == 'warming' and gap['history'] == ()
    now[0] = 8.0
    idle = monitor.snapshot()
    assert idle['usage_percent'] == 0.0 and idle['busiest_core_percent'] is None
    now[0] = 9.0
    failed = monitor.snapshot()
    assert failed['status'] == 'unavailable' and failed['usage_percent'] is None and failed['history'] == ()


def test_performance_route_reads_only_on_demand_without_a_session(tmp_path, monkeypatch):
    calls = []

    def sample():
        calls.append(True)
        return CpuTimes(4, (10, 80))

    monkeypatch.setattr(console_performance, 'read_cpu_times', sample)
    app = create_app(Config(game_exe='never-opened'), root=tmp_path, preview=True)
    with TestClient(app) as client:
        assert client.get('/api/health').status_code == 200
        assert client.get('/api/bootstrap?state=false').status_code == 200
        assert calls == []
        response = client.get('/api/performance')
        assert response.status_code == 200 and response.headers['cache-control'] == 'no-store'
        assert response.json() == {'cpu': {'status': 'warming', 'usage_percent': None, 'busiest_core_percent': None,
                                           'logical_processors': 4, 'history': []}}
    assert calls == [True] and not (tmp_path / 'runs').exists()
