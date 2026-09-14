import pytest

from app.config import Settings
from app.events import EventBus
from app.power import PowerError, PowerMonitor, fmt_uptime, parse_output


def test_parse_output_and_uptime_format():
    assert parse_output("4352.11 13327.12\nthrottled=0x50005\n") == (4352.11, 0x50005)
    assert parse_output("throttled=0x0\n12.5 30.0\n") == (12.5, 0)
    with pytest.raises(ValueError):
        parse_output("garbage\n")
    with pytest.raises(ValueError):
        parse_output("4352.11 13327.12\n")  # フラグなし
    assert fmt_uptime(4352) == "1:12" and fmt_uptime(59) == "0:00" and fmt_uptime(None) == "-"


def _monitor(outputs, rows, bus=None):
    settings = Settings()
    settings.robot.base_url = "http://192.168.100.2:8000"
    seen = []

    async def runner(host, user, key):
        seen.append((host, user, key))
        v = outputs.pop(0)
        if isinstance(v, Exception):
            raise v
        return v

    pm = PowerMonitor(lambda: settings, bus or EventBus(), row=lambda kind, **f: rows.append((kind, f)), runner=runner)
    return pm, settings, seen


async def test_transitions_rows_and_toast():
    rows, bus = [], EventBus()
    outputs = ["100.0 200.0\nthrottled=0x0\n"]
    pm, settings, seen = _monitor(outputs, rows, bus)
    st = await pm.poll_once()
    assert seen[0] == ("192.168.100.2", "pollen", "~/.ssh/reachy_mini_ed25519")
    assert st.available and st.uptime_s == 100.0 and not st.undervoltage_now and st.error is None
    assert rows[-1][0] == "system" and rows[-1][1]["detail"] == "robot uptime 0:01"
    # 低電圧の立ち上がり: トースト + CSV 行
    outputs.append("130.0 260.0\nthrottled=0x50001\n")
    st = await pm.poll_once()
    assert st.undervoltage_now and st.undervoltage_occurred
    assert rows[-1] == ("power", {"result": "undervoltage", "detail": "uptime 0:02"})
    toasts = [e for e in bus.recent if e.get("type") == "toast"]
    assert toasts and "充電" in toasts[-1]["message"]
    # 収まっても「起動後にあった」は残る。同じ状態では行を増やさない
    n = len(rows)
    outputs.append("160.0 320.0\nthrottled=0x50000\n")
    st = await pm.poll_once()
    assert not st.undervoltage_now and st.undervoltage_occurred and len(rows) == n
    # ロボットが消えた: 最後の稼働時間を残す
    outputs.append(PowerError("ロボットに ssh で届きません"))
    st = await pm.poll_once()
    assert not st.available and st.error == "ロボットに ssh で届きません" and st.undervoltage_occurred
    assert rows[-1][0] == "power" and rows[-1][1]["result"] == "lost" and "0:02" in rows[-1][1]["detail"]
    # 再起動して戻ってきた
    outputs.append("5.0 10.0\nthrottled=0x0\n")
    st = await pm.poll_once()
    assert st.available and st.uptime_s == 5.0 and not st.undervoltage_occurred
    assert rows[-1] == ("system", {"detail": "robot uptime 0:00"})
    # 稼働中に再起動(uptime が 5 秒超戻った)
    outputs.append("300.0 600.0\nthrottled=0x0\n")
    await pm.poll_once()
    outputs.append("3.0 6.0\nthrottled=0x0\n")
    await pm.poll_once()
    assert rows[-1][1]["detail"].startswith("robot rebooted")


async def test_disabled_and_bad_output():
    rows = []
    pm, settings, _ = _monitor(["nonsense\n"], rows)
    st = await pm.poll_once()
    assert not st.available and "想定外" in (st.error or "") and rows == []
    settings.power.enabled = False
    st = await pm.poll_once()
    assert not st.available and st.error == "無効"


async def test_events_are_published():
    events = []
    bus = EventBus()
    orig = bus.publish

    def capture(t, **f):
        events.append((t, f))
        orig(t, **f)

    bus.publish = capture  # type: ignore[method-assign]
    rows = []
    pm, _, _ = _monitor(["10.0 20.0\nthrottled=0x0\n"], rows, bus)
    await pm.poll_once()
    assert events and events[-1][0] == "power" and events[-1][1]["available"] is True and events[-1][1]["uptime_s"] == 10.0
