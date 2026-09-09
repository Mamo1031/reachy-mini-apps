"""サーバー(REST / SSE / プリフライト / 復旧)の統合テスト。偽デーモンと偽 TTS を注入して実行する。"""

import asyncio
import json
import time

import httpx
import pytest

from app import main
from tests.fake_daemon import FakeDaemon, make_app
from tests.test_audio import FakeTTS


@pytest.fixture
async def env(tmp_path):
    fake = FakeDaemon()  # 初期状態: モーター無効・スリープ姿勢(実機の起動直後と同じ)
    tts = FakeTTS()
    await main.build_state(data_dir=tmp_path, robot_transport=httpx.ASGITransport(app=make_app(fake)), tts=tts, monitor_interval=0.05)
    main.state.settings.motion.idle.enabled = False
    main.state.settings.motion.audio_lead_ms = 30
    main.state.settings.motion.ramp_in_s = 0.1
    main.state.settings.motion.ramp_out_s = 0.1
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://t") as c:
        yield c, fake, tts
    await main.shutdown_state()


async def wait_for(pred, timeout=10.0, what="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"timeout waiting for {what}")


async def wait_preflight(c, timeout=10.0):
    async def ok():
        r = await c.get("/api/state")
        return r.json()["preflight"]

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pf = await ok()
        if pf["synth"]["status"] in ("ok", "fail") and pf["volume"]["status"] in ("ok", "fail"):
            return pf
        await asyncio.sleep(0.1)
    raise AssertionError(f"preflight did not finish: {await ok()}")


async def test_preflight_wakes_robot_silently_and_uploads_audio(env):
    c, fake, tts = env
    pf = await wait_preflight(c)
    assert all(pf[s]["status"] == "ok" for s in main.PREFLIGHT_STEPS), pf
    assert fake.motor_mode == "enabled"
    assert abs(fake.head["pitch"]) < 1e-6 and fake.antennas == [-0.1745, 0.1745]  # ニュートラル
    assert "wake_up.wav" not in [n for _, n in fake.played]  # 無音ウェイク
    n_texts = len(set(main.state.performer.texts_for_prewarm(include_child=False)))  # 「それでははじめ！」は 2 箇所で同じ音声
    assert len(fake.sounds) == n_texts == len(tts.calls)
    assert fake.tracking_enabled and fake.tracking_weight == 1.0
    st = (await c.get("/api/state")).json()
    assert st["connection"]["state"] == "connected" and st["volume"] == 80
    assert set(st["leaves"]) >= {"A1", "A2", "B1", "C10", "BC4"}


async def test_session_play_stop_and_log(env, tmp_path):
    c, fake, tts = env
    await wait_preflight(c)
    r = await c.post("/api/session/start", json={"child_name": "はな", "suffix": "ちゃん", "order": "robot_first", "condition": "empathy"})
    assert r.status_code == 200, r.text
    sess = r.json()
    assert sess["active"] and sess["intro_sequence"][0] == "A1"
    assert any("はなちゃんっていうんだね" in t for t, _ in tts.calls)
    r = await c.post("/api/play", json={"id": "A2"})
    assert r.status_code == 200 and r.json()["accepted"]
    st = (await c.get("/api/state")).json()
    assert st["performer"]["status"] == "playing" and st["performer"]["current"]["text"].startswith("はなちゃん")
    assert st["session"]["intro_done"] == ["A2"]
    await asyncio.sleep(0.2)
    r = await c.post("/api/stop")
    assert r.status_code == 200 and r.json()["status"] == "idle"
    r = await c.post("/api/session/phase", json={"phase": "main"})
    assert r.json()["main_remaining_s"] is not None
    r = await c.post("/api/play", json={"id": "B1"})
    assert r.json()["accepted"]
    r = await c.post("/api/play", json={"id": "B1"})
    assert r.json() == {"accepted": False, "reason": "debounce"}
    r = await c.post("/api/session/end")
    assert not r.json()["active"]
    logs = list((tmp_path / "logs").glob("*_はなちゃん.csv"))
    assert len(logs) == 1
    body = logs[0].read_text(encoding="utf-8-sig")
    assert "A2" in body and "B1" in body and "stop" in body


async def test_errors_are_json(env):
    c, fake, tts = env
    await wait_preflight(c)
    r = await c.post("/api/play", json={"id": "ZZ"})
    assert r.status_code == 404 and "台本にありません" in r.json()["error"]
    r = await c.post("/api/session/start", json={"child_name": "Hana", "suffix": "ちゃん", "order": "robot_first", "condition": "empathy"})
    assert r.status_code == 400 and "ひらがな" in r.json()["error"]
    r = await c.put("/api/phrases", json={"intro": [], "empathy": [{"id": "B1", "text": "x", "gesture": "nope"}], "logical": [], "backchannel": []})
    assert r.status_code == 400 and "nope" in r.json()["error"]
    r = await c.post("/api/session/phase", json={"phase": "main"})
    assert r.status_code == 400


async def test_settings_change_applies_and_renames_robot(env):
    c, fake, tts = env
    await wait_preflight(c)
    s = (await c.get("/api/settings")).json()
    s["names"]["robot"] = "ポチ"
    s["motion"]["tracking_enabled"] = False
    r = await c.put("/api/settings", json=s)
    assert r.status_code == 200 and r.json()["ok"]
    assert fake.tracking_enabled is False
    assert main.state.settings.names.robot == "ポチ"
    assert json.loads((main.SETTINGS_PATH).read_text(encoding="utf-8"))["names"]["robot"] == "ポチ"
    # 名前が変わったので裏で全音声を作り直す → 終わるのを待つ
    assert main.state.prewarm_task is not None
    await asyncio.wait_for(main.state.prewarm_task, timeout=20)
    assert any(t.startswith("初めまして！ わたしはポチだよ") for t, _ in tts.calls)
    before = len(tts.calls)
    r = await c.post("/api/phrases/test", json={"id": "A1"})
    assert r.json()["accepted"]
    assert len(tts.calls) == before  # 作り直し済みなのでボタン時は合成しない
    st = (await c.get("/api/state")).json()
    assert st["performer"]["current"]["text"].startswith("初めまして！ わたしはポチだよ")


async def test_rest_and_wake(env):
    c, fake, tts = env
    await wait_preflight(c)
    r = await c.post("/api/robot/rest")
    assert r.json()["resting"] is True and fake.motor_mode == "disabled"
    r = await c.post("/api/play", json={"id": "BC1"})
    assert r.status_code == 409
    r = await c.post("/api/robot/wake")
    assert r.json()["resting"] is False and fake.motor_mode == "enabled"
    assert abs(fake.head["pitch"]) < 1e-6


async def test_outage_recovery_reuploads_sounds(env):
    c, fake, tts = env
    await wait_preflight(c)
    n = len(fake.sounds)
    fake.sounds.clear()  # デーモン再起動で /tmp が消えた
    fake.fail_until = time.monotonic() + 0.6
    await wait_for(lambda: main.state.monitor.snapshot.state == "disconnected", what="disconnected")
    r = await c.post("/api/play", json={"id": "BC1"})
    assert r.status_code == 503
    await wait_for(lambda: main.state.monitor.snapshot.state == "connected", timeout=15.0, what="reconnected")
    await wait_for(lambda: len(fake.sounds) == n, timeout=10.0, what="reupload")
    assert any(e["type"] == "recovery" for e in main.state.bus.recent)


async def test_dict_word_and_voices(env):
    c, fake, tts = env
    await wait_preflight(c)
    r = await c.post("/api/dict/word", json={"surface": "はな", "pronunciation": "はな", "accent_type": 1})
    assert r.status_code == 200
    entries = {e["surface"]: e for e in r.json()["entries"]}
    assert entries["はな"]["pronunciation"] == "ハナ" and entries["はな"]["accent_type"] == 1
    r = await c.get("/api/voices")
    assert r.status_code == 200 and r.json()[0]["id"] == "1"


async def test_sse_stream_snapshot_events_and_heartbeat(env):
    # httpx の ASGITransport は本文を全て読み切るまで返さないため、SSE ジェネレータを直接検証する
    from app.events import sse_stream

    c, fake, tts = env
    gen = sse_stream(main.state.bus, main.snapshot(), main.heartbeat, interval=0.2)
    first = await asyncio.wait_for(gen.__anext__(), timeout=2.0)
    head, data = first.strip().split("\n", 1)
    assert head == "event: snapshot"
    payload = json.loads(data[len("data: ") :])
    assert payload["type"] == "snapshot" and "connection" in payload and "leaves" in payload
    main.state.bus.toast("info", "hello")
    seen: list[str] = []
    for _ in range(20):  # 復旧中の connection イベント等が混ざるので、toast と heartbeat が来るまで読む
        ev = await asyncio.wait_for(gen.__anext__(), timeout=2.0)
        seen.append(ev.split("\n", 1)[0])
        if "event: heartbeat" in seen and "event: toast" in seen:
            break
    assert "event: toast" in seen and "event: heartbeat" in seen, seen
    await gen.aclose()
    assert main.state.bus.subscriber_count == 0

    r = await c.get("/api/state")
    assert r.status_code == 200 and r.json()["app_version"] == main.APP_VERSION


async def test_tts_preview_browser_returns_wav(env):
    c, fake, tts = env
    await wait_preflight(c)
    r = await c.post("/api/tts/preview", json={"text": "わたしは{robot}だよ", "target": "browser"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content.startswith(b"RIFF")
    assert tts.calls[-1][0] == "わたしはドラちゃんだよ"
