"""コードレビューで見つかった不具合の回帰テスト(偽デーモン + 偽 TTS、サーバー経由)。"""

import asyncio
import json
import time

import httpx
import pytest

from app import main
from app.config import expand
from tests.fake_daemon import FakeDaemon, make_app
from tests.test_api import wait_for, wait_preflight
from tests.test_audio import FakeTTS


class SlowTTS(FakeTTS):
    """合成に時間がかかる TTS(「準備中」の挙動を試す)。"""

    def __init__(self, delay: float = 1.0):
        super().__init__()
        self.delay = delay

    async def synthesize(self, text, voice_id, params):
        await asyncio.sleep(self.delay)
        return await super().synthesize(text, voice_id, params)


async def _boot(tmp_path, tts=None):
    fake = FakeDaemon()
    tts = tts or FakeTTS()
    await main.build_state(data_dir=tmp_path, robot_transport=httpx.ASGITransport(app=make_app(fake)), tts=tts, monitor_interval=0.05)
    main.state.settings.motion.idle.enabled = False
    main.state.settings.motion.audio_lead_ms = 30
    main.state.settings.motion.ramp_in_s = 0.1
    main.state.settings.motion.ramp_out_s = 0.1
    return fake, tts


@pytest.fixture
async def env(tmp_path):
    fake, tts = await _boot(tmp_path)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://t") as c:
        yield c, fake, tts
    await main.shutdown_state()


async def start_session(c, name="はな"):
    r = await c.post("/api/session/start", json={"child_name": name, "suffix": "ちゃん", "order": "robot_first", "condition": "empathy"})
    assert r.status_code == 200, r.text
    return r.json()


def test_expand_never_raises_on_broken_braces():
    assert expand("うん}", robot="x") == "うん}"
    assert expand("{robot", robot="x") == "{robot"
    assert expand("{0} {child.upper}", child="はな") == "{0} {child.upper}"
    assert expand("{robot}と{child}", robot="ドラちゃん", child="はなちゃん") == "ドラちゃんとはなちゃん"


async def test_pause_does_not_leak_into_next_session(env):
    c, fake, tts = env
    await wait_preflight(c)
    await start_session(c, "はな")
    await c.post("/api/pause")
    assert fake.tracking_weight == 0.0
    await c.post("/api/session/end")
    await start_session(c, "たろう")
    st = (await c.get("/api/state")).json()
    assert st["performer"]["paused"] is False
    assert fake.tracking_enabled and fake.tracking_weight == 1.0


async def test_stop_during_audio_lead_window_cancels_pending_gesture(env):
    c, fake, tts = env
    await wait_preflight(c)
    await start_session(c)
    main.state.settings.motion.audio_lead_ms = 400  # 音声先行の待ちを長くする
    r = await c.post("/api/play", json={"id": "B3"})
    assert r.json()["accepted"]
    await asyncio.sleep(0.1)  # まだ lead 待ち(player は動いていない)
    assert not main.state.player.is_playing
    n_before = len(fake.targets)
    await c.post("/api/stop")
    await asyncio.sleep(0.8)
    # 取り残されたジェスチャーが後から動き出していない
    later = [t for t in fake.targets[n_before:] if t[1]["target_head_pose"] and abs(t[1]["target_head_pose"]["pitch"]) > 0.05]
    assert later == []
    assert main.state.performer.status == "idle" and main.state.performer._play_task is None


async def test_stop_is_not_blocked_by_synthesis(tmp_path):
    fake, tts = await _boot(tmp_path, SlowTTS(delay=1.5))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://t") as c:
        try:
            await wait_preflight(c, timeout=90)
            await start_session(c)
            main.state.settings.names.robot = "ポチ"  # 未合成のテキストを作る
            play = asyncio.create_task(c.post("/api/play", json={"id": "A1"}))
            await asyncio.sleep(0.3)
            assert main.state.performer.status == "preparing"
            t0 = time.monotonic()
            r = await c.post("/api/stop")
            assert r.status_code == 200 and time.monotonic() - t0 < 0.5  # 合成を待たずに返る
            res = (await play).json()
            assert res == {"accepted": False, "reason": "superseded"}
            assert main.state.performer.status == "idle"
        finally:
            await main.shutdown_state()


async def test_new_button_during_preparing_supersedes_old(tmp_path):
    fake, tts = await _boot(tmp_path, SlowTTS(delay=0.8))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://t") as c:
        try:
            await wait_preflight(c, timeout=90)
            await start_session(c)
            main.state.settings.names.robot = "ポチ"
            p1 = asyncio.create_task(c.post("/api/play", json={"id": "A1"}))
            await asyncio.sleep(0.2)
            p2 = asyncio.create_task(c.post("/api/play", json={"id": "A3"}))
            r1, r2 = (await p1).json(), (await p2).json()
            assert r1["accepted"] is False and r1["reason"] == "superseded"
            assert r2["accepted"] is True
            assert main.state.performer.current["id"] == "A3"
            assert [n for _, n in fake.played].count("<stop>") == 0  # 何も鳴っていないのに stop_sound しない
        finally:
            await main.shutdown_state()


async def test_tracking_off_during_gesture_stays_off(env):
    c, fake, tts = env
    await wait_preflight(c)
    await start_session(c)
    await c.post("/api/play", json={"id": "B3"})  # 頭を動かす(downcast1)
    await asyncio.sleep(0.3)
    assert main.state.player.is_playing and fake.tracking_weight == 0.0
    r = await c.post("/api/tracking", json={"enabled": False})
    assert r.status_code == 200
    await c.post("/api/stop")
    await asyncio.sleep(0.3)
    assert fake.tracking_enabled is False  # ジェスチャー終了時に 1.0 へ戻されない
    assert main.state.settings.motion.tracking_enabled is False


async def test_idle_does_not_run_while_resting_and_wake_restores(env):
    c, fake, tts = env
    await wait_preflight(c)
    main.state.settings.motion.idle.enabled = True
    main.state.settings.motion.idle.interval_s = 2.0
    main.state.settings.motion.idle.jitter_s = 0.0
    main.state.performer.last_idle_at = time.monotonic() - 10
    r = await c.post("/api/robot/rest")
    assert r.json()["resting"] is True and fake.motor_mode == "disabled"
    n = len(fake.targets)
    await asyncio.sleep(2.5)
    assert len(fake.targets) == n  # 休止中はアイドルが動かない
    r = await c.post("/api/play", json={"id": "BC1"})
    assert r.status_code == 409
    # 休止中に通信断→復帰しても勝手に起き上がらない
    fake.fail_until = time.monotonic() + 0.6
    await wait_for(lambda: main.state.monitor.snapshot.state == "disconnected", what="disconnected")
    await wait_for(lambda: main.state.monitor.snapshot.state == "connected", timeout=5, what="reconnected")
    assert main.state.resting is True and fake.motor_mode == "disabled"
    r = await c.post("/api/robot/wake")
    assert r.json()["resting"] is False and fake.motor_mode == "enabled"


async def test_rest_refuses_play_immediately(env):
    c, fake, tts = env
    await wait_preflight(c)
    task = asyncio.create_task(c.post("/api/robot/rest"))
    await asyncio.sleep(0.1)
    r = await c.post("/api/play", json={"id": "BC1"})
    assert r.status_code == 409  # スリープ中の goto を殺さない
    await task
    assert fake.motor_mode == "disabled"


async def test_connection_events_written_to_session_log(env, tmp_path):
    c, fake, tts = env
    await wait_preflight(c)
    sess = await start_session(c)
    fake.fail_until = time.monotonic() + 0.6
    await wait_for(lambda: main.state.monitor.snapshot.state == "disconnected", what="disconnected")
    await wait_for(lambda: main.state.monitor.snapshot.state == "connected", timeout=5, what="reconnected")
    await c.post("/api/session/end")
    csv_text = (tmp_path / "logs" / (sess["log_stem"].rsplit("/", 1)[-1] + ".csv")).read_text(encoding="utf-8-sig")
    assert "connection" in csv_text and "disconnected" in csv_text and "recovery" in csv_text


async def test_phrase_test_does_not_pollute_session(env):
    c, fake, tts = env
    await wait_preflight(c)
    await start_session(c)
    await c.post("/api/session/phase", json={"phase": "main"})
    since0 = (await c.get("/api/state")).json()["session"]["since_last_utterance_s"]
    assert since0 is not None  # 本番開始で目安タイマーが動き出す
    await asyncio.sleep(0.6)
    r = await c.post("/api/phrases/test", json={"id": "B1"})
    assert r.json()["accepted"]
    st = (await c.get("/api/state")).json()
    assert st["session"]["since_last_utterance_s"] >= 0.5  # テスト再生ではリセットされない
    assert st["session"]["intro_done"] == []
    kinds = [e["kind"] for e in main.state.bus.recent if e["type"] == "log"]
    assert "button" not in kinds and "playing" not in kinds and "system" in kinds


async def test_gesture_test_after_phrase_resets_status(env):
    c, fake, tts = env
    await wait_preflight(c)
    await start_session(c)
    await c.post("/api/play", json={"id": "B2"})
    await asyncio.sleep(0.3)
    r = await c.post("/api/gestures/test", json={"name": "nod_small"})
    assert r.status_code == 200
    await asyncio.sleep(1.5)
    st = (await c.get("/api/state")).json()
    assert st["performer"]["status"] == "idle" and st["performer"]["current"] is None


async def test_put_settings_rejects_bad_values_and_keeps_file(env):
    c, fake, tts = env
    await wait_preflight(c)
    s = (await c.get("/api/settings")).json()
    s["robot"]["read_timeout_s"] = 0
    r = await c.put("/api/settings", json=s)
    assert r.status_code == 400 and "入力が正しくありません" in r.json()["error"]
    s = (await c.get("/api/settings")).json()
    s["motion"]["idle"]["gesture_tracking_on"] = "nope"
    r = await c.put("/api/settings", json=s)
    assert r.status_code == 400 and "nope" in r.json()["error"]
    s = (await c.get("/api/settings")).json()
    s["tts"]["backend"] = "clone"
    r = await c.put("/api/settings", json=s)
    assert r.status_code == 400
    assert json.loads(main.SETTINGS_PATH.read_text(encoding="utf-8"))["tts"]["backend"] == "voicevox"


async def test_put_phrases_rejects_empty_and_duplicates_and_keeps_backup(env):
    c, fake, tts = env
    await wait_preflight(c)
    r = await c.put("/api/phrases", json={})
    assert r.status_code == 400 and "空" in r.json()["error"]
    ph = (await c.get("/api/phrases")).json()
    ph["logical"][0]["id"] = "B1"
    r = await c.put("/api/phrases", json=ph)
    assert r.status_code == 400 and "重複" in r.json()["error"]
    ph = (await c.get("/api/phrases")).json()
    ph["backchannel"][0]["text"] = "うんうん"
    r = await c.put("/api/phrases", json=ph)
    assert r.status_code == 200
    assert main.PHRASES_PATH.with_suffix(".json.bak").exists()


async def test_corrupt_cached_wav_is_resynthesized(env):
    c, fake, tts = env
    await wait_preflight(c)
    await start_session(c)
    text = main.state.performer.expand_text("うん")
    p = main.state.audio.path(main.state.audio.key(text))
    p.write_bytes(b"garbage")  # セッション開始後にキャッシュが壊れた
    n = len(tts.calls)
    r = await c.post("/api/play", json={"id": "BC1"})
    assert r.json()["accepted"]
    assert len(tts.calls) == n + 1 and p.read_bytes().startswith(b"RIFF")


async def test_missing_sound_on_robot_is_reuploaded_on_play(env):
    c, fake, tts = env
    await wait_preflight(c)
    await start_session(c)
    fake.sounds.clear()  # 監視が気づかないうちに消えた
    r = await c.post("/api/play", json={"id": "BC2"})
    assert r.json()["accepted"]
    assert any(n.endswith(".wav") for _, n in fake.played)
    assert len(fake.sounds) == 1


async def test_unexpected_exception_is_json_and_toasted(env):
    c, fake, tts = env
    await wait_preflight(c)

    async def boom():
        raise RuntimeError("kaboom")

    main.state.performer.play_phrase = lambda *a, **k: boom()  # type: ignore[assignment]
    # Starlette は Exception ハンドラで応答した後に例外を再送出する(ASGITransport はそれを伝えるので抑止する)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app, raise_app_exceptions=False), base_url="http://t") as c2:
        r = await c2.post("/api/play", json={"id": "BC1"})
    assert r.status_code == 500 and "内部エラー" in r.json()["error"]
    assert any(e["type"] == "toast" and "kaboom" in e["message"] for e in main.state.bus.recent)


async def test_monitor_external_motor_disable_triggers_recovery(env):
    c, fake, tts = env
    await wait_preflight(c)
    fake.motor_mode = "disabled"  # ダッシュボード等から切られた
    await wait_for(lambda: fake.motor_mode == "enabled" and main.state.monitor.is_connected(), timeout=8, what="re-enabled")
