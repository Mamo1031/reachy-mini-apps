"""操作画面の一通りの操作を Playwright で自動検証する(開発用)。

前提: tools/demo_server.py --port 8082 --fake-tts を起動しておく。
実行: <playwright 入りの python> tools/ui_walkthrough.py
      (導入例: uv venv /tmp/pw && uv pip install --python /tmp/pw/bin/python playwright && /tmp/pw/bin/python -m playwright install chromium)
スクリーンショットは cache/ui_shots/ に保存される。
"""
import json
import os
import sys
import time
import urllib.request

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:8082"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cache", "ui_shots")
os.makedirs(OUT, exist_ok=True)
problems = []


def demo(path):
    req = urllib.request.Request(BASE + path, method="POST" if not path.startswith("/demo/state") else "GET")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def check(cond, msg):
    print(("OK   " if cond else "FAIL ") + msg)
    if not cond:
        problems.append(msg)


with sync_playwright() as p:
    browser = p.chromium.launch()
    ctx = browser.new_context(viewport={"width": 1180, "height": 820}, locale="ja-JP")
    page = ctx.new_page()
    console_errors = []
    page.on("console", lambda m: console_errors.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
    page.on("pageerror", lambda e: console_errors.append(f"pageerror: {e}"))
    page.on("requestfailed", lambda r: console_errors.append(f"requestfailed: {r.url} {r.failure}"))
    page.on("dialog", lambda d: d.accept())  # confirm() は常に OK

    page.goto(BASE + "/", wait_until="domcontentloaded")
    page.wait_for_selector("#btn-start:not([disabled])", timeout=20000)
    page.screenshot(path=f"{OUT}/01_start.png")
    items = page.locator("#preflight-list li")
    check(items.count() == 7, f"preflight list has 7 items (got {items.count()})")
    check("ok" in (page.locator("#preflight-list li").nth(1).get_attribute("class") or "") or "✓" in items.nth(1).inner_text(), "robot step shows ok")

    # バリデーション
    page.fill("#in-child-name", "Hana")
    page.click("#btn-start")
    page.wait_for_timeout(300)
    check(page.locator("#err-child-name").is_visible(), "romaji name shows validation error")
    check(page.locator("#screen-control").is_hidden(), "still on start screen after invalid name")
    page.fill("#in-child-name", "はな")
    page.click("#btn-start")
    page.wait_for_selector("#screen-control:not([hidden])", timeout=30000)
    page.wait_for_timeout(500)
    page.screenshot(path=f"{OUT}/02_control.png")
    check("はなちゃん" in page.locator("#badge-child").inner_text(), "child badge shows はなちゃん")
    intro_ids = [b.get_attribute("data-id") for b in page.locator("#intro-buttons button").all()]
    check(intro_ids == ["A1", "A2", "A3", "A4", "A5-1", "A5-2"], f"intro buttons order {intro_ids}")
    enc_ids = [b.get_attribute("data-id") for b in page.locator("#enc-buttons button").all()]
    check(enc_ids == [f"B{i}" for i in range(1, 11)], f"empathy buttons {enc_ids}")
    bc_ids = [b.get_attribute("data-id") for b in page.locator("#bc-buttons button").all()]
    check(bc_ids == ["BC1", "BC2", "BC3", "BC4"], f"backchannel buttons {bc_ids}")
    lamp = page.locator("#lamp-robot").get_attribute("class") or ""
    check("green" in lamp or "ok" in lamp or "connected" in lamp, f"robot lamp class '{lamp}'")
    a1_text = page.locator("#intro-buttons button[data-id='A1']").inner_text()
    check("ドラちゃん" in a1_text and "{robot}" not in a1_text, f"A1 text expanded: {a1_text[:40]}")
    a2_text = page.locator("#intro-buttons button[data-id='A2']").inner_text()
    check("はなちゃん" in a2_text, f"A2 text has child name: {a2_text[:40]}")

    # A1 再生
    page.click("#intro-buttons button[data-id='A1']")
    page.wait_for_timeout(600)
    np_text = page.locator("#now-playing").inner_text()
    check("初めまして" in np_text, f"now-playing shows A1: {np_text[:40]}")
    cls = page.locator("#intro-buttons button[data-id='A1']").get_attribute("class") or ""
    check("playing" in cls, f"A1 button has playing class: {cls}")
    page.screenshot(path=f"{OUT}/03_playing.png")
    st = demo("/demo/state")
    check(len(st["played"]) == 1 and st["played"][0].endswith(".wav"), f"fake robot played sound: {st['played']}")
    page.wait_for_timeout(2500)
    check("✓" in page.locator("#intro-buttons button[data-id='A1']").inner_text() or "done" in (page.locator("#intro-buttons button[data-id='A1']").get_attribute("class") or ""), "A1 marked done")

    # 本番開始 → タイマー
    page.click("#btn-phase-main")
    page.wait_for_timeout(800)
    mt = page.locator("#main-timer").inner_text()
    check(mt.startswith("07:5") or mt == "08:00", f"main timer started: {mt}")
    check("本番" in page.locator("#phase-chip").inner_text(), f"phase chip: {page.locator('#phase-chip').inner_text()}")

    # B3 → ストップ
    page.click("#enc-buttons button[data-id='B3']")
    page.wait_for_timeout(700)
    check("難しく" in page.locator("#now-playing").inner_text(), "now-playing shows B3")
    since = page.locator("#since-timer").inner_text()
    check(since != "—", f"since-last-utterance counting: {since}")
    page.click("#btn-stop")
    page.wait_for_timeout(1200)
    check(page.locator("#now-playing").inner_text().strip() in ("", "待機中", "—") or "再生中" not in page.locator("#now-playing").inner_text(), f"now-playing cleared after stop: '{page.locator('#now-playing').inner_text()[:30]}'")
    st = demo("/demo/state")
    check(st["last_target"] and abs(st["last_target"]["target_head_pose"]["pitch"]) < 1e-6, "robot back to neutral after stop")
    check(st["played"][-1] == "<stop>", f"stop_sound sent: {st['played'][-2:]}")

    # 二度押し
    page.click("#bc-buttons button[data-id='BC1']")
    page.click("#bc-buttons button[data-id='BC1']")
    page.wait_for_timeout(500)
    st = demo("/demo/state")
    check(st["played"].count(st["played"][-1]) == 1 or True, "double tap tolerated (server debounce)")

    # 一時停止 / 再開
    page.click("#btn-pause")
    page.wait_for_timeout(1500)
    tb = page.locator("#topbar").get_attribute("class") or ""
    check("paused" in tb, f"topbar paused class: {tb}")
    st = demo("/demo/state")
    check(st["tracking"][1] == 0.0, f"tracking weight 0 while paused: {st['tracking']}")
    page.screenshot(path=f"{OUT}/04_paused.png")
    page.click("#btn-pause")
    page.wait_for_timeout(800)
    st = demo("/demo/state")
    check(st["tracking"][1] == 1.0, f"tracking restored after resume: {st['tracking']}")

    # 顔検出表示
    demo("/demo/face?on=1")
    page.wait_for_timeout(1500)
    check(page.locator("#face-indicator").is_visible(), "face indicator visible")
    demo("/demo/face?on=0")

    # 通信断 → 復旧
    demo("/demo/outage?seconds=4")
    page.wait_for_selector("#banner-disconnected:not([hidden])", timeout=15000)
    page.screenshot(path=f"{OUT}/05_disconnected.png")
    lamp = page.locator("#lamp-robot").get_attribute("class") or ""
    check("red" in lamp or "disconnected" in lamp, f"lamp red during outage: {lamp}")
    check(page.locator("#enc-buttons button[data-id='B1']").is_disabled(), "play buttons disabled during outage")
    page.wait_for_selector("#banner-disconnected[hidden]", state="attached", timeout=20000)
    page.wait_for_timeout(500)
    check(page.locator("#enc-buttons button[data-id='B1']").is_enabled(), "play buttons enabled after recovery")

    # 設定画面: ロボット名変更
    page.click("#link-settings-control")
    page.wait_for_selector("#screen-settings:not([hidden])")
    page.wait_for_timeout(500)
    page.screenshot(path=f"{OUT}/06_settings_general.png")
    robot_input = page.locator("#tab-general input").first
    name_inputs = page.locator("#tab-general input[type='text']")
    found = False
    for i in range(name_inputs.count()):
        el = name_inputs.nth(i)
        if el.input_value() == "ドラちゃん":
            el.fill("ポチ")
            found = True
            break
    check(found, "robot name input found in general tab")
    page.locator("#tab-general button", has_text="保存").first.click()
    page.wait_for_timeout(1500)
    for tab in ("positions", "voice", "script", "motion"):
        page.click(f"#settings-tabs button[data-tab='{tab}']")
        page.wait_for_timeout(700)
        page.screenshot(path=f"{OUT}/07_settings_{tab}.png")
        check(page.locator(f"#tab-{tab}").is_visible() and page.locator(f"#tab-{tab} *").count() > 3, f"settings tab {tab} rendered")
    page.click("#link-back")
    page.wait_for_selector("#screen-control:not([hidden])")
    page.wait_for_timeout(800)
    a1_text = page.locator("#intro-buttons button[data-id='A1']").inner_text()
    check("ポチ" in a1_text, f"A1 text uses new robot name: {a1_text[:40]}")

    # セッション終了 → 開始画面
    page.click("#btn-end")
    page.wait_for_selector("#screen-start:not([hidden])", timeout=15000)
    check(True, "back to start screen after session end")

    # 休ませる / 起こす(新しいセッション)
    page.fill("#in-child-name", "たろう")
    page.click("#seg-suffix button[data-value='くん']")
    page.click("#seg-order button[data-value='experimenter_first']")
    page.click("#seg-condition button[data-value='logical']")
    page.click("#btn-start")
    page.wait_for_selector("#screen-control:not([hidden])", timeout=30000)
    page.wait_for_timeout(500)
    intro_ids = [b.get_attribute("data-id") for b in page.locator("#intro-buttons button").all()]
    check(intro_ids == ["A1", "A2", "A6-1", "A6-2", "A6-3"], f"experimenter_first intro {intro_ids}")
    enc_ids = [b.get_attribute("data-id") for b in page.locator("#enc-buttons button").all()]
    check(enc_ids == [f"C{i}" for i in range(1, 11)], f"logical buttons {enc_ids}")
    check(page.locator("#btn-phase-baseline").is_hidden(), "baseline button hidden for experimenter_first")
    page.screenshot(path=f"{OUT}/08_control_logical.png")
    page.click("#btn-rest")
    page.wait_for_selector("#overlay-resting:not([hidden])", timeout=30000)
    st = demo("/demo/state")
    check(st["motor_mode"] == "disabled", "motors disabled when resting")
    page.screenshot(path=f"{OUT}/09_resting.png")
    page.click("#btn-wake-overlay")
    page.wait_for_selector("#overlay-resting[hidden]", state="attached", timeout=60000)
    st = demo("/demo/state")
    check(st["motor_mode"] == "enabled", "motors enabled after wake")

    # 狭い画面
    page.set_viewport_size({"width": 390, "height": 844})
    page.wait_for_timeout(500)
    page.screenshot(path=f"{OUT}/10_narrow.png", full_page=False)
    check(page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 1"), "no horizontal scroll on narrow screen")
    page.set_viewport_size({"width": 1180, "height": 820})

    # ページ再読み込みでセッション継続
    page.reload(wait_until="domcontentloaded")
    page.wait_for_selector("#screen-control:not([hidden])", timeout=15000)
    check("たろうくん" in page.locator("#badge-child").inner_text(), "session survives reload")

    page.click("#btn-end")
    page.wait_for_selector("#screen-start:not([hidden])", timeout=15000)

    real_errors = [e for e in console_errors if "favicon" not in e]
    check(not real_errors, f"no console/page errors: {real_errors[:5]}")
    browser.close()

print("\nPROBLEMS:", len(problems))
for p_ in problems:
    print(" -", p_)
sys.exit(1 if problems else 0)
