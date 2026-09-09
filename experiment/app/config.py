"""設定(settings.json)・台本(phrases.json)・ジェスチャー(gestures.json)のモデルと読み書き。

- 欠損キーはデフォルトで補完し、未知キーは無視する(将来の項目追加に強い)。
- 保存は一時ファイル→ os.replace の原子的置換。
- 壊れた JSON は `.broken-<時刻>` に退避してデフォルトで起動する(実験当日に起動不能にならないため)。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Literal, TypeVar

from pydantic import BaseModel, ConfigDict, Field, RootModel

DATA_DIR = Path(__file__).resolve().parents[1]
SETTINGS_PATH = DATA_DIR / "settings.json"
PHRASES_PATH = DATA_DIR / "phrases.json"
GESTURES_PATH = DATA_DIR / "gestures.json"
MOVES_DIR = DATA_DIR / "moves"
CACHE_DIR = DATA_DIR / "cache"
LOGS_DIR = DATA_DIR / "logs"

OrderCondition = Literal["robot_first", "experimenter_first"]
EncouragementCondition = Literal["empathy", "logical"]


class Base(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)


# ---------------------------------------------------------------- settings


class RobotSettings(Base):
    base_url: str = "http://reachy-mini.local:8000"
    connect_timeout_s: float = Field(1.0, gt=0, le=10)
    read_timeout_s: float = Field(3.0, gt=0, le=30)


class NamesSettings(Base):
    robot: str = "ドラちゃん"
    experimenter: str = "はるかお姉さん"


class SessionSettings(Base):
    main_minutes: float = Field(8.0, gt=0, le=120)
    cue_interval_s: float = Field(30.0, gt=0, le=600)


class Position(Base):
    """ロボットから見た対象物の方向(度)。yaw は左が正・右が負(ロボットの座標系)。"""

    label: str = ""
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0


class VoiceParams(Base):
    speed: float = 0.9
    pitch: float = 0.03
    intonation: float = 1.15
    volume: float = 1.0
    post_phoneme: float = 0.3
    pause_scale: float = 1.2


class VoicevoxSettings(Base):
    engine_dir: str = "tools/voicevox/macos-arm64"
    host: str = "127.0.0.1"
    port: int = 50021
    autostart: bool = True


class DictEntry(Base):
    surface: str
    pronunciation: str  # カタカナ
    accent_type: int = 0


class TTSSettings(Base):
    backend: Literal["voicevox"] = "voicevox"  # バックエンドを追加したらここに足す
    voice_id: str = "1"  # VOICEVOX: ずんだもん あまあま
    params: VoiceParams = Field(default_factory=VoiceParams)
    voicevox: VoicevoxSettings = Field(default_factory=VoicevoxSettings)
    user_dict: list[DictEntry] = Field(default_factory=lambda: [DictEntry(surface="はるか", pronunciation="ハルカ", accent_type=0)])


class EnvelopeSettings(Base):
    roll_deg: float = Field(30.0, gt=0, le=60)
    pitch_deg: float = Field(25.0, gt=0, le=60)
    yaw_deg: float = Field(45.0, gt=0, le=90)
    xyz_mm: float = Field(30.0, gt=0, le=60)
    antenna_deg: float = Field(170.0, gt=0, le=180)


class IdleSettings(Base):
    enabled: bool = True
    interval_s: float = Field(10.0, ge=2, le=120)
    jitter_s: float = Field(4.0, ge=0, le=60)
    gesture_tracking_on: str = "antenna_twitch"
    gesture_tracking_off: str = "nod_small"


class MotionSettings(Base):
    tracking_enabled: bool = True
    tracking_weight: float = Field(1.0, ge=0, le=1)
    wobbling_enabled: bool = False
    stream_hz: float = Field(30.0, ge=10, le=60)  # 実機は 1 リクエスト約 30 ms(接続の再利用ができない)ため 30 Hz が実用上限
    ramp_in_s: float = Field(0.3, ge=0, le=2)
    ramp_out_s: float = Field(0.4, ge=0.05, le=2)
    audio_lead_ms: float = Field(150.0, ge=0, le=2000)
    envelope: EnvelopeSettings = Field(default_factory=EnvelopeSettings)
    idle: IdleSettings = Field(default_factory=IdleSettings)


class UISettings(Base):
    debounce_ms: int = Field(500, ge=0, le=5000)


def _default_positions() -> dict[str, Position]:
    return {
        "sample": Position(label="お手本", yaw_deg=-35.0, pitch_deg=10.0),
        "photo": Position(label="写真カード", yaw_deg=35.0, pitch_deg=10.0),  # 仮: お手本の反対側。設定画面で実際の配置に合わせる
        "pieces": Position(label="ピース", yaw_deg=0.0, pitch_deg=18.0),
        "up": Position(label="上", yaw_deg=0.0, pitch_deg=-15.0),
    }


class Settings(Base):
    robot: RobotSettings = Field(default_factory=RobotSettings)
    names: NamesSettings = Field(default_factory=NamesSettings)
    session: SessionSettings = Field(default_factory=SessionSettings)
    positions: dict[str, Position] = Field(default_factory=_default_positions)
    tts: TTSSettings = Field(default_factory=TTSSettings)
    motion: MotionSettings = Field(default_factory=MotionSettings)
    ui: UISettings = Field(default_factory=UISettings)


# ---------------------------------------------------------------- phrases


class PhrasePart(Base):
    id: str
    text: str
    gesture: str = "nod_small"


class IntroStep(Base):
    id: str
    label: str = ""
    text: str | None = None
    gesture: str | None = None
    order: list[OrderCondition] = Field(default_factory=list)  # 空 = 両条件で使う
    parts: list[PhrasePart] = Field(default_factory=list)

    def leaves(self) -> list[PhrasePart]:
        if self.parts:
            return list(self.parts)
        return [PhrasePart(id=self.id, text=self.text or "", gesture=self.gesture or "nod_small")]

    def applies_to(self, order: str) -> bool:
        return not self.order or order in self.order


class Phrase(Base):
    id: str
    trigger: str = ""
    text: str
    gesture: str = "nod_small"


class PhraseLeaf(Base):
    """再生単位(ボタン 1 つ分)。"""

    id: str
    category: Literal["intro", "empathy", "logical", "backchannel"]
    text: str
    gesture: str
    label: str = ""


class Phrases(Base):
    intro: list[IntroStep] = Field(default_factory=list)
    empathy: list[Phrase] = Field(default_factory=list)
    logical: list[Phrase] = Field(default_factory=list)
    backchannel: list[Phrase] = Field(default_factory=list)

    def leaves(self) -> dict[str, PhraseLeaf]:
        out: dict[str, PhraseLeaf] = {}
        for step in self.intro:
            for part in step.leaves():
                out[part.id] = PhraseLeaf(id=part.id, category="intro", text=part.text, gesture=part.gesture, label=step.label)
        for cat in ("empathy", "logical", "backchannel"):
            for ph in getattr(self, cat):
                out[ph.id] = PhraseLeaf(id=ph.id, category=cat, text=ph.text, gesture=ph.gesture, label=ph.trigger)
        return out

    def intro_sequence(self, order: str) -> list[str]:
        ids: list[str] = []
        for step in self.intro:
            if step.applies_to(order):
                ids.extend(p.id for p in step.leaves())
        return ids


# ---------------------------------------------------------------- gestures


class Keyframe(Base):
    """キーフレーム。head は度 / mm(未指定の軸は前値保持)、antennas は物理角度(度)[右, 左]。"""

    head: dict[str, float] | None = None
    antennas: list[float] | None = None
    d: float = 0.3  # このフレームに到達するまでの秒数
    hold: float | None = None  # 指定時は姿勢を保持する秒数(head/antennas は無視)


GestureKind = Literal["keyframes", "recorded", "point", "sequence"]


class GestureDef(Base):
    kind: GestureKind
    description: str = ""
    # keyframes
    frames: list[Keyframe] = Field(default_factory=list)
    # recorded
    file: str | None = None
    start: float | None = None
    end: float | None = None
    speed: float = 1.0
    normalize_xyz: bool = True  # 最初のフレームの並進を 0 に揃える
    # point
    target: str | None = None
    turn_s: float = 0.6
    hold_s: float = 0.4
    nod_deg: float = 8.0
    # sequence
    steps: list[str] = Field(default_factory=list)


class Gestures(RootModel[dict[str, GestureDef]]):
    def get(self, name: str) -> GestureDef | None:
        return self.root.get(name)

    def names(self) -> list[str]:
        return sorted(self.root)


# ---------------------------------------------------------------- defaults


def default_phrases() -> Phrases:
    return Phrases(
        intro=[
            IntroStep(id="A1", label="あいさつ＆自己紹介", text="初めまして！ わたしは{robot}だよ。あなたのお名前はなんていうの？", gesture="greet_wiggle"),
            IntroStep(id="A2", label="名前への反応", text="{child}っていうんだね！ 素敵なお名前だね。今日はいっしょに遊べてとっても嬉しいな！", gesture="happy_lean_clap"),
            IntroStep(id="A3", label="課題への誘いかけ", text="これから一緒にブロックのゲームをしよう！ このピースを使って、お手本のお馬さんを作ってみてね。", gesture="point_sample", order=["robot_first"]),
            IntroStep(id="A4", label="応援・ベースラインスタート", text="形や色をよく見ながら、順番に考えてみよう！ わたし、応援しているからね！", gesture="nod_big_x2", order=["robot_first"]),
            IntroStep(
                id="A5",
                label="本番スタート",
                order=["robot_first"],
                parts=[
                    PhrasePart(id="A5-1", text="できたね、よくがんばったね。それでは次のお手本を見せるから、今度はこれをつくってみよう。大丈夫かな？", gesture="happy_lean"),
                    PhrasePart(id="A5-2", text="それでははじめ！", gesture="enthusiastic"),
                ],
            ),
            IntroStep(
                id="A6",
                label="本番スタート",
                order=["experimenter_first"],
                parts=[
                    PhrasePart(id="A6-1", text="さっきは{experimenter}とブロックで遊んだんだよね。今回は{robot}と一緒にあそびましょう。とっても楽しみ！", gesture="welcoming"),
                    PhrasePart(id="A6-2", text="それでは始めましょう。このお手本をブロックでつくってみよう。大丈夫かな？", gesture="nod_gentle"),
                    PhrasePart(id="A6-3", text="それでははじめ！", gesture="enthusiastic"),
                ],
            ),
        ],
        empathy=[
            Phrase(id="B1", trigger="課題開始直後", text="頑張っているね。応援してるよ！", gesture="lean_in_nod"),
            Phrase(id="B2", trigger="停滞時・悩み", text="考えているんだね。ゆっくりで大丈夫だよ。", gesture="calm_antennas"),
            Phrase(id="B3", trigger="苦戦・足止め", text="難しく感じることもあるよね。", gesture="head_lower"),
            Phrase(id="B4", trigger="失敗時・崩れた時", text="うまくいかなくて悔しいよね。", gesture="tilt_sympathy"),
            Phrase(id="B5", trigger="諦めそうな時・おこり始めたら", text="そういう気持ちになるよね。", gesture="nod_slow_deep"),
            Phrase(id="B6", trigger="課題進行時", text="だいぶできたね！いい感じに進んでいるね。", gesture="banzai_small"),
            Phrase(id="B7", trigger="試行錯誤時", text="大丈夫だよ。一緒に続けてみよう。", gesture="beckon"),
            Phrase(id="B8", trigger="失敗後の再挑戦時", text="もう一回やってみようと思ったんだね、すごいね！", gesture="antenna_clap"),
            Phrase(id="B9", trigger="手が止まり不安そうな時", text="焦らなくて大丈夫。自分のペースでやってみようね。", gesture="shake_gentle_antenna_up"),
            Phrase(id="B10", trigger="集中している時", text="一生懸命取り組んでいて、とってもかっこいいよ！", gesture="lean_in_hold"),
        ],
        logical=[
            Phrase(id="C1", trigger="課題開始直後・確認", text="お手本をよく見てみよう。", gesture="point_sample"),
            Phrase(id="C2", trigger="組み立て順序指示", text="下から順番に作ってみよう。", gesture="sweep_bottom_to_top"),
            Phrase(id="C3", trigger="視点変更", text="後ろからの写真も見てみよう。", gesture="point_photo"),
            Phrase(id="C4", trigger="形状の比較促し", text="形は見本と一緒かな？", gesture="head_tilt"),
            Phrase(id="C5", trigger="次のステップ", text="次はどのピースを使うか考えてみよう。", gesture="antenna_pikopiko"),
            Phrase(id="C6", trigger="部分完成・進行時", text="できてきたね。このまま上の部分も作ってみよう！", gesture="look_up_nod"),
            Phrase(id="C7", trigger="色の比較促し", text="色が一緒か、もう一度お手本を見て確かめてみよう。", gesture="point_sample"),
            Phrase(id="C8", trigger="ピースの向き確認", text="ピースの向きを変えてみると、うまくはまるかも知れないよ。", gesture="roll_twist"),
            Phrase(id="C9", trigger="崩れた・失敗時", text="どこが違っていたか、もう一度見本と比べてみよう。", gesture="point_sample"),
            Phrase(id="C10", trigger="残り時間の意識促し", text="組み立て方の順番を、頭の中で整理してみよう。", gesture="thoughtful"),
        ],
        backchannel=[
            Phrase(id="BC1", text="うん", gesture="nod_small"),
            Phrase(id="BC2", text="ちがうよ", gesture="shake_small"),
            Phrase(id="BC3", text="そうだね", gesture="nod_small"),
            Phrase(id="BC4", text="できたね", gesture="happy_lean_short"),
        ],
    )


def _kf(head: dict[str, float] | None = None, antennas: list[float] | None = None, d: float = 0.3) -> Keyframe:
    return Keyframe(head=head, antennas=antennas, d=d)


def _hold(seconds: float) -> Keyframe:
    return Keyframe(hold=seconds)


def default_gestures() -> Gestures:
    ant_neutral = [10.0, 10.0]
    g: dict[str, GestureDef] = {
        # --- 頷き・首振り(カスタム)
        "nod_small": GestureDef(kind="keyframes", description="小さく頷く", frames=[_kf({"pitch": 8}, d=0.25), _kf({"pitch": 0}, d=0.3)]),
        "nod_gentle": GestureDef(kind="keyframes", description="やさしく頷く", frames=[_kf({"pitch": 10}, d=0.4), _kf({"pitch": 0}, d=0.5)]),
        "shake_small": GestureDef(kind="keyframes", description="小さく首を横に振る", frames=[_kf({"yaw": -12}, d=0.25), _kf({"yaw": 12}, d=0.35), _kf({"yaw": 0}, d=0.25)]),
        "shake_gentle_antenna_up": GestureDef(
            kind="keyframes",
            description="やさしく首を横に振ったあとアンテナを上げる",
            frames=[_kf({"yaw": -10}, d=0.4), _kf({"yaw": 10}, d=0.6), _kf({"yaw": 0}, d=0.4), _kf(antennas=[-25, -25], d=0.4), _hold(0.5), _kf(antennas=ant_neutral, d=0.4)],
        ),
        "tilt_sympathy": GestureDef(kind="keyframes", description="首を傾げて共感", frames=[_kf({"roll": 14, "pitch": 6}, d=0.6), _hold(0.8), _kf({"roll": 0, "pitch": 0}, d=0.6)]),
        "lean_in_nod": GestureDef(kind="keyframes", description="前傾して頷く", frames=[_kf({"pitch": 10, "x": 15}, d=0.5), _kf({"pitch": 16}, d=0.3), _kf({"pitch": 10}, d=0.3), _hold(0.6)]),
        "lean_in_hold": GestureDef(kind="keyframes", description="体を乗り出してじっと見つめる", frames=[_kf({"pitch": 8, "x": 18}, d=0.7), _hold(2.0)]),
        "sweep_bottom_to_top": GestureDef(kind="keyframes", description="下から上へ視線を動かす", frames=[_kf({"pitch": 20}, d=0.5), _kf({"pitch": -12}, d=1.2), _kf({"pitch": 0}, d=0.5)]),
        "roll_twist": GestureDef(kind="keyframes", description="ひねる・回す", frames=[_kf({"roll": -20}, d=0.4), _kf({"roll": 20}, d=0.6), _kf({"roll": -15}, d=0.5), _kf({"roll": 0}, d=0.4)]),
        "antenna_twitch": GestureDef(kind="keyframes", description="アンテナだけ小さく動かす(アイドル用)", frames=[_kf(antennas=[25, 25], d=0.2), _kf(antennas=ant_neutral, d=0.3)]),
        # --- 公式モーション集
        "greet_wiggle": GestureDef(kind="recorded", description="アンテナと首を振る(welcoming1)", file="welcoming1"),
        "welcoming": GestureDef(kind="recorded", description="歓迎(welcoming2)", file="welcoming2"),
        "happy_lean": GestureDef(kind="recorded", description="大きめに斜めになる(cheerful1)", file="cheerful1"),
        "happy_lean_short": GestureDef(kind="recorded", description="短い喜び(cheerful1 前半)", file="cheerful1", end=1.6),
        "antenna_clap": GestureDef(kind="recorded", description="拍手のようにアンテナをたたく(success1)", file="success1"),
        "enthusiastic": GestureDef(kind="recorded", description="張り切る(enthusiastic1)", file="enthusiastic1"),
        "calm_antennas": GestureDef(kind="recorded", description="頭は静止、アンテナだけゆっくり(serenity1)", file="serenity1"),
        "head_lower": GestureDef(kind="recorded", description="頭を下げる(downcast1 前半)", file="downcast1", end=3.0),
        "nod_slow_deep": GestureDef(kind="recorded", description="深くゆっくり頷く(understanding1)", file="understanding1"),
        "nod_big_x2": GestureDef(
            kind="keyframes",
            description="大きく頷く×2",
            frames=[_kf({"pitch": 18}, d=0.35), _kf({"pitch": 0}, d=0.35), _kf({"pitch": 18}, d=0.35), _kf({"pitch": 0}, d=0.4)],
        ),
        "banzai_small": GestureDef(kind="recorded", description="小さくバンザイ(amazed1)", file="amazed1"),
        "beckon": GestureDef(kind="recorded", description="手を差し伸べる(come1)", file="come1"),
        "head_tilt": GestureDef(kind="recorded", description="頭を傾ける(inquiring2)", file="inquiring2"),
        "antenna_pikopiko": GestureDef(kind="recorded", description="アンテナをぴこぴこ(inquiring3)", file="inquiring3"),
        "thoughtful": GestureDef(kind="recorded", description="考え込む(thoughtful1 前半)", file="thoughtful1", end=4.0),
        # --- 指さし(方向は settings.positions)
        "point_sample": GestureDef(kind="point", description="お手本の方を向いて頷く", target="sample"),
        "point_photo": GestureDef(kind="point", description="写真カードの方を向いて頷く", target="photo"),
        "point_pieces": GestureDef(kind="point", description="ピースの方を向いて頷く", target="pieces"),
        "look_up_nod": GestureDef(kind="point", description="上を向いて頷く", target="up"),
        # --- 連続
        "happy_lean_clap": GestureDef(kind="sequence", description="大きめに斜め → 小さく拍手", steps=["happy_lean", "antenna_clap"]),
    }
    return Gestures(g)


# ---------------------------------------------------------------- io

M = TypeVar("M", bound=BaseModel)


class ConfigError(Exception):
    pass


def load_or_create(path: Path, model: type[M], default: M) -> tuple[M, str | None]:
    """ファイルを読む。無ければデフォルトを書いて返す。壊れていれば退避してデフォルトを返す。

    戻り値: (モデル, 警告メッセージ or None)
    """
    if not path.exists():
        save_atomic(path, default)
        return default, None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return model.model_validate(raw), None
    except Exception as e:  # JSON 構文エラー / 検証エラー
        broken = path.with_name(f"{path.name}.broken-{time.strftime('%Y%m%d-%H%M%S')}")
        try:
            os.replace(path, broken)
        except OSError:
            pass
        save_atomic(path, default)
        return default, f"{path.name} を読めなかったのでデフォルトに戻しました(元のファイルは {broken.name})。詳細: {e}"


def save_atomic(path: Path, model: BaseModel) -> None:
    """一時ファイルに書いて fsync → 置換。直前の内容は `<name>.bak` に 1 世代残す(誤保存の保険)。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(model.model_dump_json(indent=2))
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        try:
            shutil.copyfile(path, path.with_suffix(path.suffix + ".bak"))
        except OSError:
            pass
    os.replace(tmp, path)


_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def expand(text: str, **values: str) -> str:
    """プレースホルダ {robot} {child} {experimenter} を展開する。

    未知のキーはそのまま残し、波括弧が崩れていても例外にしない(str.format と違い、台本に
    '{' が 1 つ紛れ込んでも再生が止まらない)。
    """
    return _PLACEHOLDER_RE.sub(lambda m: values.get(m.group(1), m.group(0)), text)


def validate_phrases(ph: Phrases, gesture_names: set[str]) -> list[str]:
    """台本の不備を列挙する(保存前に呼ぶ)。空なら OK。"""
    errors: list[str] = []
    ids: list[str] = []
    for step in ph.intro:
        ids.extend(p.id for p in step.leaves())
    for cat in ("empathy", "logical", "backchannel"):
        ids.extend(p.id for p in getattr(ph, cat))
    if not ids:
        errors.append("台本が空です")
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        errors.append("ID が重複しています: " + ", ".join(dup))
    for k, v in ph.leaves().items():
        if not v.text.strip():
            errors.append(f"{k}: 本文が空です")
        if v.gesture not in gesture_names:
            errors.append(f"{k}: ジェスチャー '{v.gesture}' がありません")
    for order in ("robot_first", "experimenter_first"):
        if not ph.intro_sequence(order):
            errors.append(f"順序条件 {order} のイントロがありません")
    return errors
