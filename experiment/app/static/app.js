/* =====================================================================
 * Reachy Mini 実験コントローラ — ブラウザ側（依存なし・単一ファイル）
 *
 * 構成:
 *   1. ユーティリティ（DOM 生成・整形・API 呼び出し・トースト）
 *   2. 状態（サーバー snapshot のミラー）と SSE（自動再接続・ウォッチドッグ）
 *   3. 画面: 開始 / 操作 / 設定
 *   4. タイマー tick（250 ms）
 *
 * 方針: 例外を外に漏らさない。全ての fetch にタイムアウト。失敗はトーストで知らせる。
 * ===================================================================== */
'use strict';
(function () {
  // ------------------------------------------------------------------ 定数
  const PREFLIGHT_STEPS = [
    ['voicevox', 'VOICEVOX'],
    ['robot', 'ロボット接続'],
    ['motors', 'モーター'],
    ['synth', '音声合成'],
    ['upload', '音声転送'],
    ['tracking', '顔追跡設定'],
    ['volume', '音量'],
  ];
  const PF_ICON = { pending: '○', running: '◌', ok: '✓', fail: '✕' };
  const PHASE_LABEL = { intro: 'イントロ', baseline: 'ベースライン', main: '本番', ended: '終了' };
  const COND_LABEL = { empathy: '共感', logical: '論理' };
  const ORDER_LABEL = { robot_first: 'ロボット先行', experimenter_first: '実験者先行' };
  const CONN_LABEL = { connected: '接続中', degraded: '不安定', disconnected: '切断', recovering: '復旧中' };
  const LOG_KIND = {
    button: 'ボタン', playing: '再生', done: '完了', interrupted: '中断', stop: '停止', pause: '一時停止',
    resume: '再開', phase: 'フェーズ', session: 'セッション', system: 'システム', error: 'エラー', tracking: '顔追跡',
  };
  const SSE_EVENTS = ['snapshot', 'connection', 'face', 'session', 'performer', 'preflight', 'toast', 'log',
    'settings_changed', 'resting', 'volume', 'heartbeat', 'recovery', 'error'];
  const HIRAGANA_RE = /^[ぁ-ゖー]+$/;
  const SSE_TIMEOUT_MS = 8000;   // これ以上メッセージが無ければ SSE を張り直す（ハートビートは 3 秒毎）
  const MAX_LOG = 100;
  const MAX_TOAST = 4;
  const TABS = ['general', 'positions', 'voice', 'script', 'motion'];
  // iPad Safari の音声再生ロック解除用（無音 10 ms の WAV。外部リソースではない）
  const SILENT_WAV = 'data:audio/wav;base64,UklGRnQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YVAAAACAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgA==';

  // ------------------------------------------------------------------ 状態
  const S = {
    snap: null,            // サーバー snapshot のミラー
    screen: 'start',       // start | control | settings
    serverOk: false,       // SSE が生きているか（サーバーランプ）
    serverConnecting: false,
    clockOffset: 0,        // server_time*1000 - Date.now()
    remainingAt: null,     // {value, at}: 本番残り秒（ハートビート間は外挿）
    sinceAt: null,         // {value, at}: 前回の声かけからの秒数
    starting: false,       // セッション開始中
    pendingId: null,       // API 応答待ちのボタン
    debounced: new Set(),  // デバウンス中のボタン id
    busy: new Set(),       // 実行中の制御操作
    logs: [],              // 新しい順
    logKeys: new Set(),
    logDirty: false,
    buttonsKey: '',        // 再生ボタン群の構造キー（変化時のみ DOM を作り直す）
    voices: null,          // null=未取得 | 'error' | [{id,name,credit}]
    voicesRequested: false,
    draft: null,           // 設定画面の編集中データ {settings, phrases}
    gesturesText: null,    // ジェスチャー JSON テキスト
    previewText: null,
    volumeDraft: null,
    settingsTab: 'general',
    es: null,
    sseLastMsg: 0,
    reconnectTimer: null,
  };

  // ------------------------------------------------------------------ DOM ユーティリティ
  const $ = (sel, root) => (root || document).querySelector(sel);
  const $$ = (sel, root) => Array.from((root || document).querySelectorAll(sel));
  const PROP_KEYS = new Set(['value', 'checked', 'disabled', 'hidden', 'selected', 'open', 'readOnly']);

  /** 要素生成。文字列の子は textContent として追加される（HTML は解釈しない）。 */
  function h(tag, attrs, ...children) {
    const e = document.createElement(tag);
    if (attrs) {
      for (const k of Object.keys(attrs)) {
        const v = attrs[k];
        if (v === null || v === undefined || v === false) continue;
        if (k === 'class') e.className = v;
        else if (k === 'dataset') Object.assign(e.dataset, v);
        else if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
        else if (k.startsWith('on') && typeof v === 'function') e.addEventListener(k.slice(2), v);
        else if (PROP_KEYS.has(k)) e[k] = v;
        else e.setAttribute(k, v === true ? '' : String(v));
      }
    }
    appendChildren(e, children);
    return e;
  }
  function appendChildren(parent, children) {
    // 配列以外(単一ノード / 文字列 / null)も受け付ける
    if (children === null || children === undefined || children === false) return;
    if (!Array.isArray(children)) children = [children];
    for (const c of children) {
      if (c === null || c === undefined || c === false) continue;
      if (Array.isArray(c)) appendChildren(parent, c);
      else parent.append(c instanceof Node ? c : String(c));
    }
  }
  function setChildren(parent, children) {
    parent.replaceChildren();
    appendChildren(parent, children);
  }

  // ------------------------------------------------------------------ 整形
  const pad2 = (n) => String(n).padStart(2, '0');
  function fmtClock(unixSec) {
    const d = new Date((Number(unixSec) || 0) * 1000);
    return `${pad2(d.getHours())}:${pad2(d.getMinutes())}:${pad2(d.getSeconds())}`;
  }
  function fmtMMSS(sec) {
    const s = Math.max(0, Math.ceil(sec));
    return `${pad2(Math.floor(s / 60))}:${pad2(s % 60)}`;
  }
  function fmtNum(v) { return typeof v === 'number' ? String(Math.round(v * 1000) / 1000) : String(v); }
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const clone = (o) => JSON.parse(JSON.stringify(o));
  const serverNowSec = () => (Date.now() + S.clockOffset) / 1000;
  function setClock(serverTime) {
    if (typeof serverTime === 'number' && isFinite(serverTime)) S.clockOffset = serverTime * 1000 - Date.now();
  }
  function stripMeta(d) {
    const o = Object.assign({}, d);
    delete o.type;
    delete o.time;
    return o;
  }
  const sessionActive = () => !!(S.snap && S.snap.session && S.snap.session.active);
  function childDisplay() { return sessionActive() ? S.snap.session.child : '〇〇ちゃん'; }
  /** 台本のプレースホルダを展開（表示用） */
  function expandText(text) {
    const names = (S.snap && S.snap.settings && S.snap.settings.names) || {};
    return String(text || '')
      .replace(/\{robot\}/g, names.robot || 'ロボット')
      .replace(/\{child\}/g, childDisplay())
      .replace(/\{experimenter\}/g, names.experimenter || '実験者');
  }

  // ------------------------------------------------------------------ トースト
  function toast(level, message) {
    try {
      const box = $('#toasts');
      if (!box) return;
      message = String(message || '');
      // 同じ文面が表示中なら重ねず、表示時間だけ延長する（サーバー側とクライアント側の二重通知対策）
      for (const t of Array.from(box.children)) {
        if (t.dataset.msg === message) {
          clearTimeout(t._timer);
          t._timer = setTimeout(() => t.remove(), level === 'error' ? 8000 : 5000);
          return;
        }
      }
      while (box.children.length >= MAX_TOAST) box.firstElementChild.remove();
      const t = h('div', { class: `toast toast-${level || 'info'}`, dataset: { msg: message }, role: 'status' },
        h('span', { class: 'toast-msg' }, message),
        h('button', { type: 'button', class: 'toast-close', 'aria-label': '閉じる', onclick: () => t.remove() }, '×'));
      box.append(t);
      t._timer = setTimeout(() => t.remove(), level === 'error' ? 8000 : 5000);
    } catch (e) {
      console.error('toast failed', e);
    }
  }

  // ------------------------------------------------------------------ API
  /**
   * JSON API 呼び出し。body があれば POST（opts.method で上書き可）。
   * 非 2xx は Error(json.error || ...) を投げる。全て AbortController でタイムアウト。
   * opts.raw = true なら Response をそのまま返す（音声バイナリ用）。
   */
  async function api(path, body, opts) {
    opts = opts || {};
    const method = opts.method || (body === undefined ? 'GET' : 'POST');
    const timeoutMs = opts.timeoutMs || 4000;
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const init = { method, signal: ctrl.signal, cache: 'no-store', headers: {} };
      if (body !== undefined) {
        init.headers['Content-Type'] = 'application/json';
        init.body = JSON.stringify(body);
      }
      const res = await fetch(path, init);
      if (opts.raw) {
        if (!res.ok) throw new Error(await errorFromResponse(res));
        return res;
      }
      const text = await res.text();
      let json = null;
      if (text) {
        try { json = JSON.parse(text); } catch (_) { json = null; }
      }
      if (!res.ok) throw new Error(errorFromJson(json, res.status));
      return json;
    } catch (e) {
      if (e && e.name === 'AbortError') throw new Error('サーバーが応答しません（タイムアウト）');
      if (e instanceof TypeError) throw new Error('サーバーに接続できません');
      throw e;
    } finally {
      clearTimeout(timer);
    }
  }
  function errorFromJson(json, status) {
    if (json && typeof json.error === 'string') return json.error;
    if (json && json.detail) return typeof json.detail === 'string' ? json.detail : '入力が正しくありません';
    return `サーバーエラー (HTTP ${status})`;
  }
  async function errorFromResponse(res) {
    let json = null;
    try { json = await res.json(); } catch (_) { json = null; }
    return errorFromJson(json, res.status);
  }

  /** 実行中フラグ付きで制御操作を実行する（二重押し防止・エラーはトースト）。 */
  async function control(key, fn) {
    if (S.busy.has(key)) return;
    S.busy.add(key);
    renderBusy();
    try {
      await fn();
    } catch (e) {
      toast('error', e && e.message ? e.message : String(e));
    } finally {
      S.busy.delete(key);
      renderBusy();
    }
  }
  /** 実行中フラグの変化を今の画面に反映する。 */
  function renderBusy() {
    if (!S.snap) return;
    try {
      if (S.screen === 'control') renderControls();
      else if (S.screen === 'start') renderStart();
      else if (S.screen === 'settings') {
        for (const b of $$('.save-bar .btn')) b.disabled = S.busy.has('save');
      }
    } catch (e) { console.error(e); }
  }

  // ------------------------------------------------------------------ SSE
  function connectSSE() {
    clearTimeout(S.reconnectTimer);
    S.reconnectTimer = null;
    closeSSE();
    let es;
    try {
      es = new EventSource('/api/events');
    } catch (e) {
      scheduleReconnect(2000);
      return;
    }
    S.es = es;
    S.serverConnecting = true;
    S.sseLastMsg = Date.now();
    es.onopen = () => {
      S.serverOk = true;
      S.serverConnecting = false;
      S.sseLastMsg = Date.now();
      renderTopbarSafe();
    };
    es.onerror = () => {
      S.serverOk = false;
      if (es.readyState === EventSource.CLOSED) {
        S.serverConnecting = false;
        scheduleReconnect(1000);
      } else {
        S.serverConnecting = true; // ブラウザが自動再試行中
      }
      renderTopbarSafe();
    };
    for (const type of SSE_EVENTS) {
      es.addEventListener(type, (ev) => {
        S.sseLastMsg = Date.now();
        if (!S.serverOk) {
          S.serverOk = true;
          S.serverConnecting = false;
        }
        let data;
        try { data = JSON.parse(ev.data); } catch (_) { return; }
        try { handleEvent(type, data); } catch (e) { console.error('event handling failed', type, e); }
      });
    }
  }
  function closeSSE() {
    if (S.es) {
      try { S.es.close(); } catch (_) { /* ignore */ }
      S.es = null;
    }
  }
  function scheduleReconnect(ms) {
    if (S.reconnectTimer) return;
    S.reconnectTimer = setTimeout(() => {
      S.reconnectTimer = null;
      connectSSE();
    }, ms);
  }
  // ウォッチドッグ: 8 秒メッセージが無ければ切断扱いにして張り直す
  setInterval(() => {
    try {
      if (S.es && Date.now() - S.sseLastMsg > SSE_TIMEOUT_MS) {
        S.serverOk = false;
        S.serverConnecting = false;
        closeSSE();
        renderTopbarSafe();
        scheduleReconnect(1000);
      }
    } catch (e) { console.error(e); }
  }, 1000);

  function renderTopbarSafe() {
    try { if (S.snap && S.screen === 'control') renderTopbar(); } catch (e) { console.error(e); }
  }

  // ------------------------------------------------------------------ イベント適用
  function handleEvent(type, data) {
    if (type === 'snapshot') { applySnapshot(data); return; }
    if (type === 'heartbeat') { applyHeartbeat(data); return; }
    if (!S.snap) return; // snapshot 未受信なら無視
    const snap = S.snap;
    switch (type) {
      case 'connection':
        snap.connection = Object.assign({}, snap.connection, stripMeta(data));
        addLog(data);
        break;
      case 'face':
        snap.connection.face_detected = !!data.detected;
        break;
      case 'session':
        applySession(stripMeta(data));
        break;
      case 'performer': {
        const prev = snap.performer || {};
        snap.performer = stripMeta(data);
        setClock(data.server_time);
        // 励ましの再生開始で「前回の声かけ」を即座に 0 に（ハートビート待ちにしない）
        const cur = snap.performer.current;
        if (snap.performer.status === 'playing' && cur && (cur.category === 'empathy' || cur.category === 'logical')
            && !(prev.current && prev.current.id === cur.id && prev.current.started_at === cur.started_at)) {
          S.sinceAt = { value: 0, at: Date.now() };
        }
        if (snap.performer.status !== 'preparing' && S.pendingId) S.pendingId = null;
        break;
      }
      case 'preflight': {
        if (!data.step) return;
        const st = { status: data.status || 'pending', message: data.message || '' };
        if (Array.isArray(data.progress)) st.progress = data.progress;
        snap.preflight = snap.preflight || {};
        snap.preflight[data.step] = st;
        if (data.step === 'voicevox' && st.status === 'ok') maybeLoadVoices();
        break;
      }
      case 'toast':
        toast(data.level || 'info', data.message || '');
        addLog(data);
        break;
      case 'log':
        addLog(data);
        break;
      case 'settings_changed':
        refreshState();
        return;
      case 'resting':
        snap.resting = !!data.resting;
        break;
      case 'volume':
        snap.volume = data.volume;
        break;
      case 'recovery':
      case 'error':
        addLog(data);
        break;
      default:
        return;
    }
    render();
  }

  function applySnapshot(snap) {
    if (!snap || typeof snap !== 'object') return;
    S.snap = snap;
    setClock(snap.server_time);
    updateTimersFromSession(snap.session || { active: false });
    for (const ev of snap.recent_events || []) addLog(ev);
    if (snap.preflight && snap.preflight.voicevox && snap.preflight.voicevox.status === 'ok') maybeLoadVoices();
    autoScreen();
    render();
  }

  function applyHeartbeat(d) {
    setClock(d.server_time);
    if (!S.snap) return;
    const snap = S.snap;
    if (typeof d.connection === 'string' && snap.connection && snap.connection.state !== d.connection) {
      snap.connection.state = d.connection;
      if (d.connection === 'connected') snap.connection.reason = '';
    }
    if (typeof d.performer === 'string' && snap.performer && snap.performer.status !== d.performer) {
      snap.performer.status = d.performer;
      if (d.performer === 'idle') snap.performer.current = null;
    }
    if (snap.session && snap.session.active) {
      S.remainingAt = d.main_remaining_s == null ? null : { value: Number(d.main_remaining_s), at: Date.now() };
      S.sinceAt = d.since_last_utterance_s == null ? null : { value: Number(d.since_last_utterance_s), at: Date.now() };
    }
    render();
  }

  function applySession(sess) {
    if (!S.snap || !sess) return;
    S.snap.session = sess;
    updateTimersFromSession(sess);
    autoScreen();
  }
  function updateTimersFromSession(sess) {
    if (!sess || !sess.active) {
      S.remainingAt = null;
      S.sinceAt = null;
      return;
    }
    S.remainingAt = sess.main_remaining_s == null ? null : { value: Number(sess.main_remaining_s), at: Date.now() };
    S.sinceAt = sess.since_last_utterance_s == null ? null : { value: Number(sess.since_last_utterance_s), at: Date.now() };
  }
  function applyPerformer(p) {
    if (p && typeof p.status === 'string' && S.snap) {
      S.snap.performer = p;
      setClock(p.server_time);
      render();
    }
  }

  /** セッションの有無に合わせて開始画面 / 操作画面を自動で切り替える（設定画面中は触らない）。 */
  function autoScreen() {
    const active = sessionActive();
    if (active && S.screen === 'start' && !S.starting) showScreen('control');
    else if (!active && S.screen === 'control') showScreen('start');
  }

  let refreshing = false;
  async function refreshState() {
    if (refreshing) return;
    refreshing = true;
    try {
      const snap = await api('/api/state', undefined, { timeoutMs: 4000 });
      if (snap) applySnapshot(snap);
    } catch (e) {
      console.warn('refreshState failed', e);
    } finally {
      refreshing = false;
    }
  }

  // ------------------------------------------------------------------ ログ
  const logKey = (ev) => `${ev.type}:${ev.time}:${ev.kind || ev.level || ev.state || ''}`;
  function addLog(ev) {
    if (!ev || typeof ev !== 'object' || !ev.type) return;
    const key = logKey(ev);
    if (S.logKeys.has(key)) return;
    S.logKeys.add(key);
    S.logs.push(ev);
    S.logs.sort((a, b) => (b.time || 0) - (a.time || 0));
    if (S.logs.length > MAX_LOG) {
      for (const d of S.logs.splice(MAX_LOG)) S.logKeys.delete(logKey(d));
    }
    S.logDirty = true;
  }
  function logLine(ev) {
    switch (ev.type) {
      case 'log': {
        const parts = [ev.item_id, ev.text, ev.gesture ? `[${ev.gesture}]` : '', ev.detail].filter(Boolean);
        if (ev.result === 'error') parts.push('(失敗)');
        return { tag: LOG_KIND[ev.kind] || ev.kind || 'ログ', text: parts.join(' '), cls: (ev.kind === 'error' || ev.result === 'error') ? 'is-error' : '' };
      }
      case 'toast':
        return { tag: ev.level === 'error' ? 'エラー' : ev.level === 'warn' ? '注意' : 'お知らせ', text: ev.message || '', cls: ev.level === 'error' ? 'is-error' : ev.level === 'warn' ? 'is-warn' : '' };
      case 'connection':
        return { tag: '接続', text: `${CONN_LABEL[ev.state] || ev.state || ''} ${ev.reason || ''}`.trim(), cls: ev.state === 'connected' ? 'is-ok' : 'is-warn' };
      case 'recovery':
        return { tag: '復旧', text: '復旧が完了しました', cls: 'is-ok' };
      case 'error':
        return { tag: 'エラー', text: ev.message || ev.detail || '', cls: 'is-error' };
      default:
        return { tag: ev.type, text: ev.message || ev.detail || '', cls: '' };
    }
  }
  function renderLog() {
    if (!S.logDirty) return;
    S.logDirty = false;
    const list = $('#log-list');
    if (!list) return;
    setChildren(list, S.logs.map((ev) => {
      const l = logLine(ev);
      return h('li', { class: `log-item ${l.cls}` }, h('time', null, fmtClock(ev.time)), h('span', { class: 'log-tag' }, l.tag), h('span', { class: 'log-text' }, l.text));
    }));
    const latest = $('#log-latest');
    if (latest) {
      const ev = S.logs[0];
      latest.textContent = ev ? `${fmtClock(ev.time)} ${logLine(ev).tag}: ${logLine(ev).text}` : '';
    }
  }

  // ------------------------------------------------------------------ 画面切替
  function showScreen(name) {
    S.screen = name;
    for (const n of ['start', 'control', 'settings']) {
      const el = $(`#screen-${n}`);
      if (el) el.hidden = n !== name;
    }
    document.body.dataset.screen = name;
    window.scrollTo(0, 0);
    if (name !== 'settings') S.draft = null;
    render();
  }

  function render() {
    if (!S.snap) return;
    try {
      if (S.screen === 'start') renderStart();
      else if (S.screen === 'control') renderControl();
      renderLog();
    } catch (e) {
      console.error('render failed', e);
    }
  }

  // ================================================================== 開始画面
  function renderStart() {
    const snap = S.snap;
    const conn = snap.connection || {};
    const pf = snap.preflight || {};
    $('#start-version').textContent = snap.app_version ? `v${snap.app_version}` : '';

    // 警告（設定ファイルの復旧など）
    const warnBox = $('#start-warnings');
    const warnings = Array.isArray(snap.warnings) ? snap.warnings : [];
    warnBox.hidden = warnings.length === 0;
    setChildren(warnBox, warnings.length ? h('ul', null, warnings.map((w) => h('li', null, String(w)))) : null);

    // プリフライト一覧
    setChildren($('#preflight-list'), PREFLIGHT_STEPS.map(([key, label]) => {
      const st = pf[key] || { status: 'pending', message: '' };
      const status = PF_ICON[st.status] ? st.status : 'pending';
      const li = h('li', { class: `pf pf-${status}` },
        h('span', { class: 'pf-icon' }, PF_ICON[status]),
        h('span', { class: 'pf-label' }, label),
        h('span', { class: 'pf-msg' }, st.message || ''));
      const pr = st.progress;
      if (Array.isArray(pr) && pr.length === 2 && Number(pr[1]) > 0) {
        const pct = clamp((Number(pr[0]) / Number(pr[1])) * 100, 0, 100);
        li.append(h('div', { class: 'pf-bar' }, h('i', { style: { width: `${pct.toFixed(0)}%` } })), h('span', { class: 'pf-count' }, `${pr[0]} / ${pr[1]}`));
      }
      return li;
    }));
    const anyFail = PREFLIGHT_STEPS.some(([k]) => pf[k] && pf[k].status === 'fail');
    $('#btn-preflight-retry').disabled = !anyFail || S.busy.has('preflight');
    const connText = conn.state === 'connected'
      ? `ロボット接続中（${conn.robot_url || ''}）`
      : `ロボット: ${CONN_LABEL[conn.state] || conn.state || '不明'}${conn.reason ? '：' + conn.reason : ''}`;
    const connBox = $('#start-conn');
    const connMode = snap.resting ? 'resting' : 'normal';
    if (connBox.dataset.mode !== connMode) {
      connBox.dataset.mode = connMode;
      setChildren(connBox, [h('span', { id: 'start-conn-text' }),
        snap.resting ? h('button', { type: 'button', class: 'btn btn-sm', onclick: doWake }, '起こす') : null]);
    }
    $('#start-conn-text').textContent = snap.resting ? `${connText} — ロボットは休止中です ` : connText;
    if (snap.resting) $('#start-conn').querySelector('button').disabled = S.busy.has('rest');

    // 開始ボタン
    const synthOk = pf.synth && pf.synth.status === 'ok';
    const ready = conn.state === 'connected' && synthOk;
    const btn = $('#btn-start');
    btn.disabled = !ready || S.starting;
    const mode = S.starting ? 'starting' : 'idle';
    if (btn.dataset.mode !== mode) {
      btn.dataset.mode = mode;
      setChildren(btn, S.starting ? [h('span', { class: 'spinner' }), '名前の音声を準備中…'] : ['準備してはじめる']);
    }
    const hint = $('#start-hint');
    if (S.starting) hint.textContent = '名前入りの音声を合成してロボットへ転送しています（最大 15 秒ほど）';
    else if (!ready) {
      const why = [];
      if (conn.state !== 'connected') why.push('ロボットが未接続');
      if (!synthOk) why.push('音声の準備が未完了');
      hint.textContent = `開始できません: ${why.join('・')}`;
    } else hint.textContent = '';

    renderStartCredit();
  }

  function renderStartCredit() {
    const el = $('#start-credit');
    if (!el || !S.snap) return;
    const tts = (S.snap.settings && S.snap.settings.tts) || {};
    const v = Array.isArray(S.voices) ? S.voices.find((x) => String(x.id) === String(tts.voice_id)) : null;
    el.textContent = v ? `音声: ${v.credit || 'VOICEVOX'}（${v.name}）` : `音声: VOICEVOX（声 ID ${tts.voice_id != null ? tts.voice_id : '?'}）`;
  }

  function segValue(id) {
    const b = $(`#${id} button.active`);
    return b ? b.dataset.value : '';
  }

  function validateName(showError) {
    const input = $('#in-child-name');
    const name = input.value.trim();
    const ok = HIRAGANA_RE.test(name);
    const err = $('#err-child-name');
    err.hidden = ok || (!showError && name === '');
    input.classList.toggle('invalid', !ok && (showError || name !== ''));
    return ok ? name : null;
  }

  async function startSession(ev) {
    if (ev) ev.preventDefault();
    if (S.starting || !S.snap) return;
    const name = validateName(true);
    if (!name) {
      $('#in-child-name').focus();
      return;
    }
    const body = { child_name: name, suffix: segValue('seg-suffix'), order: segValue('seg-order'), condition: segValue('seg-condition') };
    S.starting = true;
    renderStart();
    try {
      // 声や台本を変えた直後は全音声を作り直すため時間がかかる(サーバー側は進捗を preflight で流す)
      const sess = await api('/api/session/start', body, { timeoutMs: 90000 });
      S.starting = false;
      if (sess && sess.active) {
        applySession(sess);
        showScreen('control');
      } else {
        toast('warn', 'セッションを開始できませんでした（サーバーの応答が不正です）');
      }
    } catch (e) {
      toast('error', `セッションを開始できません: ${e.message}`);
      // 失敗・タイムアウトのどちらでもサーバーの実際の状態に合わせ直す(準備中表示が残らないように)
      refreshState().catch(() => {});
    } finally {
      S.starting = false;
      render();
    }
  }

  // ================================================================== 操作画面
  function renderControl() {
    renderTopbar();
    renderPlayButtons();
    updatePlayButtons();
    renderControls();
    tick();
  }

  function lampColor(state) {
    if (state === 'connected') return 'green';
    if (state === 'degraded' || state === 'recovering') return 'amber';
    return 'red';
  }
  function setLamp(el, color, text) {
    el.classList.remove('green', 'amber', 'red');
    el.classList.add(color);
    el.querySelector('small').textContent = text || '';
  }

  function renderTopbar() {
    const snap = S.snap;
    const conn = snap.connection || {};
    const sess = snap.session || { active: false };
    const perf = snap.performer || {};

    setLamp($('#lamp-robot'), lampColor(conn.state),
      conn.state === 'connected' ? '' : `${CONN_LABEL[conn.state] || conn.state || ''}${conn.reason ? '：' + conn.reason : ''}`);
    setLamp($('#lamp-server'), S.serverOk ? 'green' : (S.serverConnecting ? 'amber' : 'red'),
      S.serverOk ? '' : (S.serverConnecting ? '再接続中…' : '切断'));
    $('#face-indicator').hidden = !conn.face_detected;

    $('#badge-child').textContent = sess.active ? sess.child : '';
    const cond = $('#badge-condition');
    cond.textContent = COND_LABEL[sess.condition] || '';
    cond.classList.toggle('cond-empathy', sess.condition === 'empathy');
    cond.classList.toggle('cond-logical', sess.condition === 'logical');
    $('#badge-order').textContent = ORDER_LABEL[sess.order] || '';
    const chip = $('#phase-chip');
    chip.textContent = sess.active ? (PHASE_LABEL[sess.phase] || sess.phase || '') : '';
    chip.dataset.phase = sess.phase || '';

    $('#topbar').classList.toggle('paused', !!perf.paused);

    const np = $('#now-playing');
    np.classList.remove('paused', 'preparing');
    if (perf.paused) {
      np.textContent = '⏸ 一時停止中 — 「再開」を押すまで顔追跡とアイドル動作は止まります';
      np.classList.add('paused');
    } else if (perf.status === 'preparing') {
      np.textContent = '準備中…';
      np.classList.add('preparing');
    } else if (perf.status === 'playing' && perf.current) {
      np.textContent = `再生中: ${perf.current.id} ${perf.current.text || ''}`;
    } else {
      np.textContent = '';
    }

    const disconnected = conn.state === 'disconnected' || conn.state === 'recovering';
    const banner = $('#banner-disconnected');
    banner.hidden = !disconnected;
    if (disconnected) $('#banner-reason').textContent = `${CONN_LABEL[conn.state] || ''}${conn.reason ? '：' + conn.reason : ''}`;
    $('#overlay-resting').hidden = !snap.resting;
  }

  function canPlay() {
    const conn = S.snap.connection || {};
    return conn.state !== 'disconnected' && conn.state !== 'recovering' && !S.snap.resting;
  }

  function playButton(item, category) {
    const head = h('span', { class: 'pb-head' }, h('span', { class: 'pb-num' }, item.id));
    if (category === 'intro') head.append(h('span', { class: 'pb-label' }, item.label || ''));
    else if (item.trigger) head.append(h('span', { class: 'pb-trigger' }, item.trigger));
    head.append(h('span', { class: 'pb-check' }, '✓'));
    return h('button', { type: 'button', class: `btn btn-play cat-${category}`, dataset: { id: item.id }, onclick: () => playPhrase(item.id) },
      head,
      h('span', { class: 'pb-text' }, expandText(item.text)),
      h('span', { class: 'pb-progress' }, h('i')));
  }

  /** ボタン群の DOM は構造が変わったときだけ作り直す（タップ中の DOM 差し替えを避ける）。 */
  function renderPlayButtons() {
    const snap = S.snap;
    const sess = snap.session || { active: false };
    const phrases = snap.phrases || {};
    const leaves = snap.leaves || {};
    const key = JSON.stringify([sess.intro_sequence, sess.condition, sess.child, snap.settings && snap.settings.names, leaves, phrases]);
    if (key === S.buttonsKey) return;
    S.buttonsKey = key;

    setChildren($('#intro-buttons'), (sess.intro_sequence || []).map((id) => playButton(leaves[id] || { id, text: '', label: '' }, 'intro')));
    const cond = sess.condition === 'logical' ? 'logical' : 'empathy';
    const other = cond === 'empathy' ? 'logical' : 'empathy';
    $('#enc-title').textContent = `励まし（${COND_LABEL[cond]}）`;
    setChildren($('#enc-buttons'), (phrases[cond] || []).map((p) => playButton(p, cond)));
    $('#enc-other-title').textContent = `もう一方の条件（${COND_LABEL[other]}・誤操作注意）`;
    setChildren($('#enc-other-buttons'), (phrases[other] || []).map((p) => playButton(p, other)));
    setChildren($('#bc-buttons'), (phrases.backchannel || []).map((p) => playButton(p, 'backchannel')));
  }

  function updatePlayButtons() {
    if (!S.snap || S.screen !== 'control') return;
    const perf = S.snap.performer || {};
    const sess = S.snap.session || {};
    const playingId = perf.status !== 'idle' && perf.current ? perf.current.id : null;
    const done = new Set(sess.intro_done || []);
    const blocked = !canPlay();
    for (const b of $$('.btn-play')) {
      const id = b.dataset.id;
      b.disabled = blocked || S.debounced.has(id);
      b.classList.toggle('playing', id === playingId);
      b.classList.toggle('pending', id === S.pendingId && id !== playingId);
      b.classList.toggle('done', done.has(id));
      if (id !== playingId) {
        const bar = b.querySelector('.pb-progress > i');
        if (bar) bar.style.width = '0%';
      }
    }
  }

  function setDebounce(id) {
    const ui = (S.snap && S.snap.settings && S.snap.settings.ui) || {};
    const ms = clamp(Number(ui.debounce_ms) || 500, 100, 5000);
    S.debounced.add(id);
    setTimeout(() => {
      S.debounced.delete(id);
      updatePlayButtons();
    }, ms);
  }

  async function playPhrase(id) {
    if (!S.snap || S.debounced.has(id)) return;
    setDebounce(id);
    S.pendingId = id;
    updatePlayButtons();
    try {
      // 未合成の音声があると合成 + 転送で数秒かかることがあるので少し長め
      const r = await api('/api/play', { id }, { timeoutMs: 10000 });
      if (r && r.accepted === false && r.reason !== 'debounce') toast('warn', `再生できませんでした: ${r.reason || ''}`);
    } catch (e) {
      toast('error', e.message);
    } finally {
      if (S.pendingId === id) S.pendingId = null;
      updatePlayButtons();
    }
  }

  function renderPhaseButtons() {
    const sess = S.snap.session || { active: false };
    const bBase = $('#btn-phase-baseline');
    const bMain = $('#btn-phase-main');
    bBase.hidden = sess.order !== 'robot_first';
    bBase.classList.toggle('active', sess.phase === 'baseline');
    bMain.classList.toggle('active', sess.phase === 'main');
    bBase.disabled = !sess.active || S.busy.has('phase');
    bMain.disabled = !sess.active || S.busy.has('phase');
    bMain.textContent = sess.phase === 'main' ? '本番中（タイマー進行中）' : '本番開始（タイマー開始）';
  }

  function setPhase(phase) {
    const sess = S.snap.session || {};
    if (!sess.active) return;
    if (phase === 'main' && sess.phase === 'main' && !window.confirm('すでに本番中です。タイマーはそのまま続きます。よろしいですか？')) return;
    control('phase', async () => {
      const s = await api('/api/session/phase', { phase });
      if (s) applySession(s);
      render();
    });
  }

  function renderControls() {
    if (!S.snap || S.screen !== 'control') return;
    const perf = S.snap.performer || {};
    const bPause = $('#btn-pause');
    bPause.textContent = perf.paused ? '▶ 再開' : '⏸ 一時停止';
    bPause.classList.toggle('active', !!perf.paused);
    bPause.disabled = S.busy.has('pause');
    // Face-tracking switch: show the current state (ON / OFF / paused) and what a press will do,
    // so the label is never read as a command.
    const bTrack = $('#btn-tracking');
    const trackingOn = !!perf.tracking_enabled;
    const trackingPaused = trackingOn && !!perf.paused;
    $('#tracking-state').textContent = trackingPaused ? '一時停止中' : (trackingOn ? 'ON' : 'OFF');
    $('#tracking-hint').textContent = trackingOn ? '押すと OFF にする' : '押すと ON にする';
    bTrack.classList.toggle('sw-on', trackingOn && !trackingPaused);
    bTrack.classList.toggle('sw-paused', trackingPaused);
    bTrack.classList.toggle('sw-off', !trackingOn);
    bTrack.setAttribute('aria-checked', trackingOn ? 'true' : 'false');
    bTrack.disabled = S.busy.has('tracking');
    const bRest = $('#btn-rest');
    bRest.textContent = S.snap.resting ? '起こす' : 'ロボットを休ませる';
    bRest.disabled = S.busy.has('rest');
    $('#btn-wake-overlay').disabled = S.busy.has('rest');
    $('#btn-end').disabled = S.busy.has('end') || !sessionActive();
    const stopping = S.busy.has('stop');
    $('#btn-stop').classList.toggle('is-busy', stopping);
    $('#btn-stop-float').classList.toggle('is-busy', stopping);
    renderPhaseButtons();
  }

  function doStop() {
    control('stop', async () => {
      applyPerformer(await api('/api/stop', {}, { timeoutMs: 6000 }));
    });
  }
  function doPauseToggle() {
    const paused = !!(S.snap.performer && S.snap.performer.paused);
    control('pause', async () => {
      applyPerformer(await api(paused ? '/api/resume' : '/api/pause', {}, { timeoutMs: 6000 }));
    });
  }
  function doTrackingToggle() {
    const enabled = !!(S.snap.performer && S.snap.performer.tracking_enabled);
    control('tracking', async () => {
      applyPerformer(await api('/api/tracking', { enabled: !enabled }));
    });
  }
  function doEndSession() {
    if (!window.confirm('セッションを終了しますか？（ログを閉じて開始画面に戻ります）')) return;
    control('end', async () => {
      const s = await api('/api/session/end', {}, { timeoutMs: 8000 });
      if (s) applySession(s);
      toast('info', 'セッションを終了しました');
      showScreen('start');
    });
  }
  function doRestToggle() {
    if (S.snap.resting) { doWake(); return; }
    if (!window.confirm('ロボットを休ませます（スリープ姿勢にしてモーターを OFF）。よろしいですか？')) return;
    control('rest', async () => {
      const r = await api('/api/robot/rest', {}, { timeoutMs: 20000 });
      if (r && typeof r.resting === 'boolean') S.snap.resting = r.resting;
      render();
    });
  }
  function doWake() {
    control('rest', async () => {
      toast('info', 'ロボットを起こしています…');
      const r = await api('/api/robot/wake', {}, { timeoutMs: 60000 });
      if (r && typeof r.resting === 'boolean') S.snap.resting = r.resting;
      render();
    });
  }
  function doPreflightRetry() {
    control('preflight', async () => {
      await api('/api/preflight', {});
      toast('info', '準備を再試行しています');
    });
  }

  // ================================================================== タイマー tick
  function tick() {
    if (!S.snap || S.screen !== 'control') return;
    const sess = S.snap.session || { active: false };
    const settings = S.snap.settings || {};
    const perf = S.snap.performer || {};
    const now = Date.now();

    // 本番タイマー
    const mt = $('#main-timer');
    const mb = $('#main-box');
    if (sess.active && sess.phase === 'main' && S.remainingAt) {
      const rem = Math.max(0, S.remainingAt.value - (now - S.remainingAt.at) / 1000);
      mt.textContent = fmtMMSS(rem);
      mb.classList.toggle('over', rem <= 0);
      mb.classList.toggle('warn', rem > 0 && rem <= 60);
    } else {
      mt.textContent = '--:--';
      mb.classList.remove('over', 'warn');
    }

    // 前回の声かけ
    const sb = $('#since-box');
    const st = $('#since-timer');
    if (sess.active && S.sinceAt) {
      const since = Math.max(0, S.sinceAt.value + (now - S.sinceAt.at) / 1000);
      st.textContent = `${Math.floor(since)} 秒`;
      const cue = Number(settings.session && settings.session.cue_interval_s) || 30;
      sb.classList.toggle('due', since >= cue);
    } else {
      st.textContent = '—';
      sb.classList.remove('due');
    }

    // 再生プログレス
    if (perf.status === 'playing' && perf.current && Number(perf.current.duration) > 0) {
      const p = clamp((serverNowSec() - Number(perf.current.started_at)) / Number(perf.current.duration), 0, 1);
      const bar = $('.btn-play.playing .pb-progress > i');
      if (bar) bar.style.width = `${(p * 100).toFixed(1)}%`;
    }
  }
  setInterval(() => { try { tick(); } catch (e) { console.error(e); } }, 250);

  // ================================================================== 設定画面
  function enterSettings() {
    if (!S.snap) { toast('warn', 'サーバーの状態を受信してから開いてください'); return; }
    S.draft = { settings: clone(S.snap.settings || {}), phrases: clone(S.snap.phrases || {}) };
    ensureSettingsShape(S.draft.settings);
    S.volumeDraft = S.snap.volume != null ? Number(S.snap.volume) : 50;
    showScreen('settings');
    buildSettingsForms();
    loadVoices(false);
    loadGestures();
  }
  function leaveSettings() {
    showScreen(sessionActive() ? 'control' : 'start');
  }
  function ensureSettingsShape(s) {
    s.robot = s.robot || {};
    s.names = s.names || {};
    s.session = s.session || {};
    s.positions = s.positions || {};
    s.tts = s.tts || {};
    s.tts.params = s.tts.params || {};
    s.tts.user_dict = Array.isArray(s.tts.user_dict) ? s.tts.user_dict : [];
    s.motion = s.motion || {};
    s.motion.envelope = s.motion.envelope || {};
    s.motion.idle = s.motion.idle || {};
    s.ui = s.ui || {};
  }
  function selectTab(name) {
    S.settingsTab = name;
    for (const t of TABS) {
      const p = $(`#tab-${t}`);
      if (p) p.hidden = t !== name;
    }
    for (const b of $$('#settings-tabs button')) b.classList.toggle('active', b.dataset.tab === name);
    window.scrollTo(0, 0);
  }
  function buildSettingsForms() {
    if (!S.draft) return;
    buildGeneralTab();
    buildPositionsTab();
    buildVoiceTab();
    buildScriptTab();
    buildMotionTab();
    selectTab(S.settingsTab || 'general');
  }
  /** 設定（settings.json 由来）のタブだけ作り直す。台本タブの編集内容は保持する。 */
  function rebuildSettingsTabs() {
    if (!S.draft) return;
    buildGeneralTab();
    buildPositionsTab();
    buildVoiceTab();
    buildMotionTab();
  }

  // ---- フォーム部品（draft のオブジェクトに直接バインド）
  function fieldRow(label, controlEl, hint) {
    return h('label', { class: 'field' }, h('span', { class: 'field-label' }, label), controlEl, hint ? h('small', { class: 'hint' }, hint) : null);
  }
  function bindText(obj, key, attrs) {
    const i = h('input', Object.assign({ type: 'text', autocapitalize: 'off', autocorrect: 'off', spellcheck: false }, attrs || {}));
    i.value = obj[key] == null ? '' : String(obj[key]);
    i.addEventListener('input', () => { obj[key] = i.value; });
    return i;
  }
  function bindNumber(obj, key, attrs, opts) {
    opts = opts || {};
    const i = h('input', Object.assign({ type: 'number', inputmode: 'decimal' }, attrs || {}));
    i.value = obj[key] == null ? '' : String(obj[key]);
    i.addEventListener('input', () => {
      const v = opts.int ? parseInt(i.value, 10) : parseFloat(i.value);
      if (!Number.isNaN(v)) obj[key] = v;
    });
    return i;
  }
  function bindCheck(obj, key, label) {
    const i = h('input', { type: 'checkbox' });
    i.checked = !!obj[key];
    i.addEventListener('change', () => { obj[key] = i.checked; });
    return h('label', { class: 'check' }, i, h('span', null, label));
  }
  function bindRange(obj, key, r, onChange) {
    const i = h('input', { type: 'range', min: r.min, max: r.max, step: r.step });
    const out = h('output');
    const v0 = typeof obj[key] === 'number' ? obj[key] : Number(r.min);
    i.value = String(v0);
    out.textContent = fmtNum(v0);
    i.addEventListener('input', () => {
      obj[key] = parseFloat(i.value);
      out.textContent = fmtNum(obj[key]);
      if (onChange) onChange(obj[key]);
    });
    return h('div', { class: 'range' }, i, out);
  }
  function gestureSelect(obj, key) {
    const names = ((S.snap && S.snap.gesture_names) || []).slice();
    const cur = obj[key] || '';
    if (cur && !names.includes(cur)) names.unshift(cur);
    const info = (S.snap && S.snap.gesture_info) || {};
    // 名前だけだと分かりにくいので「名前 — 日本語の説明」で表示する
    const sel = h('select', null, names.map((n) => h('option', { value: n }, info[n] ? `${n} — ${info[n]}` : n)));
    sel.value = cur || (names[0] || '');
    sel.addEventListener('change', () => { obj[key] = sel.value; });
    return sel;
  }
  function saveBar(fn, label) {
    return h('div', { class: 'save-bar' }, h('button', { type: 'button', class: 'btn btn-primary btn-lg', onclick: fn }, label || '保存'));
  }
  const row = (...els) => h('div', { class: 'row' }, els);

  // ---- 保存
  function saveSettings() {
    control('save', async () => {
      const r = await api('/api/settings', S.draft.settings, { method: 'PUT', timeoutMs: 8000 });
      toast('info', '設定を保存しました');
      for (const w of (r && r.warnings) || []) toast('warn', w);
      await refreshState();
      if (S.screen === 'settings' && S.draft) {
        S.draft.settings = clone(S.snap.settings || {});
        ensureSettingsShape(S.draft.settings);
        rebuildSettingsTabs();
      }
    });
  }
  function savePhrases() {
    control('save', async () => {
      const r = await api('/api/phrases', S.draft.phrases, { method: 'PUT', timeoutMs: 8000 });
      toast('info', '台本を保存しました');
      for (const w of (r && r.warnings) || []) toast('warn', w);
      await refreshState();
      if (S.screen === 'settings' && S.draft) {
        S.draft.phrases = clone(S.snap.phrases || {});
        buildScriptTab();
      }
    });
  }

  // ---- 一般
  function buildGeneralTab() {
    const d = S.draft.settings;
    setChildren($('#tab-general'), [
      h('div', { class: 'card' },
        h('h3', null, '名前'),
        fieldRow('ロボットの名前', bindText(d.names, 'robot'), '台本の {robot} に入ります'),
        fieldRow('実験者の名前', bindText(d.names, 'experimenter'), '台本の {experimenter} に入ります')),
      h('div', { class: 'card' },
        h('h3', null, 'ロボット接続'),
        fieldRow('ロボット URL', bindText(d.robot, 'base_url', { inputmode: 'url' }), '例: http://reachy-mini.local:8000'),
        row(fieldRow('接続タイムアウト（秒）', bindNumber(d.robot, 'connect_timeout_s', { min: 0.1, step: 0.1 })),
          fieldRow('読み取りタイムアウト（秒）', bindNumber(d.robot, 'read_timeout_s', { min: 0.1, step: 0.1 })))),
      h('div', { class: 'card' },
        h('h3', null, 'セッション'),
        row(fieldRow('本番の長さ（分）', bindNumber(d.session, 'main_minutes', { min: 0.5, step: 0.5 })),
          fieldRow('声かけ目安間隔（秒）', bindNumber(d.session, 'cue_interval_s', { min: 1, step: 1 }), 'この秒数を過ぎると「前回の声かけ」が黄色になります'),
          fieldRow('デバウンス（ms）', bindNumber(d.ui, 'debounce_ms', { min: 0, step: 50 }, { int: true }), '同じボタンの連打を無視する時間'))),
      saveBar(saveSettings),
    ]);
  }

  // ---- 位置
  function buildPositionsTab() {
    const pos = S.draft.settings.positions;
    const cards = Object.keys(pos).map((key) => {
      const p = pos[key];
      return h('div', { class: 'card' },
        h('h3', null, `${p.label || key} `, h('small', { class: 'muted' }, `(${key})`)),
        fieldRow('表示名', bindText(p, 'label')),
        row(fieldRow('yaw（度）', bindNumber(p, 'yaw_deg', { step: 1 })), fieldRow('pitch（度）', bindNumber(p, 'pitch_deg', { step: 1 }))),
        h('button', { type: 'button', class: 'btn', onclick: () => testPosition(key) }, 'テスト'),
        h('small', { class: 'hint muted' }, ' テストは保存済みの値で動きます'));
    });
    setChildren($('#tab-positions'), [
      h('p', { class: 'note' }, 'yaw は左が正・右が負（ロボットから見た向き）。pitch は下向きが正・上向きが負。'),
      cards,
      saveBar(saveSettings),
    ]);
  }
  function testPosition(key) {
    const name = key === 'up' ? 'look_up_nod' : `point_${key}`;
    control(`test-${key}`, async () => {
      await api('/api/gestures/test', { name }, { timeoutMs: 6000 });
    });
  }

  // ---- 声
  function buildVoiceTab() {
    const tts = S.draft.settings.tts;
    const p = tts.params;
    const a1 = S.snap.leaves && S.snap.leaves.A1 && S.snap.leaves.A1.text;
    if (S.previewText == null) S.previewText = a1 || 'こんにちは。わたしは{robot}だよ。';
    const previewInput = h('textarea', { rows: 2, id: 'preview-text' });
    previewInput.value = S.previewText;
    previewInput.addEventListener('input', () => { S.previewText = previewInput.value; });

    setChildren($('#tab-voice'), [
      h('div', { class: 'card' },
        h('h3', null, '声'),
        h('div', { id: 'voice-select-box' }),
        h('div', { id: 'voice-credit', class: 'muted' })),
      h('div', { class: 'card' },
        h('h3', null, '話し方'),
        fieldRow('話速', bindRange(p, 'speed', { min: 0.5, max: 2, step: 0.05 })),
        fieldRow('音高', bindRange(p, 'pitch', { min: -0.15, max: 0.15, step: 0.01 })),
        fieldRow('抑揚', bindRange(p, 'intonation', { min: 0, max: 2, step: 0.05 })),
        fieldRow('音量（合成）', bindRange(p, 'volume', { min: 0, max: 2, step: 0.05 })),
        fieldRow('末尾の余白（秒）', bindRange(p, 'post_phoneme', { min: 0, max: 1.5, step: 0.05 })),
        fieldRow('句読点の間（倍率）', bindRange(p, 'pause_scale', { min: 0.5, max: 2, step: 0.05 }))),
      h('div', { class: 'card' },
        h('h3', null, '試聴'),
        fieldRow('テキスト', previewInput, '{robot} {child} {experimenter} は名前に置き換わります。上の声・話し方の未保存の値で試聴できます。'),
        row(h('button', { type: 'button', class: 'btn', onclick: previewBrowser }, '試聴（ブラウザ）'),
          h('button', { type: 'button', class: 'btn', onclick: previewRobot }, '試聴（ロボット）')),
        h('audio', { id: 'preview-audio', controls: true, preload: 'none' })),
      h('div', { class: 'card' },
        h('h3', null, '名前の読み辞書'),
        h('p', { class: 'muted' }, '読み間違える名前を登録します。登録はすぐに反映・保存されます（保存ボタン不要）。'),
        h('div', { id: 'dict-list' }),
        dictForm()),
      saveBar(saveSettings),
    ]);
    renderVoiceSelect();
    renderDictList();
  }

  function renderVoiceSelect() {
    const box = $('#voice-select-box');
    if (!box || !S.draft) return;
    const tts = S.draft.settings.tts;
    if (Array.isArray(S.voices) && S.voices.length) {
      const groups = new Map();
      for (const v of S.voices) {
        const g = String(v.credit || '').replace(/^VOICEVOX:/, '') || String(v.name || '').split(' ')[0] || 'その他';
        if (!groups.has(g)) groups.set(g, []);
        groups.get(g).push(v);
      }
      const sel = h('select', { class: 'select-lg' });
      if (!S.voices.some((v) => String(v.id) === String(tts.voice_id))) {
        sel.append(h('option', { value: String(tts.voice_id) }, `（現在の設定: ${tts.voice_id}）`));
      }
      for (const [g, list] of groups) {
        sel.append(h('optgroup', { label: g }, list.map((v) => h('option', { value: String(v.id) }, v.name))));
      }
      sel.value = String(tts.voice_id);
      sel.addEventListener('change', () => {
        tts.voice_id = sel.value;
        renderVoiceCredit();
      });
      setChildren(box, fieldRow('VOICEVOX の声（キャラクター別）', sel));
    } else {
      const failed = S.voices === 'error';
      setChildren(box, [
        h('p', { class: 'muted' }, failed ? '声の一覧を取得できませんでした（VOICEVOX が起動していない可能性があります）。' : '声の一覧を読み込み中…', ' ',
          failed ? h('button', { type: 'button', class: 'btn btn-sm', onclick: () => loadVoices(true) }, '再読み込み') : null),
        fieldRow('声 ID（直接入力）', bindText(tts, 'voice_id', { inputmode: 'numeric' })),
      ]);
    }
    renderVoiceCredit();
  }
  function renderVoiceCredit() {
    const el = $('#voice-credit');
    if (!el || !S.draft) return;
    const id = String(S.draft.settings.tts.voice_id);
    const v = Array.isArray(S.voices) ? S.voices.find((x) => String(x.id) === id) : null;
    el.textContent = v ? `クレジット表記: ${v.credit || 'VOICEVOX'}（${v.name}）` : 'クレジット表記: VOICEVOX';
  }
  async function loadVoices(force) {
    if (Array.isArray(S.voices) && !force) { renderVoiceSelect(); renderStartCredit(); return; }
    S.voices = null;
    S.voicesRequested = true;
    renderVoiceSelect();
    try {
      const v = await api('/api/voices', undefined, { timeoutMs: 6000 });
      S.voices = Array.isArray(v) ? v : [];
    } catch (e) {
      S.voices = 'error';
    }
    renderVoiceSelect();
    renderStartCredit();
  }
  /** 開始画面のクレジット表示用に、VOICEVOX が使えるようになったら一度だけ取得する。 */
  function maybeLoadVoices() {
    if (S.voicesRequested) return;
    loadVoices(false).catch(() => {});
  }

  function currentPreviewBody(target) {
    const tts = S.draft.settings.tts;
    return { text: S.previewText || '', voice_id: String(tts.voice_id), params: tts.params, target };
  }
  function previewBrowser() {
    const audio = $('#preview-audio');
    // iPad Safari: ユーザー操作の同期コンテキストで一度 play() しておくと、非同期取得後の再生が許可される
    try {
      audio.src = SILENT_WAV;
      const p = audio.play();
      if (p && p.catch) p.catch(() => {});
    } catch (_) { /* ignore */ }
    control('preview', async () => {
      const res = await api('/api/tts/preview', currentPreviewBody('browser'), { timeoutMs: 20000, raw: true });
      const blob = await res.blob();
      if (audio._url) URL.revokeObjectURL(audio._url);
      audio._url = URL.createObjectURL(blob);
      audio.src = audio._url;
      try {
        await audio.play();
      } catch (e) {
        toast('warn', '自動再生できませんでした。プレーヤーの再生ボタンを押してください');
      }
    });
  }
  function previewRobot() {
    control('preview', async () => {
      const r = await api('/api/tts/preview', currentPreviewBody('robot'), { timeoutMs: 20000 });
      toast('info', `ロボットで再生しました（${r && r.duration != null ? fmtNum(r.duration) : '?'} 秒）`);
    });
  }

  function renderDictList() {
    const list = $('#dict-list');
    if (!list || !S.draft) return;
    const entries = S.draft.settings.tts.user_dict || [];
    setChildren(list, entries.length
      ? h('table', { class: 'dict' },
        h('thead', null, h('tr', null, h('th', null, '単語'), h('th', null, '読み'), h('th', null, 'アクセント核'))),
        h('tbody', null, entries.map((e) => h('tr', null, h('td', null, e.surface), h('td', null, e.pronunciation), h('td', null, String(e.accent_type))))))
      : h('p', { class: 'muted' }, '登録なし'));
  }
  function dictForm() {
    const sIn = h('input', { type: 'text', placeholder: '例: はるか', autocapitalize: 'off', autocorrect: 'off' });
    const pIn = h('input', { type: 'text', placeholder: '例: ハルカ（省略時は単語と同じ）', autocapitalize: 'off', autocorrect: 'off' });
    const aIn = h('input', { type: 'number', inputmode: 'numeric', min: 0, step: 1, value: '0' });
    const submit = () => control('dict', async () => {
      const surface = sIn.value.trim();
      if (!surface) { toast('warn', '単語を入力してください'); return; }
      const accent = parseInt(aIn.value, 10);
      const body = { surface, pronunciation: pIn.value.trim() || null, accent_type: Number.isNaN(accent) ? 0 : accent };
      const r = await api('/api/dict/word', body, { timeoutMs: 8000 });
      if (r && Array.isArray(r.entries)) {
        S.draft.settings.tts.user_dict = r.entries;
        renderDictList();
      }
      toast('info', `「${surface}」を登録しました`);
      sIn.value = '';
      pIn.value = '';
      aIn.value = '0';
      refreshState(); // サーバー側 settings と同期（後で設定を保存しても辞書が消えないように）
    });
    return h('div', { class: 'dict-form' },
      row(fieldRow('単語', sIn), fieldRow('読み（カタカナ）', pIn), fieldRow('アクセント核（0 = 平板）', aIn)),
      h('button', { type: 'button', class: 'btn', onclick: submit }, '辞書に登録'));
  }

  // ---- 台本
  function buildScriptTab() {
    const P = S.draft.phrases;
    const introCards = (P.intro || []).map((step) => {
      const scope = step.order && step.order.length ? `（${step.order.map((o) => ORDER_LABEL[o] || o).join('・')} のみ）` : '（両条件）';
      return h('div', { class: 'card' },
        h('h3', null, `${step.id} ${step.label || ''} `, h('small', { class: 'muted' }, scope)),
        fieldRow('見出し', bindText(step, 'label')),
        (step.parts && step.parts.length) ? step.parts.map((part) => phraseEditor(part, false)) : phraseEditor(step, false));
    });
    const cat = (key, title) => h('div', { class: 'card' }, h('h3', null, title), (P[key] || []).map((ph) => phraseEditor(ph, true)));
    setChildren($('#tab-script'), [
      h('p', { class: 'note' }, '「ロボットで再生」は保存済みの台本を再生します。編集後は「保存」を押してから試してください。名前の音声はセッション開始時に合成されます。'),
      h('h3', { class: 'section-title' }, 'イントロ'),
      introCards,
      cat('empathy', '励まし（共感）'),
      cat('logical', '励まし（論理）'),
      cat('backchannel', '相づち'),
      saveBar(savePhrases),
    ]);
  }
  function phraseEditor(obj, withTrigger) {
    const ta = h('textarea', { rows: 2 });
    ta.value = obj.text == null ? '' : String(obj.text);
    ta.addEventListener('input', () => { obj.text = ta.value; });
    return h('div', { class: 'phrase-row' },
      h('div', { class: 'phrase-head' },
        h('span', { class: 'pb-num' }, obj.id),
        withTrigger ? bindText(obj, 'trigger', { placeholder: '場面（トリガー）', class: 'trigger-input' }) : null,
        h('button', { type: 'button', class: 'btn btn-sm', onclick: () => testPhrase(obj.id) }, 'ロボットで再生')),
      ta,
      fieldRow('ジェスチャー', gestureSelect(obj, 'gesture')));
  }
  function testPhrase(id) {
    control(`test-${id}`, async () => {
      const r = await api('/api/phrases/test', { id }, { timeoutMs: 10000 });
      if (r && r.accepted === false && r.reason !== 'debounce') toast('warn', `再生できませんでした: ${r.reason || ''}`);
    });
  }

  // ---- 動作
  function buildMotionTab() {
    const m = S.draft.settings.motion;
    const idle = m.idle;
    const env = m.envelope;
    const ta = h('textarea', { class: 'json', id: 'gestures-json', rows: 18, spellcheck: false, placeholder: '読み込み中…' });
    ta.value = S.gesturesText == null ? '' : S.gesturesText;
    ta.addEventListener('input', () => { S.gesturesText = ta.value; });

    const volOut = h('output', null, String(S.volumeDraft));
    const vol = h('input', { type: 'range', min: 0, max: 100, step: 1 });
    vol.value = String(S.volumeDraft);
    vol.addEventListener('input', () => {
      S.volumeDraft = parseInt(vol.value, 10);
      volOut.textContent = String(S.volumeDraft);
    });

    setChildren($('#tab-motion'), [
      h('div', { class: 'card' },
        h('h3', null, '顔追跡・頭揺れ'),
        bindCheck(m, 'tracking_enabled', '顔追跡を有効にする（操作画面のトグルと同じ）'),
        fieldRow('追跡の強さ（0–1）', bindRange(m, 'tracking_weight', { min: 0, max: 1, step: 0.05 })),
        bindCheck(m, 'wobbling_enabled', '頭揺れ（wobbling）を有効にする')),
      h('div', { class: 'card' },
        h('h3', null, 'アイドル動作'),
        bindCheck(idle, 'enabled', '待機中に小さな動作をする'),
        row(fieldRow('間隔（秒）', bindNumber(idle, 'interval_s', { min: 2, step: 1 })), fieldRow('ゆらぎ（秒）', bindNumber(idle, 'jitter_s', { min: 0, step: 1 }))),
        row(fieldRow('顔追跡 ON のときの動作', gestureSelect(idle, 'gesture_tracking_on')), fieldRow('顔追跡 OFF のときの動作', gestureSelect(idle, 'gesture_tracking_off')))),
      h('div', { class: 'card' },
        h('h3', null, '動作タイミング'),
        row(fieldRow('音声の先行（ms）', bindNumber(m, 'audio_lead_ms', { min: 0, step: 10 }), '音声開始から動作開始までの遅れ'),
          fieldRow('送信レート（Hz）', bindNumber(m, 'stream_hz', { min: 5, max: 100, step: 5 })),
          fieldRow('立ち上がり（秒）', bindNumber(m, 'ramp_in_s', { min: 0, step: 0.1 })),
          fieldRow('戻り（秒）', bindNumber(m, 'ramp_out_s', { min: 0, step: 0.1 })))),
      h('div', { class: 'card' },
        h('h3', null, '可動範囲の上限（エンベロープ）'),
        row(fieldRow('roll（度）', bindNumber(env, 'roll_deg', { min: 0, step: 1 })),
          fieldRow('pitch（度）', bindNumber(env, 'pitch_deg', { min: 0, step: 1 })),
          fieldRow('yaw（度）', bindNumber(env, 'yaw_deg', { min: 0, step: 1 })),
          fieldRow('並進（mm）', bindNumber(env, 'xyz_mm', { min: 0, step: 1 })),
          fieldRow('アンテナ（度）', bindNumber(env, 'antenna_deg', { min: 0, step: 5 })))),
      saveBar(saveSettings),
      h('div', { class: 'card' },
        h('h3', null, 'ジェスチャー定義（JSON）'),
        h('p', { class: 'muted' }, 'name → 定義（kind: keyframes / recorded / point / sequence）。台本で使用中のジェスチャーは削除できません。'),
        ta,
        row(h('button', { type: 'button', class: 'btn btn-primary', onclick: saveGestures }, '検証して保存'),
          h('button', { type: 'button', class: 'btn', onclick: () => loadGestures(true) }, '読み直す'))),
      h('div', { class: 'card' },
        h('h3', null, 'ロボットの音量'),
        h('div', { class: 'range' }, vol, volOut),
        h('button', { type: 'button', class: 'btn', onclick: applyVolume }, '適用（テスト音が鳴ります）')),
    ]);
  }
  async function loadGestures(force) {
    if (S.gesturesText != null && !force) return;
    try {
      const g = await api('/api/gestures', undefined, { timeoutMs: 6000 });
      S.gesturesText = JSON.stringify(g, null, 2);
    } catch (e) {
      toast('error', `ジェスチャー定義を読み込めません: ${e.message}`);
      return;
    }
    const ta = $('#gestures-json');
    if (ta) ta.value = S.gesturesText;
  }
  function saveGestures() {
    let parsed;
    try {
      parsed = JSON.parse(S.gesturesText || '');
    } catch (e) {
      toast('error', `JSON が正しくありません: ${e.message}`);
      return;
    }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      toast('error', 'JSON はジェスチャー名をキーにしたオブジェクトである必要があります');
      return;
    }
    control('save', async () => {
      const r = await api('/api/gestures', parsed, { method: 'PUT', timeoutMs: 8000 });
      toast('info', 'ジェスチャー定義を保存しました');
      for (const w of (r && r.warnings) || []) toast('warn', w);
      await loadGestures(true);
      await refreshState();
    });
  }
  function applyVolume() {
    const v = clamp(Number(S.volumeDraft) || 0, 0, 100);
    if (!window.confirm(`ロボットの音量を ${v} にします。テスト音が鳴ります。よろしいですか？`)) return;
    control('volume', async () => {
      const r = await api('/api/volume', { volume: v }, { timeoutMs: 8000 });
      toast('info', `音量を ${r && r.volume != null ? r.volume : v} に設定しました`);
    });
  }

  // ================================================================== 初期化
  function bindStatic() {
    // 開始画面
    $('#start-form').addEventListener('submit', (e) => { startSession(e).catch((err) => toast('error', String(err))); });
    $('#in-child-name').addEventListener('input', () => validateName(false));
    $('#btn-preflight-retry').addEventListener('click', doPreflightRetry);
    $('#link-settings-start').addEventListener('click', enterSettings);
    for (const seg of $$('.seg')) {
      seg.addEventListener('click', (e) => {
        const b = e.target.closest('button[data-value]');
        if (!b || !seg.contains(b)) return;
        for (const x of seg.querySelectorAll('button')) x.classList.toggle('active', x === b);
      });
    }
    // 操作画面
    $('#btn-stop').addEventListener('click', doStop);
    $('#btn-stop-float').addEventListener('click', doStop);
    $('#btn-pause').addEventListener('click', doPauseToggle);
    $('#btn-tracking').addEventListener('click', doTrackingToggle);
    $('#btn-end').addEventListener('click', doEndSession);
    $('#btn-rest').addEventListener('click', doRestToggle);
    $('#btn-wake-overlay').addEventListener('click', doWake);
    $('#btn-phase-baseline').addEventListener('click', () => setPhase('baseline'));
    $('#btn-phase-main').addEventListener('click', () => setPhase('main'));
    $('#link-settings-control').addEventListener('click', enterSettings);
    // 設定画面
    $('#link-back').addEventListener('click', leaveSettings);
    $('#settings-tabs').addEventListener('click', (e) => {
      const b = e.target.closest('button[data-tab]');
      if (b) selectTab(b.dataset.tab);
    });
    // キーボード: Escape でストップ（操作画面）
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && S.screen === 'control') {
        e.preventDefault();
        doStop();
      }
    });
    // 復帰時は状態を取り直す
    document.addEventListener('visibilitychange', () => {
      if (document.visibilityState === 'visible') {
        refreshState();
        if (!S.es) connectSSE();
      }
    });
    window.addEventListener('online', () => {
      refreshState();
      connectSSE();
    });
    window.addEventListener('pageshow', (e) => { if (e.persisted) { refreshState(); connectSSE(); } });
    // 想定外の例外は握りつぶさず通知だけする
    window.addEventListener('error', (e) => {
      console.error(e.error || e.message);
      toast('error', '画面内でエラーが発生しました（再読み込みで直ることがあります）');
    });
    window.addEventListener('unhandledrejection', (e) => {
      console.error(e.reason);
      const msg = e.reason && e.reason.message ? e.reason.message : '処理に失敗しました';
      toast('error', msg);
      e.preventDefault();
    });
  }

  function init() {
    try {
      bindStatic();
    } catch (e) {
      console.error('bindStatic failed', e);
    }
    connectSSE();
    // SSE の snapshot が届かない場合の保険（同時に走っても applySnapshot は冪等）
    setTimeout(() => { if (!S.snap) refreshState(); }, 1500);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
