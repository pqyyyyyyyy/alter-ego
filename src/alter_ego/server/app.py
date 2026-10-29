"""FastAPI + WebSocket 后端。把各模块串起来并驱动前端。

人格与对话链路已接入；感知和在线 TTS 按环境变量可选开启。

设计要点：**前端不做任何业务判断，只听指令渲染。**
所有逻辑在后端，这样前端可独立开发（用 mock 驱动），
而后端换实现不影响前端。
"""
from __future__ import annotations

import asyncio
import base64
import logging
import time
import uuid
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..contracts import EMOTIONS, DialogTurn, Emotion, PersonaContext, State
from ..dialog import DialogOrchestrator
from ..llm.providers import ProviderRegistry
from ..persona.card import PersonaCard
from ..persona.examples import ExamplePool
from ..persona.rewriter import LocalRewriter
from .lifecycle import Lifecycle, synthesize

_log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
WEB_DIR = ROOT / "web"
CHARACTER = ROOT / "characters" / "example"      # 可用环境变量 ALTER_EGO_CHARACTER 覆盖


class Hub:
    """WebSocket 连接池。广播失败的连接自动摘除。"""

    def __init__(self) -> None:
        self._clients: dict[WebSocket, str] = {}
        self.on_disconnect = None
        self.join_lock = asyncio.Lock()

    async def join(self, ws: WebSocket, state='IDLE') -> str:
        await ws.accept()
        client_id = uuid.uuid4().hex
        async with self.join_lock:
            owner = self.owner or client_id
            await ws.send_json({'type': 'hello', 'protocol': 'aeg-ws-v2', 'client_id': client_id,
                                'playback_owner': client_id == owner})
            await ws.send_json({'type': 'sync', 'state': state, 'playback_owner': client_id == owner})
            self._clients[ws] = client_id
            if self.owner == client_id and owner != client_id:
                await ws.send_json({'type': 'hello', 'protocol': 'aeg-ws-v2', 'client_id': client_id,
                                    'playback_owner': True})
        _log.info("前端已连接，当前 %d 个客户端", len(self._clients))
        return client_id

    async def leave(self, ws: WebSocket) -> None:
        was_owner = self._clients.get(ws) == self.owner
        client_id = self._clients.pop(ws, None)
        if client_id and self.on_disconnect:
            await self.on_disconnect(client_id)
        if was_owner:
            for client, cid in list(self._clients.items()):
                try:
                    await asyncio.wait_for(client.send_json({'type': 'hello', 'protocol': 'aeg-ws-v2',
                        'client_id': cid, 'playback_owner': cid == self.owner}), timeout=1.)
                except Exception:
                    await self.leave(client)

    @property
    def client_ids(self): return list(self._clients.values())

    @property
    def owner(self): return next(iter(self._clients.values()), None)

    def has(self, client_id): return client_id is not None and client_id in self._clients.values()

    async def close_client(self, client_id):
        for ws, cid in list(self._clients.items()):
            if cid == client_id:
                try: await asyncio.wait_for(ws.close(code=1012), timeout=1.)
                except Exception: pass
                await self.leave(ws)

    async def send(self, msg: dict) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            try:
                await asyncio.wait_for(ws.send_json(msg), timeout=1.)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.close_client(self._clients.get(ws))

    @property
    def count(self) -> int:
        return len(self._clients)


class Runtime:
    """运行时状态与各模块句柄。"""

    def __init__(self) -> None:
        self.hub = Hub()
        self.state: State = "IDLE"
        self.card: PersonaCard | None = None
        self.char_dir: Path | None = None
        self.rewriter: LocalRewriter | None = None
        self.registry: ProviderRegistry | None = None
        self.dialog: DialogOrchestrator | None = None
        self.dialog_error: str = "未加载"
        self.warmup_sec: float = -1.0
        self.tts: object | None = None
        self.tts_error: str = "未启用"

        # 感知链路。摄像头是隐私敏感设备，默认不开 ——
        # 见 lifespan 里 ALTER_EGO_PERCEPTION 的说明。
        self.camera: object | None = None
        self.pipeline: object | None = None
        self.perception_task: asyncio.Task | None = None
        self.perception_error: str = "未启用"
        self.camera_view: dict | None = None
        self.camera_monitor = None

        # 感知层给人格层的上下文。这两个字段是两层之间唯一的耦合点 ——
        # 感知只管写，人格只管读，中间不共享对象。
        self.current_user: str | None = None
        self.current_user_id: str | None = None
        self.visual_hint: str | None = None
        self.lifecycle: Lifecycle | None = None
        self.fixture = False
        self.perception_stopping = False

        # 立绘轮换计数。同一情绪连续出现时换候选图，视觉上不单调。
        self._sprite_turn: int = 0

    async def set_state(self, state: State) -> None:
        if state == self.state:
            return
        prev, self.state = self.state, state
        await self.hub.send({"type": "state", "value": state})
        await self._send_state_animation(prev, state)

    async def _send_state_animation(self, prev: State, state: State) -> None:
        """状态切换时推对应的立绘动画。

        动画与情绪是两条独立消息 —— 这样「待机时睡着」「唤醒时打哈欠」
        不必塞进 Emotion 枚举，云端也就不会在「情绪」里挑「动作」。
        """
        if self.card is None:
            return
        # IDLE→NOTICED 用唤醒序列（睡着→打哈欠→擦眼睛），
        # 其他路径进 NOTICED 不播 —— 她本来就醒着，再打个哈欠很怪。
        key = {
            "IDLE": "idle",
            "NOTICED": "wake" if prev == "IDLE" else None,
            "DIALOG": "dialog",
        }.get(state)
        if not key:
            return
        anim = self.card.animation(key)
        if anim is None or not anim.frames:
            return
        await self.hub.send({"type": "animation", "name": key, **anim.to_dict()})

    def available_sprites(self, emotion: str) -> list[str]:
        if self.card is None or self.char_dir is None:
            return []
        sprite_dir = (self.char_dir / "sprites" / "aligned").resolve()
        available = []
        for name in self.card.sprites_for(emotion):
            path = (sprite_dir / name).resolve()
            if sprite_dir in path.parents and path.is_file():
                available.append(name)
        return available

    async def send_emotion(self, emotion: str) -> str | None:
        """推情绪 + 该情绪对应的立绘文件名。

        **文件名由后端给，前端不要自己按情绪名拼。** 一个情绪可以有多张
        候选图，轮换逻辑在后端（见 card.sprite_for）—— 前端不做业务判断。
        """
        payload: dict = {"type": "emotion", "value": emotion}
        sprites = self.available_sprites(emotion)
        if sprites:
            payload["sprite"] = sprites[self._sprite_turn % len(sprites)]
            self._sprite_turn += 1
        await self.hub.send(payload)
        return payload.get("sprite")

    def persona_context(self) -> PersonaContext:
        """用当前感知状态拼出改写上下文。"""
        return PersonaContext(
            user_name=self.current_user,
            visual_hint=self.visual_hint,
        )


rt = Runtime()


# ==================== 感知链路 ====================

def _start_perception() -> str:
    """同步地建后端、摄像头与管线。返回空串表示成功，否则返回错误说明。

    建后端要加载两个 ONNX 模型（约 0.6 s），开摄像头要 1-2 s，
    都是阻塞的，所以在 lifespan 里用 to_thread 调。
    """
    import os

    from ..perception import Camera, FileIdentityStore, PerceptionPipeline
    from ..perception.backends.insightface_backend import InsightFaceBackend

    # 感知只用 3 线程。板子 4 核要分给 ASR 与 TTS，
    # 而实测 3 线程的检测已是 2.68 ms（PC，320×320），再快没有意义。
    backend = InsightFaceBackend(
        det_size=(320, 320),
        num_threads=int(os.environ.get("ALTER_EGO_PERCEPTION_THREADS", "3")),
    )
    store = FileIdentityStore(ROOT / "data" / "users")
    camera = Camera(int(os.environ.get("ALTER_EGO_CAMERA", "0")))
    camera.open()

    rt.camera = camera
    rt.pipeline = PerceptionPipeline(backend, store)
    return ""


async def _perception_loop() -> None:
    """每帧一次 step，把事件翻译成状态切换与前端消息。

    `pipeline.step` 是阻塞的 CPU 工作（ONNX 前向），必须丢到线程里，
    否则会卡住事件循环 —— 表现是 WebSocket 消息延迟、前端口型卡顿，
    而不是报错。

    **已量化的代价**（2026-10-04 实测，320×320 / 3 线程，PC）：

        主线程连续调用              3.82 ms
        主线程 + 0.5 s 间隔          3.69 ms
        to_thread 连续              5.62 ms
        to_thread + 0.5 s 间隔       7.01 ms   ← 本循环的跑法

    即 `to_thread` 的每帧派发开销约 3.2 ms，相对 3.8 ms 的本体几乎翻倍。
    记下来是因为：后端里看到的检测耗时（/debug/perception 报 6.87 ms）
    会显著高于 bench_perception.py 的数字，不写明会被当成「哪里慢了」去查。

    暂不优化。5 fps 下总开销 35 ms/s，约 3.5% 单核，不值得为此改架构。
    若板子上吃紧，解法是把循环放进一个长驻线程（每个事件一次交接，
    而不是每帧一次），不是去调 to_thread。
    """
    pipe = rt.pipeline
    cam = rt.camera
    assert pipe is not None and cam is not None

    while not rt.perception_stopping:
        try:
            frame = await asyncio.to_thread(cam.read)
            budget = rt.lifecycle.coordinator.budget() if rt.lifecycle else None
            events = await asyncio.to_thread(pipe.step, frame,
                                              allow_identity=budget.identity_extract if budget else True)
            rt.visual_hint = pipe.snapshot.visual_hint          # type: ignore[attr-defined]
            for ev in events:
                await _on_perception_event(ev)
            budget = rt.lifecycle.coordinator.budget() if rt.lifecycle else None
            fps = min(pipe.target_fps, budget.presence_fps) if budget else pipe.target_fps
            await asyncio.sleep(1.0 / fps)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # 感知出错不能把整个服务带下去 —— 角色还能听能说。
            # 但必须留下日志并记进 perception_error，/health 能查到。
            rt.perception_error = f"{type(exc).__name__}: {exc}"
            _log.warning("感知循环出错，2 秒后重试: %s", rt.perception_error)
            await asyncio.sleep(2.0)


async def _on_perception_event(ev: object) -> None:
    """事件 → 状态机 + 前端消息。

    这里是感知链路驱动交互的唯一入口。前端不做任何判断，只渲染。
    """
    kind = ev.kind                                           # type: ignore[attr-defined]

    if kind == "APPEARED":
        if rt.lifecycle:
            rt.lifecycle.coordinator.present = True
            await rt.lifecycle.identify(new_guest=True)
            await rt.lifecycle.policy()
        # IDLE → NOTICED。此时还不知道是谁，身份随后由 IDENTIFIED 补上。
        # 先切状态是故意的：让角色「先抬头看你」再「认出你」，
        # 比等 1 秒投票完才有反应自然得多。
        if rt.state == "IDLE":
            await rt.set_state("NOTICED")
        await rt.hub.send({"type": "presence", "value": True,
                           "faces": ev.n_faces})             # type: ignore[attr-defined]

    elif kind == "LEFT":
        if rt.lifecycle: await rt.lifecycle.leave()
        rt.current_user = None
        await rt.hub.send({"type": "presence", "value": False, "faces": 0})
        await rt.hub.send({"type": "user", "name": None, "known": False})
        # SPEAKING 时人走了不打断 —— 让它把话说完，由 audio_ended 收尾。
        # 话说一半硬切回 IDLE 会留下一个张着嘴的立绘。
        if rt.lifecycle is None and rt.state != "SPEAKING":
            await rt.set_state("IDLE")

    elif kind == "IDENTIFIED":
        ident = ev.identity                                  # type: ignore[attr-defined]
        if rt.lifecycle:
            await rt.lifecycle.identify(ident.user_id if ident else None, ident.name if ident else None)
        rt.current_user = ident.name if ident else None
        await rt.hub.send({
            "type": "user",
            "name": rt.current_user,
            "known": bool(rt.current_user),
            "similarity": round(ident.similarity, 3) if ident else 0.0,
        })

    elif kind == "COUNT_CHANGED":
        if rt.lifecycle:
            await rt.lifecycle.identify(new_guest=True)
        await rt.hub.send({"type": "presence", "value": True,
                           "faces": ev.n_faces})             # type: ignore[attr-defined]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载角色并预热改写模型。

    预热是必需的，不是优化 —— 冷启动首次改写 PC 实测 0.52-1.38 s，
    板子上推算 3 秒以上。不能让这 3 秒发生在用户第一次说话时。
    """
    import os
    if os.environ.get('ALTER_EGO_FIXTURE', '0') == '1':
        from .fixture import configure
        configure(rt, ROOT)
        await _setup_lifecycle()
        await rt.lifecycle.enable_asr(rt.fixture_control.asr)
        rt.lifecycle.status['perception'] = {'status': 'ready', 'reason': 'fixture_events_no_camera'}
        try:
            yield
        finally:
            await _shutdown()
        return
    char_dir = Path(os.environ.get("ALTER_EGO_CHARACTER", str(CHARACTER)))
    rt.char_dir = char_dir
    try:
        rt.card = PersonaCard.load(char_dir / "persona.yaml")
        # 带 sprite_dir 校验：映射里写错文件名不报错，只会静默兜底成 neutral，
        # 表现为「某个表情从来不出现」。启动时就该把它喊出来。
        for w in rt.card.validate(sprite_dir=char_dir / "sprites" / "aligned"):
            _log.warning("人格卡: %s", w)
        pool = ExamplePool.load(char_dir / "examples.jsonl")
        rt.registry = ProviderRegistry(str(ROOT / "providers.yaml"))
        rt.rewriter = LocalRewriter(rt.card, pool, rt.registry.get("local"))
        rt.dialog = DialogOrchestrator(rt.card, rt.rewriter, rt.registry)
        rt.dialog_error = ""
        _log.info("角色 %s 已加载，示例池 %d 条 %s",
                  rt.card.name, len(pool), pool.situations())
        _log.info("对话编排器就绪，云端降级链 %s", rt.registry.dialog_chain)

        rt.warmup_sec = await asyncio.to_thread(rt.rewriter.warmup)
        if rt.rewriter.last_error:
            _log.warning("预热时改写失败（llama-server 没起？）: %s",
                         rt.rewriter.last_error)
        else:
            _log.info("改写模型预热完成，耗时 %.2fs", rt.warmup_sec)
    except Exception as exc:
        # 加载失败不阻止服务启动 —— 前端联调不需要人格层
        rt.dialog_error = f"{type(exc).__name__}: {exc}"
        _log.error("角色加载失败，/debug/say 与 /debug/chat 将不可用: %s", exc)

    # 感知链路默认**不开**，要显式 ALTER_EGO_PERCEPTION=1。
    # 理由不是技术而是礼貌与隐私：z 和 w 在自己机器上起后端调前端时，
    # 不该被莫名打开摄像头。板子上的启动脚本里写死开启即可。
    if os.environ.get("ALTER_EGO_PERCEPTION", "0") == "1" and not os.environ.get('ALTER_EGO_CAMERA_URL'):
        try:
            err = await asyncio.to_thread(_start_perception)
            if err:
                rt.perception_error = err
                _log.error("感知链路启动失败: %s", err)
            else:
                rt.perception_error = ""
                rt.perception_task = asyncio.create_task(_perception_loop())
                _log.info("感知链路已启动（摄像头已打开）")
        except Exception as exc:
            rt.perception_error = f"{type(exc).__name__}: {exc}"
            _log.error("感知链路启动失败，服务继续: %s", rt.perception_error)
    else:
        _log.info("感知链路未启用（设 ALTER_EGO_PERCEPTION=1 开启）")

    if os.environ.get("ALTER_EGO_TTS", "0") == "edge":
        try:
            from ..voice.edge_tts import EdgeTTS

            rt.tts = EdgeTTS(os.environ.get("ALTER_EGO_TTS_VOICE", "ja-JP-NanamiNeural"))
            rt.tts_error = ""
            _log.info("在线日语 TTS 已启用")
        except Exception as exc:
            rt.tts_error = f"{type(exc).__name__}: {exc}"
            _log.error("在线 TTS 加载失败，保留字幕: %s", rt.tts_error)
    elif os.environ.get("ALTER_EGO_TTS", "0") == "gpt-sovits":
        try:
            from ..voice.gpt_sovits import RemoteGPTSoVITS

            rt.tts = RemoteGPTSoVITS(
                os.environ["ALTER_EGO_TTS_URL"],
                timeout=float(os.environ.get("ALTER_EGO_TTS_TIMEOUT", "90")),
            )
            rt.tts_error = ""
            _log.info("GPT-SoVITS HTTP 客户端已启用；连接在首次合成时验证")
        except Exception as exc:
            rt.tts_error = f"{type(exc).__name__}: {exc}"
            _log.error("GPT-SoVITS 客户端初始化失败，保留字幕: %s", rt.tts_error)

    await _setup_lifecycle()
    if os.environ.get('ALTER_EGO_ASR', '0') == '1':
        rt.lifecycle.status['asr'] = {'status': 'loading', 'reason': ''}
        try:
            from ..voice.asr import SenseVoiceASR
            model_dir = os.environ.get('ALTER_EGO_ASR_MODEL', '')
            if not model_dir: raise ValueError('ASR model directory not configured')
            asr = await asyncio.to_thread(SenseVoiceASR, model_dir,
                                         int(os.environ.get('ALTER_EGO_ASR_THREADS', '2')))
            await rt.lifecycle.enable_asr(asr)
        except Exception as exc:
            rt.lifecycle.status['asr'] = {'status': 'failed', 'reason': type(exc).__name__}
    camera_url = os.environ.get('ALTER_EGO_CAMERA_URL')
    if camera_url:
        from .camera_bridge import CameraMonitor
        monitor = CameraMonitor(rt, camera_url)
        rt.camera_monitor = monitor
        await monitor.poll_once()
        rt.perception_task = asyncio.create_task(monitor.run())
    try:
        yield
    finally:
        await _shutdown()


async def _setup_lifecycle():
    class UnavailableDialog:
        history_turns = 0
        def reply(self, *args, **kwargs): raise RuntimeError('dialog_not_loaded')
    factory = lambda: (DialogOrchestrator(rt.card, rt.rewriter, rt.registry)
                       if rt.dialog is not None else UnavailableDialog())
    rt.lifecycle = Lifecycle(rt, factory,
        model_timeout=float(os.environ.get('ALTER_EGO_MODEL_TIMEOUT', '120')),
        playback_timeout=float(os.environ.get('ALTER_EGO_PLAYBACK_TIMEOUT', '240')),
        stop_timeout=float(os.environ.get('ALTER_EGO_STOP_TIMEOUT', '3')),
        baseline=os.environ.get('ALTER_EGO_SCHED_BASELINE', '0') == '1',
        device_id=os.environ.get('ALTER_EGO_DEVICE_ID', 'pc-fixture' if rt.fixture else 'unspecified'))
    rt.hub.on_disconnect = rt.lifecycle.disconnect
    for name, handle, error in (('dialog', rt.dialog, rt.dialog_error),
                                ('tts', rt.tts, rt.tts_error),
                                ('perception', rt.pipeline, rt.perception_error)):
        rt.lifecycle.status[name] = {'status': 'loaded' if handle is not None else
                                    ('disabled' if error == '未启用' else 'failed'),
                                    'reason': 'connection_not_checked' if handle is not None else error}
    await rt.lifecycle.start()


async def _shutdown():
    rt.perception_stopping = True
    if rt.lifecycle: await rt.lifecycle.close()
    if rt.perception_task is not None:
        # Do not cancel a read/step thread and then close its device underneath it.
        await rt.perception_task
    if rt.camera is not None:
        rt.camera.close()                        # type: ignore[attr-defined]
    _log.info("服务关闭")


app = FastAPI(title="Alter-Ego", lifespan=lifespan)

if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/")
async def index():
    idx = WEB_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return {"hint": "web/index.html 还没建，w 正在做"}


@app.post('/camera/control')
async def camera_control(request: Request):
    if request.headers.get('X-Alter-Ego-Camera') != '1':
        raise HTTPException(403, 'camera_control_header_required')
    origin = request.headers.get('origin')
    if origin and origin != str(request.base_url).rstrip('/'):
        raise HTTPException(403, 'camera_control_same_origin_required')
    try:
        body = await request.json()
        if not isinstance(body, dict) or set(body) != {'enabled'} or type(body['enabled']) is not bool:
            raise ValueError('invalid enabled')
    except (ValueError, TypeError):
        raise HTTPException(400, 'camera_enabled_boolean_required')
    if rt.camera_monitor is None:
        raise HTTPException(409, 'camera_control_unavailable')
    try:
        view = await rt.camera_monitor.set_enabled(body['enabled'])
    except Exception:
        return {'ok': False, 'error': 'camera_control_failed'}
    return {'ok': True, 'camera': view}


@app.get('/voice/control')
async def voice_status():
    return {'ok': True, 'voice': rt.lifecycle.voice_output.snapshot()}


@app.post('/voice/control')
async def voice_control(request: Request):
    if request.headers.get('X-Alter-Ego-Voice') != '1':
        raise HTTPException(403, 'voice_control_header_required')
    origin = request.headers.get('origin')
    if origin and origin != str(request.base_url).rstrip('/'):
        raise HTTPException(403, 'voice_control_same_origin_required')
    try:
        body = await request.json()
        if (not isinstance(body, dict) or set(body) != {'enabled', 'mode'} or
                type(body['enabled']) is not bool or body['mode'] not in ('fast', 'sync')):
            raise ValueError('invalid_voice_control')
    except (ValueError, TypeError):
        raise HTTPException(400, 'voice_enabled_boolean_and_mode_required')
    return await rt.lifecycle.voice_output.control(body['enabled'], body['mode'])


@app.get("/character/sprites/{path:path}")
async def character_sprite(path: str):
    """只提供当前角色的已对齐立绘，避免前端直接暴露角色目录。

    角色目录由 ``ALTER_EGO_CHARACTER`` 决定，不能在模块导入时固定；
    因此这里在 lifespan 加载角色后再解析路径，并严格限制在
    ``sprites/aligned`` 目录内，拒绝路径穿越和人格卡等非素材文件。
    """
    if rt.char_dir is None:
        raise HTTPException(status_code=503, detail="角色尚未加载")
    sprite_dir = (rt.char_dir / "sprites" / "aligned").resolve()
    target = (sprite_dir / path).resolve()
    if sprite_dir not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="立绘不存在")
    return FileResponse(str(target))


@app.get("/health")
async def health():
    return {
        **(rt.lifecycle.health() if rt.lifecycle else {}),
        'supervised': os.environ.get('ALTER_EGO_SUPERVISED', '0') == '1',
        "state": rt.state,
        "clients": rt.hub.count,
        "character": rt.card.name if rt.card else None,
        "rewriter_ready": rt.rewriter is not None,
        "warmup_sec": round(rt.warmup_sec, 3),
        "last_rewrite_error": rt.rewriter.last_error if rt.rewriter else "未加载",
        "dialog_ready": rt.dialog is not None,
        "dialog_chain": rt.registry.dialog_chain if rt.registry else None,
        "last_dialog_error": rt.dialog_error,
        "history_turns": (rt.lifecycle.sessions.entries[rt.lifecycle.sessions.current][0].history_turns
                          if rt.lifecycle and rt.lifecycle.sessions.current in rt.lifecycle.sessions.entries else 0),
        "tts_loaded": rt.tts is not None,
        "tts_ready": bool(rt.lifecycle and rt.lifecycle.status['tts']['status'] in ('ready', 'running')),
        "last_tts_error": rt.tts_error,
        "perception_ready": bool(rt.pipeline is not None or rt.camera_view and rt.camera_view['available']),
        "perception_error": rt.perception_error,
        "camera_view": rt.camera_view,
        "current_user": rt.current_user,
        "visual_hint": rt.visual_hint,
    }


@app.get("/debug/perception")
async def debug_perception():
    """感知链路诊断。认不准人时先看这里，再去调阈值。

    看什么：
      · detect_mean_ms  —— 检测耗时，PC 参考值 2.7 ms（320×320 / 3 线程）
      · present / n_faces —— 在场判定是否符合画面实际
      · similarity      —— 低于阈值就会判访客。若本人一直在 0.3 左右，
        先去 data/aligned_check/ 看对齐图，不要先改阈值
    """
    if rt.camera_view is not None:
        return {"ok": rt.camera_view['available'], **rt.camera_view,
                "identity_scope": "visible A/B only; no speaker or memory binding"}
    if rt.pipeline is None:
        return {"ok": False, "error": rt.perception_error,
                "hint": "设 ALTER_EGO_PERCEPTION=1 重启后端"}
    return {"ok": True, **rt.pipeline.stats()}   # type: ignore[attr-defined]


@app.get("/debug/users")
async def debug_users():
    """列出已注册用户。只返回姓名与 id，不返回特征向量。

    注册与删除只走 CLI（scripts/enroll_face.py），不开 HTTP 接口 ——
    建档必须是用户在设备前的显式操作，不该能被一个网络请求触发。
    """
    if rt.fixture:
        return {'users': [], 'threshold': None, 'fixture': True}
    from ..perception import FileIdentityStore

    store = FileIdentityStore(ROOT / "data" / "users")
    return {
        "users": [{"user_id": uid, "name": name}
                  for uid, name in store.list_users()],
        "threshold": store.threshold,
        "note": "注册/删除请用 python scripts/enroll_face.py",
    }


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    client_id = None
    try:
        client_id = await rt.hub.join(ws, rt.state)
        if rt.lifecycle:
            await ws.send_json({'type': 'voice_output', **rt.lifecycle.voice_output.snapshot()})
        if rt.camera_view is not None:
            from .camera_bridge import presence_message
            await ws.send_json(presence_message(rt.camera_view))
        while True:
            msg = await ws.receive_json()
            if not isinstance(msg, dict): continue
            kind = msg.get("type")
            if kind == "ready":
                _log.info("前端就绪")
            elif kind in ('audio_started', 'audio_ended') and rt.lifecycle:
                await rt.lifecycle.acknowledgement(client_id, msg)
            else:
                _log.debug("前端消息: %s", msg)
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        _log.warning("WebSocket 异常: %s", exc)
    finally:
        await rt.hub.leave(ws)


# ==================== 调试接口（给前端联调用）====================
# 这些接口让 w 不必等完整系统就能验证真实 WebSocket 链路。
# 语音链路（z）和感知链路接入后，这些接口保留作为诊断工具。


async def _speak(turn: DialogTurn, user: str = "") -> None:
    """Isolated transport test helper. Production entries use Lifecycle.submit."""
    audio = None
    rt.tts_error = "" if rt.tts is not None else "未启用"
    if rt.tts is not None and turn.speak_ja.strip():
        try:
            mime, data, _ = await synthesize(rt.tts, turn.speak_ja)
            audio = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
        except Exception as exc:
            rt.tts_error = f"{type(exc).__name__}: {exc}"
            _log.warning("TTS 合成失败，保留字幕: %s", rt.tts_error)

    await rt.set_state("SPEAKING")
    await rt.send_emotion(turn.emotion)
    if user:
        await rt.hub.send({"type": "user", "name": user, "known": True})

    await rt.hub.send({
        "type": "speak",
        # text 是「要显示的字幕」。中文缺失时回落日文 —— 宁可显示日文，
        # 也不要空白字幕（空白看起来像前端坏了）。
        "text": turn.subtitle_zh or turn.speak_ja,
        "text_ja": turn.speak_ja,       # 实际播的日文，TTS 与调试用
        "audio": audio,
        "duration_ms": max(400, len(turn.speak_ja) * 140),
    })


@app.post("/debug/say")
async def debug_say(text: str, user: str = "", skip_rewrite: bool = False):
    """手动让角色说一句话，驱动前端完整流程。

    用法（PowerShell）：
        curl.exe -X POST "http://127.0.0.1:8000/debug/say?text=今天天气不错"

    流程：改写 → 可选日语 TTS → 推情绪、字幕和音频。
    """
    if rt.lifecycle is None or rt.rewriter is None:
        return {'ok': False, 'status': 'unavailable', 'error': 'rewriter_not_loaded'}
    return await rt.lifecycle.submit('say', text, user=user, skip_rewrite=skip_rewrite)


@app.post("/debug/chat")
async def debug_chat(text: str, user: str = ""):
    """**完整对话链路入口。** ASR 到位前用它代替语音输入。

        curl.exe -X POST "http://127.0.0.1:8000/debug/chat?text=今天天气不错"

    流程：用户中文 → 云端（日文敬体 + 中文字幕 + 情绪）→ 本地规则引擎
    （敬体→常体+角色语尾）→ TTS + 前端。

    与 /debug/say 的区别：say 跳过云端直接喂改写器（给 w 调前端用），
    chat 走真实链路。要验证日文输出和中文字幕必须用 chat。

    ASR 与此入口共享 Lifecycle.submit，识别开始前即占用轮次。
    """
    if rt.dialog is None:
        return {"ok": False, "error": rt.dialog_error or "对话编排器未加载"}

    return await rt.lifecycle.submit('chat', text, user=user)


@app.post("/debug/chat/reset")
async def debug_chat_reset():
    """清空多轮历史。换用户时该调 —— 否则新用户继承上一个人的对话。"""
    if rt.dialog is None:
        return {"ok": False, "error": "对话编排器未加载"}
    if rt.lifecycle.coordinator.active:
        return {'ok': False, 'status': 'busy', 'error': 'busy'}
    if rt.lifecycle.sessions.current is not None:
        rt.lifecycle.sessions.reset(rt.lifecycle.sessions.current)
    return {"ok": True, "history_turns": 0}


@app.post('/debug/pcm')
async def debug_pcm(request: Request):
    """Local H03 adapter diagnostic: raw 16k/mono/S16LE, no capture device opened."""
    if not rt.lifecycle or not rt.lifecycle.input:
        return {'ok': False, 'status': 'disabled', 'error': 'asr_not_enabled'}
    if rt.lifecycle.closed:
        return {'ok': False, 'status': 'closed'}
    if not rt.lifecycle.coordinator.budget().voice_submit:
        return {'ok': False, 'status': 'busy', 'error': 'input_gated'}
    data = bytearray()
    async for chunk in request.stream():
        if len(data)+len(chunk) > 960000: raise HTTPException(413, 'PCM request exceeds 30 seconds')
        data.extend(chunk)
    if not rt.lifecycle.coordinator.budget().voice_submit:
        return {'ok': False, 'status': 'busy', 'error': 'input_gated'}
    rt.lifecycle.input.feed(bytes(data))
    return {'ok': True, 'status': 'accepted_pcm', 'sequence': rt.lifecycle.input.sequence}


@app.post('/debug/transcribe')
async def debug_transcribe(request: Request):
    """Browser sentence preview: raw 16k/mono/S16LE, ASR only, never cloud/TTS.

    The browser performs silence detection and sends one finished sentence.
    Use the shared reservation to prevent concurrent native ASR/model calls.
    The user explicitly commits the accumulated text through /debug/chat.
    """
    if not rt.lifecycle or not rt.lifecycle.input:
        return {'ok': False, 'status': 'disabled', 'error': 'asr_not_enabled'}
    if rt.lifecycle.closed:
        return {'ok': False, 'status': 'closed', 'error': 'service_stopping'}
    if not rt.lifecycle.coordinator.budget().voice_submit:
        return {'ok': False, 'status': 'busy', 'error': 'busy'}
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > 960000:
            raise HTTPException(413, 'PCM request exceeds 30 seconds')
        data.extend(chunk)
    if not data or len(data) % 2:
        raise HTTPException(422, 'Nonempty S16LE PCM required')
    return await rt.lifecycle.submit('voice', pcm=bytes(data), preview=True)


@app.get('/debug/events')
async def debug_events():
    return {'schema': 'aeg-events-v1', 'rows': rt.lifecycle.events.snapshot() if rt.lifecycle else []}


@app.api_route('/debug/fixture/control', methods=['GET', 'POST'])
async def fixture_control(request: Request):
    if not rt.fixture: raise HTTPException(404)
    if request.method == 'GET': return rt.fixture_control.snapshot()
    return rt.fixture_control.update(await request.json())


@app.post('/debug/fixture/perception')
async def fixture_perception(request: Request):
    if not rt.fixture: raise HTTPException(404)
    from types import SimpleNamespace
    from ..contracts import Identity
    data = await request.json()
    kind = data.get('kind')
    if kind not in ('APPEARED', 'LEFT', 'IDENTIFIED', 'COUNT_CHANGED'): raise HTTPException(422)
    identity = Identity(data.get('user_id'), data.get('name'), 1.)
    await _on_perception_event(SimpleNamespace(kind=kind, n_faces=data.get('n_faces', 1), identity=identity))
    return {'ok': True}


@app.post('/debug/fixture/expire')
async def fixture_expire():
    if not rt.fixture: raise HTTPException(404)
    turn = rt.lifecycle.coordinator.active
    if turn:
        turn.started -= max(rt.lifecycle.coordinator.model_timeout, rt.lifecycle.coordinator.playback_timeout)
        await rt.lifecycle.tick()
    return {'ok': True}


@app.post("/debug/emotion")
async def debug_emotion(value: str):
    """单独测试情绪立绘切换。连续调同一情绪会轮换候选图。"""
    sprite = await rt.send_emotion(value)
    return {"ok": True, "emotion": value, "sprite": sprite}


@app.get("/debug/sprites")
async def debug_sprites():
    """列出情绪→立绘映射与状态动画。前端对接时先看这个。

    立绘文件名由后端下发（`emotion` 消息的 `sprite` 字段），
    前端不要按情绪名自己拼文件名 —— 一个情绪有多张候选图，轮换在后端。
    """
    if rt.card is None:
        return {"ok": False, "error": "人格卡未加载"}
    sprite_dir = rt.char_dir / "sprites" / "aligned" if rt.char_dir else None
    return {
        "ok": True,
        "character": rt.card.name,
        "emotions": {e: rt.available_sprites(e) for e in EMOTIONS} if sprite_dir else {},
        "animations": {k: a.to_dict() for k, a in rt.card.animations.items()},
        "sprite_dir": str(sprite_dir) if sprite_dir else None,
        "warnings": rt.card.validate(sprite_dir=sprite_dir),
    }


@app.post("/debug/state")
async def debug_state(value: str):
    """单独测试状态切换。"""
    if rt.lifecycle and rt.lifecycle.coordinator.active:
        return {'ok': False, 'status': 'busy', 'error': 'busy'}
    if value not in ('IDLE', 'NOTICED'):
        return {'ok': False, 'status': 'invalid', 'error': 'lifecycle_owned_state'}
    await rt.set_state(value)        # type: ignore[arg-type]
    return {"ok": True, "state": value}


@app.post("/debug/background")
async def debug_background(value: str):
    """单独测试背景切换。"""
    await rt.hub.send({"type": "background", "value": value})
    return {"ok": True, "background": value}


@app.post("/debug/demo")
async def debug_demo():
    """跑一段预设对话，演示完整流程。录演示视频前可用它检查链路。"""
    # 第二列是预期情绪，仅作注释用（实际情绪由改写器判定）。
    # 覆盖全部六个枚举值，这样跑一遍 demo 就能检查六张立绘是否都正常显示。
    script = [
        ("你好。", "happy"),
        ("今天天气不错。", "neutral"),
        ("我有点累了。", "unhappy"),
        ("你怎么会在这里？", "surprised"),
        ("谢谢你的夸奖。", "shy"),
        ("这件事我觉得不对。", "assertive"),
    ]
    for line, _ in script:
        outcome = await debug_chat(line)
        if not outcome.get('ok'): return outcome
        while rt.lifecycle.coordinator.active:
            await asyncio.sleep(.05)
    return {"ok": True, "lines": len(script)}
