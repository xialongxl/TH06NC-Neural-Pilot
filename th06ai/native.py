"""Version-pinned, read-only TH06NC state adapter.

The current build was mapped from its own instruction operands, not by adding
a guessed delta to old TH06 addresses. No write/debug/injection API is used.
"""
from __future__ import annotations

import hashlib
import math
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from .combat import ENEMY_POOL, ENEMY_CAPACITY, ENEMY_STRIDE, BOSS_POINTERS, decode_enemies, decode_enemy_bodies
from .input_timing import clock_ns
from .lasers import LASER_START,LASER_CAPACITY,LASER_STRIDE,decode_lasers
from .stage import STAGE_RVA, decode_stage

SHA256 = "48630a42a2eb6762d0db2a7e0d151203efbe67220fef7d2f9d0d51857928ac82"
BUILD_ID = "25306795"
PLAYER = 0x549FF0
BOUNDS = 0x549D30
PAUSED = 0x53D409
NC_PAUSE = 0x53D400
NC_CONTINUE = 0x53D401
NC_RESULT = 0x53D406
ENGINE_STATE = 0xC6DFBC
NC_PAUSE_UI = 0x429270 + 0x52D0
NC_MENU = 0xC53460
HUD_POINTER = 0xABAE28
INPUT_WORD = 0xABAE80
BULLET_NODE = 0x436EB0
BULLET_MANAGER = 0x436EF0
BULLET_UPDATE = 0x10FA0
PLAYER_MOTION = 0x6AAA6
STRIDE = 0x620
CAPACITY = 640
ITEM_POOL = 0xBFB2F8
ITEM_CAPACITY = 1024
ITEM_STRIDE = 0x160
POWER = 0x53CAD8
LIVES_BOMBS = 0x549D40
SCORE = 0x53D3E8
POINT_ITEMS = 0x53D40C
GRAZE_TOTAL = 0x549D20
GRAZE_DISABLED = 0x553EB8
WORLD_START = BULLET_NODE
WORLD_SIZE = GRAZE_DISABLED + 1 - WORLD_START
ACTORS_START = ENEMY_POOL
ACTORS_SIZE = ITEM_POOL + ITEM_CAPACITY * ITEM_STRIDE - ACTORS_START
MOTION_BYTES = bytes.fromhex("f3 0f 59 b7 10 77 00 00 f3 0f 11 bf a0 78 00 00 f3 0f 59 bf 14 77 00 00")
SIGNATURES = {
    PLAYER_MOTION: MOTION_BYTES,
    0x694DF: bytes.fromhex("48 8d 1d 0a 0b 4e 00"),  # lea rbx, player
    0x10FAF: bytes.fromhex("80 3d 53 c4 52 00 00"),  # same pause gate as player
    0x3CA8B: bytes.fromhex("48 8d 05 0e 45 fd ff"),  # callback -> 0x10fa0
    0x44264: bytes.fromhex("4c 8d 35 8d 70 bb 00"),  # visible item pool
    0x4461A: bytes.fromhex("66 39 15 b7 84 4f 00"),  # power comparison
    0x44C87: bytes.fromhex("80 3d b3 50 50 00 08"),  # bomb inventory cap
    0x44CC7: bytes.fromhex("80 3d 72 50 50 00 08"),  # lives cap
    0x44A2B: bytes.fromhex("48 01 05 b6 89 4f 00"),  # collected point score
    0x37C0B: bytes.fromhex("48 8d 1d a6 64 ab 00"),  # own-build enemy pool
    0x393A3: bytes.fromhex("83 83 34 02 00 00 f6"),  # player-hit HP decrement
    0x39479: bytes.fromhex("44 29 83 34 02 00 00"),  # player-shot HP decrement
    0x3BD69: bytes.fromhex("48 8b 0d 28 e4 bb 00"),  # HUD Boss pointer
    0x118FA: bytes.fromhex("8b 05 20 84 53 00"),  # bullet graze total
    0x11967: bytes.fromhex("48 81 05 76 ba 52 00 f4 01 00 00"),  # graze adds 500 score
    0x6C567: bytes.fromhex("8b 05 b3 d7 4d 00"),  # laser graze uses the same total
    0x6C5D8: bytes.fromhex("48 81 05 05 0e 4d 00 f4 01 00 00"),
    0x118DC: bytes.fromhex("44 38 2d d5 25 54 00"),  # counter-disable gate
    0x11814: bytes.fromhex("f3 0f 10 b3 f8 05 00 00"),  # bullet collision height
    0x1181C: bytes.fromhex("f3 44 0f 10 8b f4 05 00 00"),  # bullet collision width
    0x11B32: bytes.fromhex("4c 8d 83 f4 05 00 00"),  # hit-test size pointer
    0x6C122: bytes.fromhex("f3 41 0f 5d 30"),  # circular hit test: min(width, height)
    0x6C145: bytes.fromhex("f3 0f 58 b3 4c 77 00 00"),  # add player collision radius
    0x39312: bytes.fromhex("a8 04"),  # body contact collision-processing gate
    0x3931A: bytes.fromhex("f6 83 be 00 00 00 08"),  # separate contact-disable flag
    0x3932E: bytes.fromhex("24 03"),  # body contact requires low flags == 3
    0x39349: bytes.fromhex("f3 0f 10 83 c0 02 00 00"),  # enemy contact width
    0x39358: bytes.fromhex("f3 0f 10 8b c4 02 00 00"),  # enemy contact height
    0x39360: bytes.fromhex("45 33 c9"),  # rectangle mode, not bullet circle mode
    0x3938C: bytes.fromhex("e8 ff 2c 03 00"),  # call contact collision routine
    0x352A60: bytes.fromhex("ab aa 2a 3f"),  # 2/3 body-size scale
    0x352A44: bytes.fromhex("00 00 00 3f"),  # half-size scale in collision routine
    0x9806: bytes.fromhex('80 3d f3 3b 53 00 00'),  # NC pause UI gate
    0x407D1: bytes.fromhex('83 b9 b0 36 00 00 00'),  # message index gate
    0x431D7: bytes.fromhex('8b 05 f7 98 4f 00'),  # native STAGE %d / FINAL / EXTRA display
    0x3F0D3: bytes.fromhex('8b 05 fb d9 4f 00'),  # stage-specific resource selection, index-1
}
BULLET_DTYPE = np.dtype({"names": ["x", "y", "state", "width", "height"],
                         "formats": ["<f4", "<f4", "<u2", "<f4", "<f4"],
                         "offsets": [0x30, 0x34, 0x44, 0x5F4, 0x5F8], "itemsize": STRIDE})
ITEM_DTYPE = np.dtype({"names": ["active", "x", "y", "vx", "vy", "type"],
                       "formats": ["u1", "<f4", "<f4", "<f4", "<f4", "i1"],
                       "offsets": [0, 0x10, 0x14, 0x1C, 0x20, 0x34], "itemsize": ITEM_STRIDE})


@dataclass(frozen=True)
class GameResources:
    score: int
    power: int
    bombs: int
    lives: int
    point_items: int
    graze: int | None = None  # Native total, diagnostic only; None if counting is disabled.

    def __post_init__(self):
        if not (0 <= self.score < 2**64 and 0 <= self.power <= 128
                and 0 <= self.bombs <= 8 and -1 <= self.lives <= 8
                and 0 <= self.point_items <= 65535
                and (self.graze is None or 0 <= self.graze <= 999999)):
            raise RuntimeError("分数/火力/炸弹/残机读取未通过范围校验")


def decode_items(raw: bytes, width: float, height: float) -> np.ndarray:
    if len(raw) != ITEM_CAPACITY * ITEM_STRIDE:
        raise RuntimeError("道具池读取长度异常")
    data = np.frombuffer(raw, dtype=ITEM_DTYPE)
    active = data["active"] != 0
    if np.any(active & ((data["type"] < 0) | (data["type"] > 6))):
        raise RuntimeError("道具类型未知，拒绝把错误数据交给模型")
    finite = np.isfinite(data["x"]) & np.isfinite(data["y"]) & np.isfinite(data["vx"]) & np.isfinite(data["vy"])
    if np.any(active & ~finite):
        raise RuntimeError("道具坐标/速度包含无效数值")
    visible = active & (data["x"] >= 0) & (data["x"] <= width) & (data["y"] >= 0) & (data["y"] <= height)
    slots = np.flatnonzero(visible)
    return np.column_stack((slots, data["x"][visible], data["y"][visible], data["type"][visible],
                            data["vx"][visible], data["vy"][visible])).astype(np.float32)


def inspect_game(path: Path) -> dict:
    import pefile
    if not path.is_file():
        raise RuntimeError(f"找不到游戏主程序：{path}")
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != SHA256:
        raise RuntimeError(f"游戏版本尚未适配，拒绝读取未知地址：SHA256={sha}")
    pe = pefile.PE(data=raw, fast_load=True)
    if pe.FILE_HEADER.Machine != 0x8664 or pe.OPTIONAL_HEADER.SizeOfImage != 0xCBC000:
        raise RuntimeError("游戏 PE 架构或内存布局与已验证版本不符")
    for rva, expected in SIGNATURES.items():
        if pe.get_data(rva, len(expected)) != expected:
            raise RuntimeError(f"游戏代码特征与已验证版本不符：RVA 0x{rva:x}")
    return {"path": str(path), "sha256": sha, "steam_build": BUILD_ID,
            "architecture": "AMD64", "mode": "read-only", "signature_count": len(SIGNATURES)}


@dataclass
class Frame:
    tick: int
    x: float
    y: float
    state: int
    focused: bool
    paused: bool
    bullets: np.ndarray  # visible current positions only, columns slot,x,y
    bounds: tuple[float, float, float, float]
    stage_active: bool
    observed_at: float
    items: np.ndarray = field(default_factory=lambda: np.empty((0, 6), np.float32))
    resources: GameResources | None = None
    enemies: np.ndarray = field(default_factory=lambda: np.empty((0, 8), np.float32))
    bullet_sizes: np.ndarray | None = None  # width,height aligned with visible bullet rows
    player_radius: float | None = None
    enemy_bodies: np.ndarray = field(default_factory=lambda: np.empty((0, 6), np.float32))
    capture_started_ns: int = 0
    capture_finished_ns: int = 0
    input_bits: int | None = None
    input_observed_ns: int = 0
    phase: str = ''
    phase_evidence: dict = field(default_factory=dict)
    menu_selection: int | None = None
    menu_ready: bool = False
    player_shots: int | None = None
    lasers: np.ndarray = field(default_factory=lambda: np.empty((0,8),np.float32))
    stage_id: int | None = None
    stage_raw: int | None = None
    bullet_states: np.ndarray | None = None


@dataclass
class ControlSample:
    tick: int
    x: float
    y: float
    state: int
    paused: bool
    stage_active: bool
    stage_id: int | None
    phase: str
    input_bits: int | None
    input_observed_ns: int
    capture_started_ns: int
    capture_finished_ns: int


class _StateSnapshot:
    def __init__(self, world, actors):
        self.world, self.actors = memoryview(world), memoryview(actors)

    def read(self, rva, size):
        for start, region in ((WORLD_START, self.world), (ACTORS_START, self.actors)):
            offset = rva - start
            if 0 <= offset and offset + size <= len(region):
                return region[offset:offset+size]
        raise ValueError("状态字段超出已读取的快照范围")

    def active(self, base):
        node = self.read(BULLET_NODE, 0x40)
        callback = struct.unpack_from("<Q", node, 8)[0]
        argument = struct.unpack_from("<Q", node, 0x38)[0]
        return argument == base + BULLET_MANAGER and callback in (0, base + BULLET_UPDATE)


class GameReader:
    def __init__(self, game_exe: Path):
        from .win32 import ReadOnlyProcess, GameWindow, processes
        self.identity = inspect_game(game_exe)
        candidates = [pid for pid, name in processes() if name.lower() == game_exe.name.lower()]
        if len(candidates) != 1:
            raise RuntimeError(f"请先打开一个 TH06NC 游戏窗口（当前找到 {len(candidates)} 个游戏进程）")
        self.process = ReadOnlyProcess(candidates[0], game_exe.name)
        if Path(self.process.path).resolve() != game_exe.resolve():
            self.process.close()
            raise RuntimeError("正在运行的游戏并非配置中指定的主程序")
        try:
            self.window = GameWindow.find(candidates[0])
        except BaseException:
            self.process.close()
            raise
        for rva, expected in SIGNATURES.items():
            if self.process.read(rva, len(expected)) != expected:
                self.close()
                raise RuntimeError("运行中的游戏代码与磁盘不一致，拒绝接入")
        self._dtype = BULLET_DTYPE
        self._world = bytearray(WORLD_SIZE)
        self._actors = bytearray(ACTORS_SIZE)
        self.checkpoint = None
        self.snapshot_diagnostics = {}

    def read_control(self):
        """Small native control/phase sample; never a policy observation."""
        from .game_phase import classify
        started = clock_ns()
        p = self.process
        tick = struct.unpack('<ii', p.read(BULLET_MANAGER, 8))[1]
        player = p.read(PLAYER+0x7710, 0x1A0)
        x,y = struct.unpack_from('<ff',player,0x20)
        state = player[0x188]
        flags = p.read(NC_PAUSE,10)
        engine = struct.unpack('<i',p.read(ENGINE_STATE,4))[0]
        stage = decode_stage(struct.unpack('<i',p.read(STAGE_RVA,4))[0]) if engine==2 else None
        hud = p.read_indirect(HUD_POINTER,0x3668,0x4C) if engine==2 and hasattr(p,'read_indirect') else None
        message = struct.unpack_from('<i',hud,0x48)[0] if hud is not None else -1
        phase = classify(active=engine==2,engine_state=engine,player_state=state,
                         pause_flag=flags[0],continue_flag=flags[1],result_flag=flags[6],
                         script_stop=bool(flags[9]),dialogue_index=message)
        bits, observed = None, 0
        try:
            bits = int.from_bytes(p.read(INPUT_WORD,4),'little');observed=clock_ns()
        except (OSError,RuntimeError,ValueError):pass
        if not math.isfinite(x) or not math.isfinite(y) or state>5:
            raise RuntimeError('轻量自机状态读取异常')
        sample = ControlSample(tick,x,y,state,phase.paused,phase.stage_active,stage,phase.phase,
                               bits,observed,started,clock_ns())
        observer = getattr(self, 'control_observer', None)
        if observer is not None:
            try:observer(sample)
            except Exception as error:self.snapshot_diagnostics['recording_error'] = str(error)
        return sample

    def _capture(self):
        """Read current state once; a concurrent game tick is diagnostic only.

        Asynchronous read-only capture is not atomic. Crossing a tick does not
        imply a failed memory read. Do not retry, discard or stop because of it.
        Two reusable bulk buffers keep the sampling interval short; no fallback
        to a previous observation is used.
        """
        p = self.process
        if not hasattr(self, "_world"):
            self._world, self._actors = bytearray(WORLD_SIZE), bytearray(ACTORS_SIZE)
        snapshot = _StateSnapshot(self._world, self._actors)
        started = time.perf_counter()
        checkpoint = getattr(self, "checkpoint", None)
        if checkpoint is not None:
            checkpoint()
        before = p.read(BULLET_MANAGER, 8)
        for address, buffer in ((WORLD_START, self._world), (ACTORS_START, self._actors)):
            if address == ACTORS_START and not snapshot.active(p.base):
                break
            if hasattr(p, "read_into"):
                p.read_into(address, buffer)
            else:
                buffer[:] = p.read(address, len(buffer))
        after = p.read(BULLET_MANAGER, 8)
        embedded = snapshot.read(BULLET_MANAGER, 8)
        self.snapshot_diagnostics = {"attempts": 1,
                                     "elapsed_ms": round((time.perf_counter()-started)*1000, 3),
                                     "before_tick": struct.unpack("<ii", before)[1],
                                     "sample_tick": struct.unpack("<ii", embedded)[1],
                                     "after_tick": struct.unpack("<ii", after)[1],
                                     "crossed_tick": not (before == after and before == embedded)}
        return snapshot

    def read(self) -> Frame:
        started_ns, observed_at = clock_ns(), time.monotonic()
        p = self._capture()
        # Optional diagnostic read. Failure here must not create a new STOP
        # condition or invalidate an otherwise valid gameplay snapshot.
        input_bits, input_observed_ns = None, 0
        try:
            input_bits = int.from_bytes(self.process.read(INPUT_WORD, 4), 'little')
            input_observed_ns = clock_ns()
        except (OSError, RuntimeError, ValueError):
            pass
        player = p.read(PLAYER + 0x7710, 0x1A0)
        x, y = struct.unpack_from("<ff", player, 0x20)
        player_radius = struct.unpack_from("<f", player, 0x3C)[0]
        state = player[0x188]
        focused = bool(player[0x14C])
        paused = p.read(PAUSED, 1)[0]
        if paused not in (0, 1) or state > 5:
            raise RuntimeError("自机状态/暂停标记异常，拒绝继续控制")
        bounds = struct.unpack("<ffff", p.read(BOUNDS, 16))
        if any(not math.isfinite(v) for v in (x, y, *bounds)):
            raise RuntimeError("读取到非有限的游戏坐标")
        # The static node can be unlinked while the game is paused. Its pinned
        # argument and frame counter remain available; do not turn pause into
        # a fake zero-tick title screen.
        active = p.active(self.process.base)
        if not active:
            return self._complete_frame(Frame(0, x, y, state, focused, bool(paused), np.empty((0, 3), np.float32),
                         bounds, False, observed_at), started_ns, input_bits, input_observed_ns)
        manager = BULLET_MANAGER
        if not (100 <= bounds[2] <= 1500 and 100 <= bounds[3] <= 1500):
            raise RuntimeError("游戏场地尺寸未通过校验")
        tick = struct.unpack("<ii", p.read(manager, 8))[1]
        raw = p.read(manager + 8, CAPACITY * STRIDE)
        item_raw = p.read(ITEM_POOL, ITEM_CAPACITY * ITEM_STRIDE)
        enemy_raw = p.read(ENEMY_POOL, ENEMY_CAPACITY * ENEMY_STRIDE)
        boss_ptrs = struct.unpack("<QQQQ", p.read(BOSS_POINTERS, 32))
        power = struct.unpack("<H", p.read(POWER, 2))[0]
        lives, bombs = struct.unpack("<bb", p.read(LIVES_BOMBS, 2))
        score = struct.unpack("<Q", p.read(SCORE, 8))[0]
        points = struct.unpack("<H", p.read(POINT_ITEMS, 2))[0]
        graze = struct.unpack("<i", p.read(GRAZE_TOTAL, 4))[0]
        graze_disabled = p.read(GRAZE_DISABLED, 1)[0]
        data = np.frombuffer(raw, self._dtype, CAPACITY)
        states = data["state"]
        if np.any(states > 5):
            raise RuntimeError("弹幕池结构与当前版本不一致")
        # Rendering visibility: active/spawning bullets in the visible field.
        # Do not expose velocity fields, RNG, script counters or future spawns.
        width, height = float(bounds[2] + 2 * bounds[0]), float(bounds[3] + 2 * bounds[1])
        visible = ((states >= 1) & (states <= 4) & np.isfinite(data["x"]) & np.isfinite(data["y"]) &
                   (data["x"] >= 0) & (data["x"] <= width) &
                   (data["y"] >= 0) & (data["y"] <= height))
        slots = np.flatnonzero(visible)
        bullets = np.column_stack((slots, data["x"][visible], data["y"][visible])).astype(np.float32)
        sizes = np.column_stack((data["width"][visible], data["height"][visible])).astype(np.float32)
        # Spawning sprites can have unset sizes. Unknown geometry remains
        # unknown; never turn missing/invalid hitboxes into a zero-radius bullet.
        invalid = ~np.isfinite(sizes) | (sizes <= 0) | (sizes > 256)
        sizes[np.any(invalid, axis=1)] = np.nan
        if not math.isfinite(player_radius) or not 0 < player_radius <= 32:
            player_radius = None
        items = decode_items(item_raw, width, height)
        resources = GameResources(score, power, bombs, lives, points, None if graze_disabled else graze)
        enemies = decode_enemies(enemy_raw, width, height, [ptr - self.process.base for ptr in boss_ptrs if ptr])
        enemy_bodies = decode_enemy_bodies(enemy_raw, width, height, player_radius if player_radius is not None else 32.)
        frame=Frame(tick, x, y, state, focused, bool(paused), bullets, bounds, True, observed_at,
                     items, resources, enemies, sizes, player_radius, enemy_bodies)
        frame.lasers=decode_lasers(p.read(LASER_START,LASER_CAPACITY*LASER_STRIDE),width,height)
        frame.stage_raw = struct.unpack('<i', p.read(STAGE_RVA, 4))[0]
        frame.stage_id = decode_stage(frame.stage_raw)
        frame.bullet_states = states[visible].copy()
        shots=np.ndarray((80,),dtype='<u2',buffer=p.read(PLAYER+0x410,80*0x170),offset=0x10,strides=(0x170,))
        frame.player_shots=int(np.count_nonzero(shots==1))
        return self._complete_frame(frame, started_ns, input_bits, input_observed_ns)

    def _complete_frame(self, frame, started_ns, input_bits, input_observed_ns):
        from .game_phase import classify
        # Menu/dialogue state is read independently of movement and bullet
        # density. Optional diagnostic failure never fabricates a pause.
        evidence={}
        try:
            p=self.process
            if hasattr(p,'read_indirect'):
                flags=p.read(NC_PAUSE,10)
                engine=struct.unpack('<i',p.read(ENGINE_STATE,4))[0]
                pause_ui=p.read(NC_PAUSE_UI+0xB44,0x254)
                hud=p.read_indirect(HUD_POINTER,0x3668,0x2A4) if engine==2 else None
                dialogue_index=struct.unpack_from('<i',hud,0x48)[0] if hud is not None else -1
                evidence={'engine_state':engine,'nc_pause':flags[0],'nc_continue':flags[1],
                          'nc_result':flags[6],'script_stop':bool(flags[9]),
                          'pause_selection':struct.unpack_from('<i',pause_ui)[0],
                          'pause_animation':struct.unpack_from('<i',pause_ui,0x250)[0],
                          'dialogue_index':dialogue_index}
                if engine!=2:
                    evidence.update(menu_screen=struct.unpack('<i',p.read(NC_MENU+0x168B0,4))[0],
                                    menu_selection=struct.unpack('<i',p.read(NC_MENU+0x28,4))[0],
                                    menu_ticks=struct.unpack('<i',p.read(NC_MENU+0x9638,4))[0])
                state=classify(active=frame.stage_active,player_state=frame.state,
                    boss=bool(len(frame.enemies) and np.any(frame.enemies[:,7])),
                    engine_state=engine,pause_flag=flags[0],continue_flag=flags[1],result_flag=flags[6],
                    script_stop=bool(flags[9]),dialogue_index=dialogue_index,
                    pause_selection=evidence['pause_selection'],pause_animation=evidence['pause_animation'],
                    nc_screen=evidence.get('menu_screen'),selection=evidence.get('menu_selection'),
                    menu_ticks=evidence.get('menu_ticks',0))
                frame.phase,frame.paused,frame.stage_active=state.phase,state.paused,state.stage_active
                frame.menu_selection,frame.menu_ready=state.selection,state.ready
        except (OSError,RuntimeError,ValueError,struct.error) as error:
            evidence['read_error']=str(error)
            frame.phase='unknown'
        frame.phase_evidence=evidence
        if not frame.stage_active or evidence.get('engine_state') != 2:
            frame.stage_id = None
        frame.capture_started_ns = started_ns
        frame.capture_finished_ns = clock_ns()
        frame.input_bits, frame.input_observed_ns = input_bits, input_observed_ns
        self.snapshot_diagnostics['total_elapsed_ms'] = round((frame.capture_finished_ns-started_ns)/1e6, 3)
        self.snapshot_diagnostics['input_observed'] = input_bits is not None
        observer = getattr(self, 'observer', None)
        if observer is not None:
            try:observer(frame)
            except Exception as error:self.snapshot_diagnostics['recording_error'] = str(error)
        return frame

    def close(self):
        self.process.close()
