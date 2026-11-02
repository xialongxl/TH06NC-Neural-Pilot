from __future__ import annotations

import os
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .config import Config, ROOT
from .control import set_paused
from .files import atomic_json, read_json
from .model_store import list_checkpoints
from .console_service import build_command

BG, SIDE, PANEL, PANEL2 = "#10131c", "#0b0e16", "#1a2030", "#20283b"
LINE, TEXT, MUTED = "#2b354c", "#edf1fb", "#99a7c1"
ACCENT, GREEN, GOLD, FONT = "#ee7b91", "#72dac0", "#e9c58e", "Microsoft YaHei UI"
ACTIVE_STATES = {"starting", "running", "paused", "finishing", "monitoring"}


def current_session():
    pointer = read_json(ROOT / "runs" / "current.json")
    if not pointer.get("run_dir"):
        return None, {}
    path = Path(pointer["run_dir"]).resolve()
    if not path.is_relative_to((ROOT / "runs").resolve()):
        return None, {}
    return path, read_json(path / "status.json")


def session_active(status):
    return status.get("state") in ACTIVE_STATES and 0 <= time.monotonic() - status.get("monotonic", 0) < 2


def duration(seconds):
    seconds = max(0, int(seconds or 0))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def model_title(entry):
    if entry.path == ROOT / "models" / "latest.zip":
        return "最近保存"
    if entry.path.parent.name in ("repairs", "checkpoints") or entry.path.stem == "latest":
        return f"检查点 · {entry.metadata.get('steps', 0):,} 步"
    return entry.path.stem


def model_status(entry):
    return "可用" if entry.usable else entry.compatibility


class Dashboard:
    def __init__(self, root, config: Config, *, preview=False):
        self.root, self.config, self.preview = root, config, preview
        self.selected, self.process, self.page = None, None, "train"
        self.catalog, self.points, self.variables = [], [], {}
        self.sample_stamp, self.last_catalog_refresh, self.shown_paused = None, 0.0, False
        root.title("TH06NC AI · 训练控制台" + (" · 界面预览" if preview else ""))
        root.geometry("1180x840")
        root.minsize(1080, 800)
        root.configure(bg=BG)
        self._style()
        self._layout()
        self.refresh_catalog()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<F8>", lambda _: self.emergency_stop())
        # F7 is handled globally by the guard, avoiding two toggles per press.
        self.refresh()

    def _style(self):
        s = ttk.Style(self.root)
        s.theme_use("clam")
        s.configure("Model.Treeview", background=PANEL, fieldbackground=PANEL, foreground=TEXT,
                    borderwidth=0, rowheight=43, font=(FONT, 10))
        s.configure("Model.Treeview.Heading", background=PANEL2, foreground=MUTED,
                    relief="flat", padding=(9, 10), font=(FONT, 9))
        s.map("Model.Treeview", background=[("selected", "#3a3047")], foreground=[("selected", "#ffd5df")])
        s.map("Model.Treeview.Heading", background=[("active", PANEL2)])
        s.configure("Dark.TCombobox", fieldbackground=PANEL2, background=PANEL2, foreground=TEXT,
                    arrowcolor=TEXT, bordercolor=LINE, lightcolor=LINE, darkcolor=LINE, padding=6)
        s.map("Dark.TCombobox", fieldbackground=[("readonly", PANEL2)], selectbackground=[("readonly", PANEL2)],
              selectforeground=[("readonly", TEXT)])

    def label(self, parent, text="", *, color=TEXT, size=10, bold=False, bg=None, **kwargs):
        return tk.Label(parent, text=text, bg=bg or parent.cget("bg"), fg=color,
                        font=(FONT, size, "bold" if bold else "normal"), anchor="w", **kwargs)

    def vlabel(self, parent, name, text, **kwargs):
        self.variables[name] = tk.StringVar(value=text)
        return self.label(parent, textvariable=self.variables[name], **kwargs)

    def button(self, parent, text, command, *, kind="secondary", **kwargs):
        colors = {"primary": (ACCENT, "#25151e", "#f596a8"), "secondary": (PANEL2, TEXT, "#303c54"),
                  "danger": ("#492733", "#ffacb9", "#63313f"), "quiet": (parent.cget("bg"), MUTED, PANEL2)}
        bg, fg, hover = colors[kind]
        options = dict(text=text, command=command, bg=bg, fg=fg, activebackground=hover, activeforeground=fg,
                       disabledforeground="#68758d", relief="flat", borderwidth=0, padx=14, pady=9,
                       cursor="hand2", takefocus=True, font=(FONT, 10, "bold" if kind == "primary" else "normal"),
                       highlightthickness=1, highlightbackground=LINE, highlightcolor=ACCENT)
        options.update(kwargs)
        return tk.Button(parent, **options)

    def card(self, parent):
        return tk.Frame(parent, bg=PANEL, highlightbackground=LINE, highlightthickness=1)

    def _layout(self):
        sidebar = tk.Frame(self.root, bg=SIDE, width=172)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        logo = tk.Canvas(sidebar, width=52, height=52, bg=SIDE, highlightthickness=0)
        logo.pack(anchor="w", padx=25, pady=(27, 12))
        logo.create_oval(4, 4, 45, 45, outline=ACCENT, width=2)
        logo.create_arc(12, 12, 37, 37, start=15, extent=260, style="arc", outline=TEXT, width=2)
        logo.create_oval(35, 5, 44, 14, fill=ACCENT, outline=SIDE)
        self.label(sidebar, "TH06NC", size=19, bold=True).pack(anchor="w", padx=25)
        self.label(sidebar, "NEURAL PILOT", color=MUTED, size=8).pack(anchor="w", padx=27, pady=(2, 32))
        self.nav = {}
        for key, title, detail in (("train", "训练控制台", "TRAINING"), ("models", "模型库", "CHECKPOINTS"),
                                   ("diagnostics", "保护与诊断", "GUARD / LOGS")):
            btn = self.button(sidebar, f"{title}\n{detail}", lambda k=key: self.show_page(k), kind="quiet", anchor="w", justify="left")
            btn.pack(fill="x", padx=13, pady=4)
            self.nav[key] = btn
        foot = tk.Frame(sidebar, bg=SIDE)
        foot.pack(side="bottom", fill="x", padx=24, pady=23)
        self.label(foot, "PPO  ·  CUDA", color=GOLD, size=9).pack(anchor="w")
        self.label(foot, "界面预览" if self.preview else "方向键 / Shift / Z / X", color=MUTED, size=8).pack(anchor="w", pady=(8, 0))
        main = tk.Frame(self.root, bg=BG)
        main.pack(side="left", fill="both", expand=True, padx=27, pady=23)
        header = tk.Frame(main, bg=BG)
        header.pack(fill="x")
        self.page_title = self.label(header, "训练控制台", size=23, bold=True)
        self.page_title.pack(side="left")
        self.top_badge = self.label(header, "●  等待启动", color=GREEN, size=10, padx=13, pady=8, bg=PANEL)
        self.top_badge.pack(side="right")
        self.label(main, "学习型弹幕策略 · 模型、训练进度与显卡保护", color=MUTED).pack(anchor="w", pady=(4, 18))
        self.banner = self.vlabel(main, "banner", "尚未运行。选择模型与时长后，由你启动训练。", color=MUTED,
                                  padx=14, pady=10, bg=PANEL2, wraplength=850, justify="left")
        self.banner.pack(fill="x", pady=(0, 17))
        self.pages = {key: tk.Frame(main, bg=BG) for key in self.nav}
        self._training_page(self.pages["train"])
        self._models_page(self.pages["models"])
        self._diagnostics_page(self.pages["diagnostics"])
        bottom = tk.Frame(main, bg=BG)
        bottom.pack(side="bottom", fill="x", pady=(12, 0))
        self.label(bottom, "F7  暂停 / 恢复     F8  紧急停止", color=GOLD, size=9).pack(side="left")
        self.label(bottom, "关闭面板会请求暂停训练" if not self.preview else "预览模式 · 不连接游戏 / GPU", color=MUTED, size=8).pack(side="right")
        self.show_page("train")

    def _training_page(self, page):
        metrics = tk.Frame(page, bg=BG)
        metrics.pack(fill="x", pady=(0, 15))
        for i, (name, title, value, note) in enumerate((("time", "本次有效训练", "00:00", "暂停不计入训练时长"),
                  ("steps", "累计采样步数", "—", "本次新增 0 步"), ("updates", "参数更新", "—", "保留权重与优化器"))):
            metrics.columnconfigure(i, weight=1)
            card = self.card(metrics)
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))
            self.label(card, title, color=MUTED, size=9).pack(anchor="w", padx=16, pady=(11, 5))
            self.vlabel(card, name, value, size=24, bold=True, color=GOLD if i == 0 else TEXT).pack(anchor="w", padx=16)
            self.vlabel(card, name + "_note", note, color=MUTED, size=8).pack(anchor="w", padx=16, pady=(3, 12))
        body = tk.Frame(page, bg=BG)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)
        left, right = self.card(body), self.card(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 15))
        right.grid(row=0, column=1, sticky="nsew")
        inner = tk.Frame(left, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=18, pady=15)
        self.label(inner, "训练会话", size=14, bold=True).pack(anchor="w")
        self.label(inner, "当前选择的模型", color=MUTED, size=9).pack(anchor="w", pady=(12, 4))
        self.vlabel(inner, "model_name", "尚未选择模型", size=11, bold=True, wraplength=435, justify="left").pack(anchor="w")
        self.vlabel(inner, "model_meta", "在模型库中选择历史检查点", color=MUTED, size=9, wraplength=435,
                    justify="left").pack(anchor="w", pady=(4, 3))
        self.button(inner, "选择历史模型  →", lambda: self.show_page("models"), kind="quiet", pady=4).pack(anchor="w")
        time_row = tk.Frame(inner, bg=PANEL)
        time_row.pack(fill="x", pady=(13, 11))
        self.label(time_row, "本次训练时长", color=MUTED, size=9).pack(side="left")
        self.minutes = tk.StringVar(value=str(int(self.config.minutes)))
        self.time_select = ttk.Combobox(time_row, textvariable=self.minutes, values=(1, 5, 10, 30, 60), width=6,
                                       state="readonly", style="Dark.TCombobox", font=(FONT, 9))
        self.time_select.pack(side="right")
        self.label(time_row, "分钟  ", color=MUTED, size=9).pack(side="right")
        self.resume_button = self.button(inner, "继续训练", lambda: self.start("resume"), kind="primary")
        self.resume_button.pack(fill="x")
        self.new_button = self.button(inner, "新建模型训练", lambda: self.start("train"))
        self.new_button.pack(fill="x", pady=(7, 12))
        tk.Frame(inner, bg=LINE, height=1).pack(fill="x", pady=(0, 12))
        row = tk.Frame(inner, bg=PANEL)
        row.pack(fill="x")
        self.pause_button = self.button(row, "暂停训练  F7", lambda: self.request_pause(not self.shown_paused))
        self.pause_button.pack(side="left", expand=True, fill="x", padx=(0, 7))
        self.finish_button = self.button(row, "保存并结束", self.finish)
        self.finish_button.pack(side="left", expand=True, fill="x")
        self.vlabel(inner, "session_note", "继续训练保留已有权重；暂停恢复继续当前回合。", color=MUTED,
                    size=8, wraplength=435, justify="left").pack(anchor="w", pady=(10, 0))
        gpu = tk.Frame(right, bg=PANEL)
        gpu.pack(fill="both", expand=True, padx=18, pady=15)
        self.label(gpu, "显卡独立保护", size=14, bold=True).pack(anchor="w")
        self.vlabel(gpu, "guard", "○  保护待启动", color=MUTED, size=9).pack(anchor="w", pady=(8, 14))
        self.vlabel(gpu, "temperature", "— °C", size=31, bold=True, color=GREEN).pack(anchor="w")
        self.vlabel(gpu, "gpu_details", "功耗 —   ·   显存 —", color=MUTED, size=9).pack(anchor="w", pady=(3, 7))
        self.chart = tk.Canvas(gpu, height=84, bg=PANEL, highlightthickness=0)
        self.chart.pack(fill="x", pady=(5, 7))
        self.label(gpu, "70°C 停机 · 每 250 ms 采样\n监控数据超时 60 秒即停", color=MUTED, size=8, justify="left").pack(anchor="w")
        self.vlabel(gpu, "survival", "当前生存 —   ·   回合 —", size=9).pack(anchor="w", pady=(15, 6))
        self.vlabel(gpu, "resources", "P —   B —   残机 —", color=MUTED, size=9).pack(anchor="w")
        self.vlabel(gpu, "bombs", "炸弹请求 —   ·   实际消耗 —", color=MUTED, size=8).pack(anchor="w", pady=(4, 0))
        self.stop_button = self.button(gpu, "紧急停止  F8", self.emergency_stop, kind="danger")
        self.stop_button.pack(side="bottom", fill="x", pady=(14, 0))

    def _models_page(self, page):
        page.columnconfigure(0, weight=1)
        page.rowconfigure(1, weight=1)
        row = tk.Frame(page, bg=BG)
        row.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self.label(row, "历史模型与自动保存的检查点", size=13, bold=True).pack(side="left")
        self.button(row, "选择文件…", self.browse_model, kind="quiet", pady=6).pack(side="right")
        self.button(row, "刷新", self.refresh_catalog, kind="quiet", pady=6).pack(side="right", padx=6)
        columns = ("name", "steps", "updates", "time", "state")
        frame = self.card(page)
        frame.grid(row=1, column=0, sticky="nsew")
        self.table = ttk.Treeview(frame, columns=columns, show="headings", height=4, style="Model.Treeview", selectmode="browse")
        for name, heading, width in (("name", "模型 / 检查点", 265), ("steps", "累计步数", 95), ("updates", "更新", 60),
                                      ("time", "保存时间", 145), ("state", "状态", 130)):
            self.table.heading(name, text=heading)
            self.table.column(name, width=width, minwidth=55, stretch=name == "name", anchor="w")
        scroll = ttk.Scrollbar(frame, command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.table.pack(side="left", fill="both", expand=True)
        self.table.bind("<<TreeviewSelect>>", self.select_model)
        self.table.bind("<Double-1>", lambda _: self.show_page("train"))
        self.vlabel(page, "model_path", "选择模型查看详情", color=MUTED, size=9, wraplength=870, justify="left").grid(row=2, column=0, sticky="w", pady=(13, 4))
        self.label(page, "每次保存都会保留历史检查点。", color=MUTED, size=9).grid(row=3, column=0, sticky="w")
        actions = tk.Frame(page, bg=BG)
        actions.grid(row=4, column=0, sticky="ew", pady=(13, 0))
        self.button(actions, "使用所选模型  →", lambda: self.show_page("train"), kind="primary").pack(side="left")
        self.play_button = self.button(actions, "模型试玩", lambda: self.start("play"))
        self.play_button.pack(side="left", padx=8)
        self.evaluate_button = self.button(actions, "独立评估 10 局", lambda: self.start("evaluate"))
        self.evaluate_button.pack(side="left")

    def _diagnostics_page(self, page):
        box = self.card(page)
        box.pack(fill="both", expand=True)
        area = tk.Frame(box, bg=PANEL)
        area.pack(fill="both", expand=True, padx=22, pady=19)
        self.label(area, "保护状态与运行记录", size=15, bold=True).pack(anchor="w")
        self.vlabel(area, "backend", "读取训练环境…", wraplength=825, justify="left").pack(anchor="w", pady=(17, 12))
        self.label(area, "紧急停止：F8 / 停止按钮 / 核心温度 ≥70°C\n暂停时，显卡监控和工作进程心跳持续运行。\n保存并结束会正常保存权重；紧急停止直接切断训练进程。",
                   color=MUTED, justify="left").pack(anchor="w", pady=(0, 18))
        row = tk.Frame(area, bg=PANEL)
        row.pack(fill="x")
        self.diagnostic_buttons = []
        for text, mode in (("GPU 自检", "check"), ("仅开启保护监控", "monitor"), ("实机重新校准", "calibrate")):
            btn = self.button(row, text, lambda m=mode: self.start(m))
            btn.pack(side="left", padx=(0, 8))
            self.diagnostic_buttons.append(btn)
        self.vlabel(area, "run_path", "暂无运行记录", color=MUTED, size=9, wraplength=825, justify="left").pack(anchor="w", pady=(22, 12))
        self.log_text = tk.Text(area, height=10, bg=SIDE, fg=MUTED, relief="flat", borderwidth=0,
                               padx=14, pady=12, font=("Consolas", 9), wrap="word", state="disabled")
        self.log_text.pack(fill="both", expand=True)

    def show_page(self, page):
        self.page = page
        for key, widget in self.pages.items():
            widget.pack_forget()
            self.nav[key].configure(bg=PANEL2 if key == page else SIDE, fg=TEXT if key == page else MUTED)
        self.pages[page].pack(fill="both", expand=True)
        self.page_title.configure(text={"train": "训练控制台", "models": "模型库", "diagnostics": "保护与诊断"}[page])

    def refresh_catalog(self):
        previous = self.selected.path if self.selected else None
        if previous is None and not self.preview:
            saved = read_json(ROOT / "ui_state.json").get("selected_model")
            previous = Path(saved) if saved else None
        self.catalog = list_checkpoints()
        self.table.delete(*self.table.get_children())
        index = None
        for i, entry in enumerate(self.catalog):
            meta = entry.metadata
            self.table.insert("", "end", iid=str(i), values=(model_title(entry), f"{meta.get('steps', 0):,}", meta.get("gradient_updates", "—"),
                              meta.get("time", "").replace("T", " ")[:16], model_status(entry)))
            if entry.path == previous:
                index = i
        if index is None:
            index = next((i for i, entry in enumerate(self.catalog) if entry.usable), None)
        if index is not None:
            self.table.selection_set(str(index))
            self.select_model()
        else:
            self.selected = None
        self.last_catalog_refresh = time.monotonic()

    def select_model(self, _=None):
        selection = self.table.selection()
        if not selection:
            return
        self.selected = self.catalog[int(selection[0])]
        meta = self.selected.metadata
        self.variables["model_name"].set(model_title(self.selected))
        self.variables["model_meta"].set(f"{meta.get('steps', 0):,} 步  ·  {meta.get('gradient_updates', 0)} 次更新  ·  {model_status(self.selected)}")
        self.variables["model_path"].set(str(self.selected.path))
        if not self.preview:
            atomic_json(ROOT / "ui_state.json", {"selected_model": str(self.selected.path)})

    def browse_model(self):
        path = filedialog.askopenfilename(parent=self.root, title="选择本项目的模型检查点", initialdir=ROOT / "models", filetypes=[("PPO 模型", "*.zip")])
        if not path:
            return
        target = Path(path).resolve()
        if not target.is_relative_to((ROOT / "models").resolve()):
            messagebox.showerror("模型位置", "请选择本项目 models 目录或其子目录中的模型。", parent=self.root)
            return
        self.refresh_catalog()
        index = next((i for i, entry in enumerate(self.catalog) if entry.path == target), None)
        if index is not None:
            self.table.selection_set(str(index))
            self.table.see(str(index))
            self.select_model()

    def start(self, mode):
        if self.preview:
            return
        path, status = current_session()
        if session_active(status):
            if status.get("mode") == "monitor" and mode != "monitor":
                (path / "STOP").touch()
                self.variables["banner"].set("正在关闭仅监控会话，随后启动你选择的操作…")
                self.root.after(900, lambda: self.start(mode))
            else:
                self.variables["banner"].set("已有会话运行中。使用暂停／恢复，或保存并结束后再启动其他操作。")
            return
        if self.process and self.process.poll() is None:
            return
        if mode in ("resume", "play", "evaluate") and (not self.selected or not self.selected.usable):
            self.show_page("models")
            return
        command = build_command(mode, float(self.minutes.get()), self.selected.path if self.selected else None)
        log = ROOT / "runs" / "launcher.log"
        log.parent.mkdir(exist_ok=True)
        with log.open("a", encoding="utf-8") as stream:
            self.process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1"),
                                             stdout=stream, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        self.variables["banner"].set("正在启动独立显卡保护，预检通过后执行所选操作…")

    def request_pause(self, paused=True):
        if self.preview:
            return
        path, status = current_session()
        if path and session_active(status) and status.get("mode") == "train":
            set_paused(path, paused)
            self.variables["banner"].set("正在暂停：保留采样进度…" if paused else "正在恢复本次训练，继续当前回合…")

    def finish(self):
        if self.preview:
            return
        path, status = current_session()
        if path and session_active(status) and status.get("mode") == "train":
            (path / "FINISH").touch()
            self.variables["banner"].set("等待安全边界并保存模型；显卡保护继续运行。")

    def emergency_stop(self):
        if self.preview:
            return
        path, status = current_session()
        if path and session_active(status):
            (path / "STOP").touch()
            self.variables["banner"].set("已请求紧急停止，后续可从最近一次保存的检查点恢复。")

    def refresh(self):
        path, status = current_session()
        worker = status.get("worker") or {}
        active = session_active(status) and not self.preview
        live_train = active and status.get("mode") == "train"
        self.shown_paused = worker.get("user_paused") is True
        state = status.get("state", "idle")
        labels = {"idle": "等待启动", "starting": "保护预检", "running": "运行中", "monitoring": "保护监控中",
                  "paused": "训练已暂停", "finishing": "保存并结束中", "stopped": "已停止", "complete": "已完成"}
        shown = "界面预览" if self.preview else labels.get(state, "等待启动")
        self.top_badge.configure(text="●  " + shown, fg=GOLD if state == "paused" else ACCENT if state == "stopped" else GREEN)
        if self.preview:
            message = "界面预览 · 运行按钮已禁用，不连接游戏或 GPU。可浏览真实历史模型。"
        elif status:
            if state in ACTIVE_STATES and not active:
                message = "会话连接中断，请查看诊断记录。界面未收到新的状态。"
            elif state == "paused":
                message = "训练已暂停 · 模型、优化器和采样进度保留；剩余时间不减少。点击恢复或按 F7 继续。"
            elif live_train:
                message = worker.get("phase", "准备训练") + " · " + worker.get("message", "")
            else:
                message = labels.get(state, "等待启动") + " · " + status.get("reason", "")
        else:
            message = "尚未运行。选择模型与时长后，由你启动训练。"
        self.variables["banner"].set(message)
        self.variables["time"].set(duration(worker.get("training_seconds", 0)))
        self.variables["time_note"].set(f"目标 {duration(worker.get('budget_seconds', float(self.minutes.get()) * 60))} · 暂停 {duration(worker.get('paused_seconds', 0))}")
        selected_meta = self.selected.metadata if self.selected else {}
        self.variables["steps"].set(f"{worker.get('steps', selected_meta.get('steps', 0)):,}")
        self.variables["steps_note"].set(f"本次新增 {worker.get('session_steps', 0):,} 步")
        self.variables["updates"].set(str(worker.get("updates", selected_meta.get("gradient_updates", 0))))
        self.variables["updates_note"].set("训练进行中" if live_train and not self.shown_paused else "暂停保留进度" if live_train else "上次会话的训练记录" if worker else "所选模型已保存的更新")
        backend = read_json(ROOT / "training_backend.json")
        calibrated = bool(read_json(ROOT / "native_calibration.json").get("restart_verified"))
        resources_ready = bool(read_json(ROOT / "native_resources_calibration.json").get("operator_confirmed"))
        ready = calibrated and resources_ready and bool(backend.get("forward_check"))
        busy = active or bool(self.process and self.process.poll() is None)
        selected_ready = bool(self.selected and self.selected.usable)
        for button, enabled in ((self.resume_button, ready and selected_ready and not busy), (self.new_button, ready and not busy),
                (self.play_button, ready and selected_ready and not busy), (self.evaluate_button, ready and selected_ready and not busy),
                (self.pause_button, live_train and state != "finishing"), (self.finish_button, live_train and state != "finishing"),
                (self.stop_button, active)):
            button.configure(state="normal" if enabled and not self.preview else "disabled")
        self.pause_button.configure(text="恢复本次训练  F7" if self.shown_paused else "暂停训练  F7")
        self.time_select.configure(state="disabled" if busy or self.preview else "readonly")
        for button in self.diagnostic_buttons:
            button.configure(state="disabled" if busy or self.preview else "normal")
        gpu = status.get("gpu") or {}
        self.variables["guard"].set("●  独立保护持续运行" if active else "○  上次会话采样" if gpu else "○  保护待启动")
        if gpu:
            self.variables["temperature"].set(f"{gpu['temperature_c']:.0f} °C")
            self.variables["gpu_details"].set(f"{gpu['power_w']:.1f} W  ·  显存 {gpu['memory_used'] / 2**30:.1f} / {gpu['memory_total'] / 2**30:.0f} GB")
            if gpu.get("monotonic") != self.sample_stamp:
                self.points.append(gpu["temperature_c"])
                self.points = self.points[-180:]
                self.sample_stamp = gpu.get("monotonic")
        self.draw_chart()
        self.variables["survival"].set(f"生存 {worker.get('survival_seconds', 0):.1f} s   ·   第 {worker.get('episodes', 0)} 回合")
        r = worker.get("resources") or {}
        self.variables["resources"].set(f"P {r.get('power', '—')}   B {r.get('bombs', '—')}   残机 {r.get('lives', '—')}")
        self.variables["bombs"].set(f"炸弹请求 {worker.get('bomb_requests', '—')}   ·   实际消耗 {worker.get('bombs_used', '—')}")
        self.variables["backend"].set(f"{backend.get('device', 'GPU 未自检')}  ·  CUDA {backend.get('cuda_runtime', '—')}\n"
                                     f"实机校准 {'通过' if calibrated else '待完成'}  ·  资源读取 {'通过' if resources_ready else '待完成'}")
        self.variables["run_path"].set("记录目录：" + str(path) if path else "暂无运行记录")
        if self.page == "diagnostics":
            compatibility = self.selected.compatibility if self.selected else "未选择模型"
            adaptation = ("旧版模型已迁移" if worker.get("migrated_from") else
                          "炸弹策略已升级" if worker.get("bomb_policy_upgraded") else "无")
            text = (f"状态：{shown}\n原因：{status.get('reason', '—')}\n阶段：{worker.get('phase', '—')}\n"
                    f"累计步数：{worker.get('steps', 0)} / 本次新增：{worker.get('session_steps', 0)}\n"
                    f"模型来源：{worker.get('model_source') or '新建 / 未启动'}\n最近保存：{worker.get('checkpoint', '—')}\n"
                    f"所选模型检查：{compatibility}\n本次加载处理：{adaptation}\n"
                    f"弹幕输入：近场 {worker.get('bullet_observation', {}).get('near', '—')} / 全场 {worker.get('bullet_observation', {}).get('full', '—')} / 已读可见 {worker.get('bullet_observation', {}).get('visible', '—')}\n"
                    f"路径规划：{'启用' if worker.get('planning', {}).get('enabled') else '未启用'} / 可行方向 {worker.get('planning', {}).get('feasible_roots', '—')} / 耗时 {worker.get('planning', {}).get('elapsed_ms', '—')} ms\n"
                    f"敌机接触障碍：{worker.get('planning', {}).get('considered_enemy_bodies', '—')} / 几何未知 {worker.get('planning', {}).get('unknown_body_geometry', '—')}\n"
                    f"本局边缘驻留：{worker.get('loiter_seconds', '—')} 秒 / P 增加：{worker.get('power_gained', '—')} / 点道具增加：{worker.get('point_items_gained', '—')}\n"
                    f"敌机：{worker.get('enemies', '—')} / Boss HP：{worker.get('boss_hp', '—')}/{worker.get('boss_max_hp', '—')} / 累计 Boss 伤害：{worker.get('boss_damage', '—')}\n"
                    f"炸弹策略版本：{worker.get('policy_version', '旧版 / 未启动')}\n策略统计：{worker.get('policy_diagnostics', {})}\n"
                    f"保护通信：{worker.get('lease_diagnostics', {})}\n错误：{worker.get('error', '—')}")
            self.log_text.configure(state="normal")
            self.log_text.delete("1.0", "end")
            self.log_text.insert("1.0", text)
            self.log_text.configure(state="disabled")
        if not active and time.monotonic() - self.last_catalog_refresh > 8:
            self.refresh_catalog()
        self.root.after(250, self.refresh)

    def draw_chart(self):
        self.chart.delete("all")
        w, h = max(100, self.chart.winfo_width()), 84
        for temp in (40, 55, 70):
            y = h - (temp - 25) / 55 * h
            self.chart.create_line(0, y, w, y, fill=LINE if temp < 70 else "#774550", dash=(3, 4))
        self.chart.create_text(w - 3, 6, text="70°C", fill=ACCENT, anchor="ne", font=(FONT, 8))
        if len(self.points) >= 2:
            points = [v for i, temp in enumerate(self.points) for v in (i * w / max(179, len(self.points) - 1), h - (temp - 25) / 55 * h)]
            self.chart.create_line(*points, fill=GREEN, width=2, smooth=True)

    def close(self):
        if not self.preview:
            path, status = current_session()
            if path and session_active(status):
                if status.get("mode") == "train":
                    set_paused(path, True)
                elif status.get("mode") in ("play", "evaluate", "calibrate"):
                    (path / "STOP").touch()
        self.root.destroy()


def launch(config: Config, *, preview=False):
    root = tk.Tk()
    Dashboard(root, config, preview=preview)
    root.mainloop()
