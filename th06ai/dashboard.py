from __future__ import annotations
from .i18n import localized, ui_text

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
        return ui_text('backend.latest_checkpoint')
    if entry.path.parent.name in ("repairs", "checkpoints") or entry.path.stem == "latest":
        return ui_text('backend.checkpoint_0_steps', f"{entry.metadata.get('steps', 0):,}")
    return entry.path.stem


def model_status(entry):
    return ui_text('backend.available') if entry.usable else localized(entry.compatibility)


class Dashboard:
    def __init__(self, root, config: Config, *, preview=False):
        self.root, self.config, self.preview = root, config, preview
        self.selected, self.process, self.page = None, None, "train"
        self.catalog, self.points, self.variables = [], [], {}
        self.sample_stamp, self.last_catalog_refresh, self.shown_paused = None, 0.0, False
        root.title(ui_text('backend.th06nc_ai_training_console') + (ui_text('backend.preview') if preview else ""))
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
        for key, title, detail in (("train", ui_text('backend.training_console'), "TRAINING"), ("models", ui_text('backend.model_library'), "CHECKPOINTS"),
                                   ("diagnostics", ui_text('backend.safety_diagnostics'), "GUARD / LOGS")):
            btn = self.button(sidebar, f"{title}\n{detail}", lambda k=key: self.show_page(k), kind="quiet", anchor="w", justify="left")
            btn.pack(fill="x", padx=13, pady=4)
            self.nav[key] = btn
        foot = tk.Frame(sidebar, bg=SIDE)
        foot.pack(side="bottom", fill="x", padx=24, pady=23)
        self.label(foot, "PPO  ·  CUDA", color=GOLD, size=9).pack(anchor="w")
        self.label(foot, ui_text('backend.preview_2') if self.preview else ui_text('backend.arrow_keys_shift_z_x'), color=MUTED, size=8).pack(anchor="w", pady=(8, 0))
        main = tk.Frame(self.root, bg=BG)
        main.pack(side="left", fill="both", expand=True, padx=27, pady=23)
        header = tk.Frame(main, bg=BG)
        header.pack(fill="x")
        self.page_title = self.label(header, ui_text('backend.training_console'), size=23, bold=True)
        self.page_title.pack(side="left")
        self.top_badge = self.label(header, ui_text('backend.waiting_to_start'), color=GREEN, size=10, padx=13, pady=8, bg=PANEL)
        self.top_badge.pack(side="right")
        self.label(main, ui_text('backend.learned_bullet_hell_policy_models_training_progress'), color=MUTED).pack(anchor="w", pady=(4, 18))
        self.banner = self.vlabel(main, "banner", ui_text('backend.ready_to_start_choose_a_model_and'), color=MUTED,
                                  padx=14, pady=10, bg=PANEL2, wraplength=850, justify="left")
        self.banner.pack(fill="x", pady=(0, 17))
        self.pages = {key: tk.Frame(main, bg=BG) for key in self.nav}
        self._training_page(self.pages["train"])
        self._models_page(self.pages["models"])
        self._diagnostics_page(self.pages["diagnostics"])
        bottom = tk.Frame(main, bg=BG)
        bottom.pack(side="bottom", fill="x", pady=(12, 0))
        self.label(bottom, ui_text('backend.f7_pause_resume_f8_emergency_stop'), color=GOLD, size=9).pack(side="left")
        self.label(bottom, ui_text('backend.closing_this_panel_requests_a_training_pause') if not self.preview else ui_text('backend.preview_no_game_or_gpu_connection'), color=MUTED, size=8).pack(side="right")
        self.show_page("train")

    def _training_page(self, page):
        metrics = tk.Frame(page, bg=BG)
        metrics.pack(fill="x", pady=(0, 15))
        for i, (name, title, value, note) in enumerate((("time", ui_text('backend.active_training_time'), "00:00", ui_text('backend.pauses_are_excluded_from_training_time')),
                  ("steps", ui_text('backend.total_sample_steps'), "—", ui_text('backend.0_steps_this_session')), ("updates", ui_text('backend.parameter_updates'), "—", ui_text('backend.weights_and_optimizer_retained')))):
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
        self.label(inner, ui_text('backend.training_session'), size=14, bold=True).pack(anchor="w")
        self.label(inner, ui_text('backend.selected_model'), color=MUTED, size=9).pack(anchor="w", pady=(12, 4))
        self.vlabel(inner, "model_name", ui_text('backend.no_model_selected'), size=11, bold=True, wraplength=435, justify="left").pack(anchor="w")
        self.vlabel(inner, "model_meta", ui_text('backend.choose_a_checkpoint_in_the_model_library'), color=MUTED, size=9, wraplength=435,
                    justify="left").pack(anchor="w", pady=(4, 3))
        self.button(inner, ui_text('backend.choose_a_checkpoint'), lambda: self.show_page("models"), kind="quiet", pady=4).pack(anchor="w")
        time_row = tk.Frame(inner, bg=PANEL)
        time_row.pack(fill="x", pady=(13, 11))
        self.label(time_row, ui_text('backend.session_duration'), color=MUTED, size=9).pack(side="left")
        self.minutes = tk.StringVar(value=str(int(self.config.minutes)))
        self.time_select = ttk.Combobox(time_row, textvariable=self.minutes, values=(1, 5, 10, 30, 60), width=6,
                                       state="readonly", style="Dark.TCombobox", font=(FONT, 9))
        self.time_select.pack(side="right")
        self.label(time_row, ui_text('backend.minutes'), color=MUTED, size=9).pack(side="right")
        self.resume_button = self.button(inner, ui_text('backend.resume_training'), lambda: self.start("resume"), kind="primary")
        self.resume_button.pack(fill="x")
        self.new_button = self.button(inner, ui_text('backend.train_a_new_model'), lambda: self.start("train"))
        self.new_button.pack(fill="x", pady=(7, 12))
        tk.Frame(inner, bg=LINE, height=1).pack(fill="x", pady=(0, 12))
        row = tk.Frame(inner, bg=PANEL)
        row.pack(fill="x")
        self.pause_button = self.button(row, ui_text('backend.pause_training_f7'), lambda: self.request_pause(not self.shown_paused))
        self.pause_button.pack(side="left", expand=True, fill="x", padx=(0, 7))
        self.finish_button = self.button(row, ui_text('backend.save_and_finish'), self.finish)
        self.finish_button.pack(side="left", expand=True, fill="x")
        self.vlabel(inner, "session_note", ui_text('backend.resume_training_retains_the_weights_unpausing_continues'), color=MUTED,
                    size=8, wraplength=435, justify="left").pack(anchor="w", pady=(10, 0))
        gpu = tk.Frame(right, bg=PANEL)
        gpu.pack(fill="both", expand=True, padx=18, pady=15)
        self.label(gpu, ui_text('backend.independent_gpu_protection'), size=14, bold=True).pack(anchor="w")
        self.vlabel(gpu, "guard", ui_text('backend.watchdog_on_standby'), color=MUTED, size=9).pack(anchor="w", pady=(8, 14))
        self.vlabel(gpu, "temperature", "— °C", size=31, bold=True, color=GREEN).pack(anchor="w")
        self.vlabel(gpu, "gpu_details", ui_text('backend.power_vram'), color=MUTED, size=9).pack(anchor="w", pady=(3, 7))
        self.chart = tk.Canvas(gpu, height=84, bg=PANEL, highlightthickness=0)
        self.chart.pack(fill="x", pady=(5, 7))
        self.label(gpu, ui_text('backend.stop_at_70_c_sample_every_250'), color=MUTED, size=8, justify="left").pack(anchor="w")
        self.vlabel(gpu, "survival", ui_text('backend.current_survival_round'), size=9).pack(anchor="w", pady=(15, 6))
        self.vlabel(gpu, "resources", ui_text('backend.p_b_lives'), color=MUTED, size=9).pack(anchor="w")
        self.vlabel(gpu, "bombs", ui_text('backend.bomb_requests_used'), color=MUTED, size=8).pack(anchor="w", pady=(4, 0))
        self.stop_button = self.button(gpu, ui_text('backend.emergency_stop_f8'), self.emergency_stop, kind="danger")
        self.stop_button.pack(side="bottom", fill="x", pady=(14, 0))

    def _models_page(self, page):
        page.columnconfigure(0, weight=1)
        page.rowconfigure(1, weight=1)
        row = tk.Frame(page, bg=BG)
        row.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self.label(row, ui_text('backend.historical_and_automatically_saved_checkpoints'), size=13, bold=True).pack(side="left")
        self.button(row, ui_text('backend.choose_file'), self.browse_model, kind="quiet", pady=6).pack(side="right")
        self.button(row, ui_text('backend.refresh'), self.refresh_catalog, kind="quiet", pady=6).pack(side="right", padx=6)
        columns = ("name", "steps", "updates", "time", "state")
        frame = self.card(page)
        frame.grid(row=1, column=0, sticky="nsew")
        self.table = ttk.Treeview(frame, columns=columns, show="headings", height=4, style="Model.Treeview", selectmode="browse")
        for name, heading, width in (("name", ui_text('backend.model_checkpoint'), 265), ("steps", ui_text('backend.total_steps'), 95), ("updates", ui_text('backend.updates'), 60),
                                      ("time", ui_text('backend.saved_at'), 145), ("state", ui_text('backend.status'), 130)):
            self.table.heading(name, text=heading)
            self.table.column(name, width=width, minwidth=55, stretch=name == "name", anchor="w")
        scroll = ttk.Scrollbar(frame, command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.table.pack(side="left", fill="both", expand=True)
        self.table.bind("<<TreeviewSelect>>", self.select_model)
        self.table.bind("<Double-1>", lambda _: self.show_page("train"))
        self.vlabel(page, "model_path", ui_text('backend.select_a_model_to_view_details'), color=MUTED, size=9, wraplength=870, justify="left").grid(row=2, column=0, sticky="w", pady=(13, 4))
        self.label(page, ui_text('backend.each_save_preserves_historical_checkpoints'), color=MUTED, size=9).grid(row=3, column=0, sticky="w")
        actions = tk.Frame(page, bg=BG)
        actions.grid(row=4, column=0, sticky="ew", pady=(13, 0))
        self.button(actions, ui_text('backend.use_selected_model'), lambda: self.show_page("train"), kind="primary").pack(side="left")
        self.play_button = self.button(actions, ui_text('backend.play_model'), lambda: self.start("play"))
        self.play_button.pack(side="left", padx=8)
        self.evaluate_button = self.button(actions, ui_text('backend.evaluate_10_rounds'), lambda: self.start("evaluate"))
        self.evaluate_button.pack(side="left")

    def _diagnostics_page(self, page):
        box = self.card(page)
        box.pack(fill="both", expand=True)
        area = tk.Frame(box, bg=PANEL)
        area.pack(fill="both", expand=True, padx=22, pady=19)
        self.label(area, ui_text('backend.safety_status_and_run_logs'), size=15, bold=True).pack(anchor="w")
        self.vlabel(area, "backend", ui_text('backend.reading_training_environment'), wraplength=825, justify="left").pack(anchor="w", pady=(17, 12))
        self.label(area, ui_text('backend.emergency_stop_f8_stop_button_gpu_core'),
                   color=MUTED, justify="left").pack(anchor="w", pady=(0, 18))
        row = tk.Frame(area, bg=PANEL)
        row.pack(fill="x")
        self.diagnostic_buttons = []
        for text, mode in ((ui_text('backend.gpu_check'), "check"), (ui_text('backend.monitor_only'), "monitor"), (ui_text('backend.recalibrate_in_game'), "calibrate")):
            btn = self.button(row, text, lambda m=mode: self.start(m))
            btn.pack(side="left", padx=(0, 8))
            self.diagnostic_buttons.append(btn)
        self.vlabel(area, "run_path", ui_text('backend.no_run_records_yet'), color=MUTED, size=9, wraplength=825, justify="left").pack(anchor="w", pady=(22, 12))
        self.log_text = tk.Text(area, height=10, bg=SIDE, fg=MUTED, relief="flat", borderwidth=0,
                               padx=14, pady=12, font=("Consolas", 9), wrap="word", state="disabled")
        self.log_text.pack(fill="both", expand=True)

    def show_page(self, page):
        self.page = page
        for key, widget in self.pages.items():
            widget.pack_forget()
            self.nav[key].configure(bg=PANEL2 if key == page else SIDE, fg=TEXT if key == page else MUTED)
        self.pages[page].pack(fill="both", expand=True)
        self.page_title.configure(text={"train": ui_text('backend.training_console'), "models": ui_text('backend.model_library'), "diagnostics": ui_text('backend.safety_diagnostics')}[page])

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
        self.variables["model_meta"].set(ui_text('backend.legacy_model_metadata', f"{meta.get('steps', 0):,}", meta.get('gradient_updates', 0), model_status(self.selected)))
        self.variables["model_path"].set(str(self.selected.path))
        if not self.preview:
            atomic_json(ROOT / "ui_state.json", {"selected_model": str(self.selected.path)})

    def browse_model(self):
        path = filedialog.askopenfilename(parent=self.root, title=ui_text('backend.select_a_project_checkpoint'), initialdir=ROOT / "models", filetypes=[(ui_text('backend.ppo_model'), "*.zip")])
        if not path:
            return
        target = Path(path).resolve()
        if not target.is_relative_to((ROOT / "models").resolve()):
            messagebox.showerror(ui_text('backend.model_location'), ui_text('backend.choose_a_model_inside_this_project_s'), parent=self.root)
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
                self.variables["banner"].set(ui_text('backend.ending_the_monitoring_session_before_starting_the'))
                self.root.after(900, lambda: self.start(mode))
            else:
                self.variables["banner"].set(ui_text('backend.a_session_is_running_pause_or_resume'))
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
        self.variables["banner"].set(ui_text('backend.starting_the_independent_gpu_watchdog_the_selected'))

    def request_pause(self, paused=True):
        if self.preview:
            return
        path, status = current_session()
        if path and session_active(status) and status.get("mode") == "train":
            set_paused(path, paused)
            self.variables["banner"].set(ui_text('backend.pausing_with_sampling_progress_retained') if paused else ui_text('backend.resuming_training_in_the_current_round'))

    def finish(self):
        if self.preview:
            return
        path, status = current_session()
        if path and session_active(status) and status.get("mode") == "train":
            (path / "FINISH").touch()
            self.variables["banner"].set(ui_text('backend.waiting_for_a_safe_point_to_save_2'))

    def emergency_stop(self):
        if self.preview:
            return
        path, status = current_session()
        if path and session_active(status):
            (path / "STOP").touch()
            self.variables["banner"].set(ui_text('backend.emergency_stop_requested_resume_from_the_latest'))

    def refresh(self):
        path, status = current_session()
        worker = status.get("worker") or {}
        active = session_active(status) and not self.preview
        live_train = active and status.get("mode") == "train"
        self.shown_paused = worker.get("user_paused") is True
        state = status.get("state", "idle")
        labels = {"idle": ui_text('backend.waiting_to_start_2'), "starting": ui_text('backend.watchdog_preflight'), "running": ui_text('backend.running'), "monitoring": ui_text('backend.monitoring'),
                  "paused": ui_text('backend.training_paused'), "finishing": ui_text('backend.saving_and_finishing'), "stopped": ui_text('backend.stopped'), "complete": ui_text('backend.complete')}
        shown = ui_text('backend.preview_2') if self.preview else labels.get(state, ui_text('backend.waiting_to_start_2'))
        self.top_badge.configure(text="●  " + shown, fg=GOLD if state == "paused" else ACCENT if state == "stopped" else GREEN)
        if self.preview:
            message = ui_text('backend.preview_controls_disabled_browse_saved_checkpoints_and')
        elif status:
            if state in ACTIVE_STATES and not active:
                message = ui_text('backend.session_connection_lost_check_diagnostics_for_the')
            elif state == "paused":
                message = ui_text('backend.training_paused_with_weights_optimizer_and_samples')
            elif live_train:
                message = localized(worker.get("phase", ui_text('backend.ready_to_train'))) + " · " + localized(worker.get("message", ""))
            else:
                message = labels.get(state, ui_text('backend.waiting_to_start_2')) + " · " + localized(status.get("reason", ""))
        else:
            message = ui_text('backend.ready_to_start_choose_a_model_and')
        self.variables["banner"].set(message)
        self.variables["time"].set(duration(worker.get("training_seconds", 0)))
        self.variables["time_note"].set(ui_text('backend.target_0_paused_1', duration(worker.get('budget_seconds', float(self.minutes.get()) * 60)), duration(worker.get('paused_seconds', 0))))
        selected_meta = self.selected.metadata if self.selected else {}
        self.variables["steps"].set(f"{worker.get('steps', selected_meta.get('steps', 0)):,}")
        self.variables["steps_note"].set(ui_text('backend.0_new_steps_this_session', f"{worker.get('session_steps', 0):,}"))
        self.variables["updates"].set(str(worker.get("updates", selected_meta.get("gradient_updates", 0))))
        self.variables["updates_note"].set(ui_text('backend.training') if live_train and not self.shown_paused else ui_text('backend.progress_retained_while_paused') if live_train else ui_text('backend.previous_training_session') if worker else ui_text('backend.updates_saved_in_the_selected_model'))
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
        self.pause_button.configure(text=ui_text('backend.resume_this_session_f7') if self.shown_paused else ui_text('backend.pause_training_f7'))
        self.time_select.configure(state="disabled" if busy or self.preview else "readonly")
        for button in self.diagnostic_buttons:
            button.configure(state="disabled" if busy or self.preview else "normal")
        gpu = status.get("gpu") or {}
        self.variables["guard"].set(ui_text('backend.independent_watchdog_active') if active else ui_text('backend.previous_session_sample') if gpu else ui_text('backend.watchdog_on_standby'))
        if gpu:
            self.variables["temperature"].set(f"{gpu['temperature_c']:.0f} °C")
            self.variables["gpu_details"].set(ui_text('backend.0_w_vram_1_2_gb', f"{gpu['power_w']:.1f}", f"{gpu['memory_used'] / 2 ** 30:.1f}", f"{gpu['memory_total'] / 2 ** 30:.0f}"))
            if gpu.get("monotonic") != self.sample_stamp:
                self.points.append(gpu["temperature_c"])
                self.points = self.points[-180:]
                self.sample_stamp = gpu.get("monotonic")
        self.draw_chart()
        self.variables["survival"].set(ui_text('backend.survival_0_s_round_1', f"{worker.get('survival_seconds', 0):.1f}", worker.get('episodes', 0)))
        r = worker.get("resources") or {}
        self.variables["resources"].set(ui_text('backend.p_0_b_1_lives_2', r.get('power', '—'), r.get('bombs', '—'), r.get('lives', '—')))
        self.variables["bombs"].set(ui_text('backend.bomb_requests_0_used_1', worker.get('bomb_requests', '—'), worker.get('bombs_used', '—')))
        self.variables["backend"].set(ui_text('backend.legacy_backend_readiness', backend.get('device', ui_text('backend.gpu_check_pending')), backend.get('cuda_runtime', '—'), ui_text('backend.passed') if calibrated else ui_text('backend.pending'), ui_text('backend.passed') if resources_ready else ui_text('backend.pending')))
        self.variables["run_path"].set(ui_text('backend.run_directory') + str(path) if path else ui_text('backend.no_run_records_yet'))
        if self.page == "diagnostics":
            compatibility = localized(self.selected.compatibility) if self.selected else ui_text('backend.no_model_selected_2')
            adaptation = (ui_text('backend.legacy_model_migrated') if worker.get("migrated_from") else
                          ui_text('backend.bomb_policy_upgraded') if worker.get("bomb_policy_upgraded") else ui_text('backend.none'))
            text = ui_text(
                'backend.legacy_diagnostics_summary',
                shown, localized(status.get('reason', '—')), localized(worker.get('phase', '—')),
                worker.get('steps', 0), worker.get('session_steps', 0),
                worker.get('model_source') or ui_text('backend.new_not_started'),
                worker.get('checkpoint', '—'), compatibility, adaptation,
                worker.get('bullet_observation', {}).get('near', '—'),
                worker.get('bullet_observation', {}).get('full', '—'),
                worker.get('bullet_observation', {}).get('visible', '—'),
                ui_text('backend.enabled') if worker.get('planning', {}).get('enabled') else ui_text('backend.disabled'),
                worker.get('planning', {}).get('feasible_roots', '—'),
                worker.get('planning', {}).get('elapsed_ms', '—'),
                worker.get('planning', {}).get('considered_enemy_bodies', '—'),
                worker.get('planning', {}).get('unknown_body_geometry', '—'),
                worker.get('loiter_seconds', '—'), worker.get('power_gained', '—'),
                worker.get('point_items_gained', '—'), worker.get('enemies', '—'),
                worker.get('boss_hp', '—'), worker.get('boss_max_hp', '—'),
                worker.get('boss_damage', '—'),
                worker.get('policy_version', ui_text('backend.legacy_not_started')),
                worker.get('policy_diagnostics', {}), worker.get('lease_diagnostics', {}),
                localized(worker.get('error', '—')),
            )
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
