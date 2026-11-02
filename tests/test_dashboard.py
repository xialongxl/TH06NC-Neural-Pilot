import tkinter as tk

from th06ai import dashboard
from th06ai.config import Config


def test_model_actions_remain_visible_at_minimum_window_size_and_preview_is_inert(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    monkeypatch.setattr(dashboard, "list_checkpoints", lambda: [])
    monkeypatch.setattr(dashboard, "current_session", lambda: (None, {}))
    def forbidden(*args, **kwargs):
        raise AssertionError("preview must not launch workers or issue game controls")
    monkeypatch.setattr(dashboard.subprocess, "Popen", forbidden)
    root = tk.Tk()
    root.withdraw()
    try:
        app = dashboard.Dashboard(root, Config(game_exe="unused"), preview=True)
        root.geometry("1080x800")
        app.show_page("models")
        root.attributes("-alpha", 0)
        root.attributes("-disabled", True)
        root.deiconify()
        root.update()
        root.update_idletasks()
        page = app.pages["models"]
        for button in (app.play_button, app.evaluate_button):
            assert button.winfo_height() >= 35, (root.geometry(), page.winfo_geometry(), button.master.winfo_geometry())
            top = button.winfo_rooty() - page.winfo_rooty()
            assert 0 <= top and top + button.winfo_height() <= page.winfo_height()
        for mode in ("train", "resume", "play", "evaluate", "check", "monitor", "calibrate"):
            app.start(mode)
        app.request_pause()
        app.finish()
        app.emergency_stop()
        assert not list(tmp_path.iterdir()), "preview must not create markers or modify UI/model state"
    finally:
        root.destroy()
