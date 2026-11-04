import tkinter as tk

from th06ai import dashboard
from th06ai.config import Config
from th06ai.i18n import localized, msg, ui_text


def test_locale_translates_archived_watchdog_reason_without_changing_wire_text(monkeypatch):
    reason = msg('backend.ai_error_0', msg('backend.watchdog_authorization_failed',
                                          msg('backend.invalid_authorization_timestamp')))
    monkeypatch.setenv('TH06AI_LANG', 'en')
    assert localized(reason) == 'AI error: GPU watchdog authorization expired. Stopping the AI (Invalid authorization timestamp)'
    assert msg('backend.ai_error_0', 'fixture') == 'AI 报错：fixture'
    assert ui_text('backend.training_console') == 'Training Console'
    monkeypatch.setenv('TH06AI_LANG', 'zh-CN')
    assert localized(reason) == reason
    assert ui_text('backend.training_console') == '训练控制台'
    monkeypatch.delenv('TH06AI_LANG')
    monkeypatch.setattr('th06ai.i18n.locale.getlocale', lambda: ('Chinese (Simplified)_China', '936'))
    assert ui_text('backend.training_console') == '训练控制台'


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
