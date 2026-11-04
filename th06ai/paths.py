"""Shared project location and loopback defaults for the console and launcher."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONSOLE_PORT = 18765
DEFAULT_PREVIEW_PORT = 18766
