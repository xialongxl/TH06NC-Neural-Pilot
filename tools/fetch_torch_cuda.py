"""Download the vendor wheel in bounded ranges and verify the vendor digest.

Only the public PyTorch index and its official download hosts are contacted.
No game files, driver configuration, or proxy settings are touched.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urljoin, urlsplit, urlunsplit

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / ".cache" / "downloads" / "pytorch-cu130"
NAME = "torch-2.14.1+cu130-cp310-cp310-win_amd64.whl"
INDEX = "https://download.pytorch.org/whl/cu130/torch/"


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(urljoin(INDEX, href))


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    curl = shutil.which("curl.exe")
    if not curl:
        raise RuntimeError("Windows curl.exe is unavailable")
    index_file = DEST / "official-index.html"
    subprocess.run([curl, "--fail", "--location", "--silent", "--show-error", "--max-time", "30",
                    "--output", str(index_file), INDEX], check=True)
    parser = Links()
    parser.feed(index_file.read_text(encoding="utf-8"))
    candidates = [url for url in parser.links if unquote(urlsplit(url).path.rsplit("/", 1)[-1]) == NAME]
    if len(candidates) != 1:
        raise RuntimeError("The exact wheel is not uniquely listed by the vendor")
    original = urlsplit(candidates[0])
    expected = parse_qs(original.fragment).get("sha256", [""])[0]
    if not re.fullmatch("[0-9a-fA-F]{64}", expected):
        raise RuntimeError("No SHA-256 digest in the vendor index; refusing download")
    if original.hostname not in ("download.pytorch.org", "download-r2.pytorch.org"):
        raise RuntimeError("Unexpected vendor host")
    url = urlunsplit(("https", "download.pytorch.org", original.path, "", ""))
    headers = subprocess.check_output([curl, "--fail", "--head", "--location", "--silent", "--show-error",
                                       "--max-time", "30", url], text=True)
    sizes = re.findall(r"(?im)^content-length:\s*(\d+)", headers)
    if not sizes:
        raise RuntimeError("The official server did not report the wheel size")
    size = int(sizes[-1])
    if not 100_000_000 < size < 4_000_000_000:
        raise RuntimeError(f"Unexpected wheel size: {size}")
    final = DEST / NAME
    if final.exists() and final.stat().st_size == size:
        digest = hashlib.sha256()
        with final.open("rb") as stream:
            while block := stream.read(4 * 1024 * 1024):
                digest.update(block)
        if digest.hexdigest() == expected:
            print(f"Already verified: {final}", flush=True)
            return
    processes, streams, parts, ranges = [], [], [], []
    count = 8
    for index in range(count):
        start, end = size * index // count, size * (index + 1) // count - 1
        part = DEST / f"part-{index:02d}.bin"
        parts.append(part)
        ranges.append((start, end))
        stream = (DEST / f"part-{index:02d}.log").open("w", encoding="utf-8")
        streams.append(stream)
        processes.append(subprocess.Popen([curl, "--fail", "--location", "--silent", "--show-error",
                                           "--connect-timeout", "15", "--max-time", "600", "--retry", "2",
                                           "--range", f"{start}-{end}", "--output", str(part), url],
                                          stdout=stream, stderr=subprocess.STDOUT))
    started = time.monotonic()
    last = 0
    last_time = started
    try:
        while any(p.poll() is None for p in processes):
            time.sleep(5)
            current = sum(p.stat().st_size for p in parts if p.exists())
            now = time.monotonic()
            speed = (current - last) / max(.001, now - last_time) / 1_000_000
            print(f"{current/size:.1%}  {current/1e6:.0f}/{size/1e6:.0f} MB  {speed:.2f} MB/s", flush=True)
            last, last_time = current, now
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError("A range download failed; inspect .cache/downloads/pytorch-cu130/part-*.log")
        for part, (start, end), process in zip(parts, ranges, processes):
            if process.returncode or part.stat().st_size != end - start + 1:
                raise RuntimeError("Range response size mismatch; refusing wheel installation")
        pending = DEST / "combined.pending"
        digest = hashlib.sha256()
        with pending.open("wb") as out:
            for part in parts:
                with part.open("rb") as source:
                    while block := source.read(4 * 1024 * 1024):
                        digest.update(block)
                        out.write(block)
        actual = digest.hexdigest()
        if actual != expected.lower():
            raise RuntimeError(f"SHA-256 verification failed: {actual}")
        pending.replace(final)
        (DEST / "verified.json").write_text(json.dumps({"wheel": str(final), "sha256": actual,
                                                       "size": size, "source": url,
                                                       "index": INDEX}, indent=2), encoding="utf-8")
        print(f"SHA-256 verified; elapsed {time.monotonic()-started:.1f}s\n{final}", flush=True)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for stream in streams:
            stream.close()


if __name__ == "__main__":
    main()
