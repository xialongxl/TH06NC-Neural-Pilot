"""Offline documentation/index checks. Standard library only; no app imports."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / "docs/code-index.json"


def code_index():
    modules = []
    for path in sorted((ROOT / "th06ai").rglob("*.py")):
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        symbols = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append({"name": node.name, "kind": "function", "line": node.lineno,
                                "signature": ast.unparse(node.args)})
            elif isinstance(node, ast.ClassDef):
                methods = [{"name": m.name, "line": m.lineno, "signature": ast.unparse(m.args)}
                           for m in node.body if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))]
                symbols.append({"name": node.name, "kind": "class", "line": node.lineno,
                                "bases": [ast.unparse(b) for b in node.bases], "methods": methods})
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append("." * node.level + (node.module or ""))
        modules.append({"path": path.relative_to(ROOT).as_posix(),
                        "sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
                        "summary": (ast.get_docstring(tree) or "").split("\n")[0],
                        "imports": sorted(set(imports)), "symbols": symbols})
    return {"schema_version": 1, "generated_by": "python tools/check_docs.py --update-index",
            "hash_normalization": "UTF-8 without BOM; universal LF newlines for checkout portability",
            "scope": "Static Python AST, including imports inside functions; no runtime dependency proof.",
            "modules": modules,
            "tests": [p.relative_to(ROOT).as_posix() for p in sorted((ROOT / "tests").glob("test_*.py"))]}


def prose(source):
    lines = []; fence = None
    for line in source.splitlines():
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker:
            kind = marker.group(1)[0]
            if fence is None:
                fence = kind
            elif fence == kind:
                fence = None
            continue
        if fence is None:
            lines.append(line)
    if fence:
        raise ValueError("unclosed Markdown code fence")
    return "\n".join(lines)


def anchors(source):
    result = set(); counts = {}
    for heading in re.findall(r"^#{1,6}\s+(.+?)\s*#*\s*$", prose(source), re.M):
        slug = re.sub(r"[^\w\s-]", "", heading.lower())
        slug = re.sub(r"\s", "-", slug)
        count = counts.get(slug, 0); counts[slug] = count + 1
        result.add(slug if count == 0 else f"{slug}-{count}")
    return result


def documents():
    names = ("README.md", "README.zh-CN.md", "AGENTS.md", "CONTRIBUTING.md", "DISCLAIMER.md", "THIRD_PARTY_NOTICES.md",
             "frontend/README.md", "native/README.md")
    return [ROOT / name for name in names] + sorted((ROOT / "docs").glob("*.md")) + sorted((ROOT / ".github").rglob("*.md"))


def validate():
    errors = []; links = 0
    expected = code_index()
    try:
        actual = json.loads(INDEX.read_text(encoding="utf-8"))
        if actual != expected:
            errors.append("code-index.json is stale: run --update-index and review the diff")
    except (OSError, ValueError) as error:
        errors.append(f"code-index.json: {error}")
    for path in documents():
        try:
            text = prose(path.read_text(encoding="utf-8-sig"))
            for target in re.findall(r"!?\[[^\]\n]*\]\(([^\)\n]+)\)", text):
                links += 1
                target = target.strip().strip("<>")
                parsed = urlsplit(target)
                if parsed.scheme or target.startswith("//"):
                    continue
                dest = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
                if not dest.is_relative_to(ROOT) or not dest.exists():
                    errors.append(f"{path.relative_to(ROOT)}: missing/outside link {target}")
                elif parsed.fragment and dest.suffix.lower() == ".md":
                    if unquote(parsed.fragment) not in anchors(dest.read_text(encoding="utf-8-sig")):
                        errors.append(f"{path.relative_to(ROOT)}: missing heading {target}")
        except (OSError, ValueError) as error:
            errors.append(f"{path.relative_to(ROOT)}: {error}")
    try:
        context = json.loads((ROOT / "docs/ai-context.json").read_text(encoding="utf-8"))
        module_map = {m["path"]: m for m in expected["modules"]}
        component_ids = {c["id"] for c in context["components"]}
        for component in context["components"]:
            path = ROOT / component["path"]
            if not path.is_file():
                errors.append(f"AI context missing file: {component['path']}")
            for dependency in component["depends_on"]:
                if dependency not in component_ids:
                    errors.append(f"AI context unknown dependency: {dependency}")
            if component["path"] in module_map:
                names = {s["name"] for s in module_map[component["path"]]["symbols"]}
                for symbol in component["symbols"]:
                    if symbol not in names:
                        errors.append(f"AI context missing symbol: {component['path']}:{symbol}")
        for group, paths in context["test_groups"].items():
            for path in paths:
                if not (ROOT / path).is_file():
                    errors.append(f"AI context missing test ({group}): {path}")
        grouped = {p for paths in context["test_groups"].values() for p in paths}
        for path in set(expected["tests"]) - grouped:
            errors.append(f"AI context missing test-group assignment: {path}")
        code_map = (ROOT / "docs/code-map.md").read_text(encoding="utf-8")
        for path in module_map:
            if "../" + path not in code_map:
                errors.append(f"Code map missing Python module: {path}")
        example = json.loads((ROOT / "config.example.json").read_text(encoding="utf-8"))
        if example["episode_seconds"] != 0 or example["safety"]["temperature_limit_c"] != 70:
            errors.append("Review documented sample defaults: episode_seconds / core temperature")
        if not example["game_exe"].endswith("th06nc.exe"):
            errors.append("config.example.json must point to the executable, not just a folder")
        for path in (ROOT / "LICENSE", ROOT / "DISCLAIMER.md", ROOT / "CONTRIBUTING.md"):
            if not path.is_file():
                errors.append(f"Missing release document: {path.name}")
    except (OSError, ValueError, KeyError) as error:
        errors.append(f"Context/config: {error}")
    return {"ok": not errors, "documents": len(documents()), "local_and_external_links_parsed": links,
            "python_modules_indexed": len(expected["modules"]), "test_modules_indexed": len(expected["tests"]),
            "external_links_fetched": False, "game_or_gpu_imports": False, "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update-index", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.update_index:
        INDEX.parent.mkdir(exist_ok=True)
        INDEX.write_text(json.dumps(code_index(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result = validate()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
