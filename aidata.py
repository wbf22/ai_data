#!/usr/bin/env python3
"""Capture an opencode session as AI training data.

Usage:
    python3 aidata.py start            # zip the repo per .ai_data_config -> src.zip
    python3 aidata.py end [--session ID] [--no-sanitize]
                                       # export the session, copy src.zip, label it, save rubric

The "start" step reads .ai_data_config (gitignore-style) from the current repo
and writes src.zip. The "end" step uses `opencode export` (CLI, no TUI needed)
to grab the current project's most recent session, copies it and src.zip into
data/<session>/, sanitizes secrets/PII out of every captured artifact, collects
the repo's license files into data/<session>/licenses/, then runs an interactive
labeling rubric.

Sanitization is on by default and reports what it redacted (sanitize-report.*).
Pass --no-sanitize to keep the raw data.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from pathspec.patterns.gitignore.spec import GitIgnoreSpecPattern

REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = REPO_ROOT / "data"
CONFIG_NAME = ".ai_data_config"
ZIP_NAME = "src.zip"
# where the .opencode/plugin/aidata-capture.ts plugin drops system prompt + tool defs
CAPTURE_DIR = Path(os.environ.get("AIDATA_CAPTURE_DIR", Path.home() / ".local" / "share" / "opencode" / "aidata-capture"))

PROBLEM_ITEMS = [
    ("broken_code", "broken code (if 'no' implies code runs without problems or was otherwise acceptable)"),
    ("skipped_prompt", "skipped part of prompt"),
    ("went_beyond", "went beyond prompt"),
    ("bad_code_style", "bad code or code style that works"),
    ("dishonest_final", "dishonest final message"),
    ("final_too_long", "final message too long"),
    ("final_too_short", "final message too short"),
    ("other", "other"),
]

EXCEPTIONAL_ITEMS = [
    ("clever_solution", "clever solution"),
    ("first_try", "first try"),
    ("exceptionally_well_integrated", "exceptionally well integrated in existing code"),
    ("other", "other"),
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def ts_from_ms(ms: int | float) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone().isoformat(timespec="seconds")


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def die(msg: str) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(1)


# --------------------------------------------------------------------------- #
# start: build src.zip from .ai_data_config
# --------------------------------------------------------------------------- #
def load_rules(config: Path) -> list[tuple[bool, GitIgnoreSpecPattern]]:
    rules: list[tuple[bool, GitIgnoreSpecPattern]] = []
    for raw in config.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        if negate:
            line = line[1:].strip()
        if not line:
            continue
        try:
            rules.append((negate, GitIgnoreSpecPattern(line)))
        except Exception as exc:  # noqa: BLE001
            die(f"bad pattern {line!r} in {config}: {exc}")
    return rules


def is_included(rel: str, rules: list[tuple[bool, GitIgnoreSpecPattern]]) -> bool:
    include = False
    for negate, pattern in rules:
        if pattern.match_file(rel):
            include = not negate
    return include


def collect_files(
    repo: Path,
    rules: list[tuple[bool, GitIgnoreSpecPattern]],
    exclude_files: set[Path],
    exclude_dirs: set[Path],
) -> list[str]:
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(repo):
        here = Path(dirpath)
        dirnames[:] = [d for d in dirnames if d != ".git" and (here / d) not in exclude_dirs]
        for name in filenames:
            path = here / name
            if path in exclude_files:
                continue
            rel = path.relative_to(repo).as_posix()
            if is_included(rel, rules):
                found.append(rel)
    return sorted(found)


def cmd_start(args: argparse.Namespace) -> None:
    repo = Path(args.repo).expanduser().resolve()
    config = repo / CONFIG_NAME
    if not config.exists():
        die(f"no {CONFIG_NAME} found in {repo} (create one listing what to include)")

    rules = load_rules(config)
    zip_path = (repo / args.output).resolve()

    exclude_files = {zip_path}
    exclude_dirs: set[Path] = set()
    if DATA_ROOT.resolve().is_relative_to(repo):
        exclude_dirs.add(DATA_ROOT.resolve())

    files = collect_files(repo, rules, exclude_files, exclude_dirs)
    if not files:
        die(f"{CONFIG_NAME} matched no files in {repo}")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            zf.write(repo / rel, arcname=rel)

    size = zip_path.stat().st_size
    print(f"wrote {zip_path} ({len(files)} files, {size:,} bytes)")
    for rel in files:
        print(f"  + {rel}")


# --------------------------------------------------------------------------- #
# sanitize: redact secrets and PII from captured artifacts
# --------------------------------------------------------------------------- #
TRANSCRIPT = frozenset({"transcript"})
SOURCE = frozenset({"source"})
BOTH = frozenset({"transcript", "source"})


def _assignment_repl(match: re.Match, token: str) -> str:
    return f"{match.group(1)}{match.group(2)}{match.group(3)}{token}"


# Keys whose values are secrets even when the value doesn't look like one
# (e.g. structured JSON where the key and value are separate fields).
SECRET_KEY_RE = re.compile(
    r"(?i)(?:^|[_-])(api[_-]?key|apikey|secret|token|password|passwd|pwd|access[_-]?key|private[_-]?key|client[_-]?secret|credential)(?:$|[_-])"
)


def is_secret_key(key: str) -> bool:
    return bool(SECRET_KEY_RE.search(key))


REDACTION_RULES: list[dict] = [
    {
        "name": "private_key",
        "pattern": re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----", re.DOTALL),
        "scopes": BOTH,
    },
    {"name": "anthropic_key", "pattern": re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b"), "scopes": BOTH},
    {"name": "openai_key", "pattern": re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"), "scopes": BOTH},
    {"name": "github_pat", "pattern": re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "scopes": BOTH},
    {"name": "github_token", "pattern": re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"), "scopes": BOTH},
    {"name": "gitlab_pat", "pattern": re.compile(r"\bglpat-[A-Za-z0-9_\-]{20,}\b"), "scopes": BOTH},
    {"name": "slack_token", "pattern": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"), "scopes": BOTH},
    {"name": "aws_access_key", "pattern": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "scopes": BOTH},
    {"name": "google_api_key", "pattern": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "scopes": BOTH},
    {"name": "jwt", "pattern": re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b"), "scopes": BOTH},
    {"name": "bearer_token", "pattern": re.compile(r"(?i)\bBearer\s+[A-Za-z0-9_\-\.=]{20,}"), "scopes": BOTH},
    {
        "name": "secret_assignment",
        "pattern": re.compile(
            r"(?i)([\"']?\b(?:api[_-]?key|secret[_-]?key|client[_-]?secret|access[_-]?key|secret|token|password|passwd|pwd)\b[\"']?)"
            r"(\s*[:=]\s*)([\"']?)([A-Za-z0-9_\-\.=+/]{12,})"
        ),
        "scopes": BOTH,
        "repl": _assignment_repl,
    },
    {"name": "email", "pattern": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "scopes": TRANSCRIPT},
    {"name": "home_path", "pattern": re.compile(r"/(?:home|Users)/[^/\s\"'\\]+"), "scopes": TRANSCRIPT},
]


def redact(text: str, scopes: frozenset[str]) -> tuple[str, Counter]:
    """Apply every rule matching `scopes`, returning the new text and per-rule counts."""
    counts: Counter = Counter()
    for rule in REDACTION_RULES:
        if not (rule["scopes"] & scopes):
            continue
        name = rule["name"]
        token = f"[REDACTED:{name}]"
        rule_repl = rule.get("repl")

        def callback(match, name=name, token=token, rule_repl=rule_repl):
            counts[name] += 1
            if rule_repl is not None:
                return rule_repl(match, token)
            return token

        text = rule["pattern"].sub(callback, text)
    return text, counts


def sanitize_data(obj, scopes: frozenset[str]):
    """Recursively redact every string in a JSON-like structure."""
    counts: Counter = Counter()

    def walk(value):
        if isinstance(value, str):
            new, found = redact(value, scopes)
            counts.update(found)
            return new
        if isinstance(value, dict):
            result = {}
            for key, val in value.items():
                if isinstance(val, str) and val and is_secret_key(key) and not val.startswith("[REDACTED:"):
                    counts["secret_key"] += 1
                    result[key] = "[REDACTED:secret_key]"
                else:
                    result[key] = walk(val)
            return result
        if isinstance(value, list):
            return [walk(v) for v in value]
        return value

    return walk(obj), counts


def decode_text(raw: bytes) -> str | None:
    """Return decoded text, or None if the bytes look binary / are not UTF-8."""
    if b"\x00" in raw:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def sanitize_zip(zip_path: Path, scopes: frozenset[str]) -> dict[str, Counter]:
    """Rewrite a zip in place, redacting text entries. Returns per-entry counts."""
    per_entry: dict[str, Counter] = {}
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(zip_path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            raw = zin.read(info.filename)
            text = decode_text(raw)
            if text is None:
                zout.writestr(info, raw)
                continue
            new, counts = redact(text, scopes)
            if counts:
                per_entry[info.filename] = counts
            zout.writestr(info, new.encode("utf-8"))
    tmp.replace(zip_path)
    return per_entry


def render_sanitize_markdown(report: dict) -> str:
    lines = [f"# Sanitize report - {report['session_id']}", ""]
    lines.append(f"- **Generated:** {report['generated_at']}")
    lines.append(f"- **Total redactions:** {sum(report['totals'].values())}")
    lines.append("")
    lines.append("Redacted spans are replaced with `[REDACTED:<rule>]`.")
    lines.append("")
    lines.append("| Artifact | Rule | Count |")
    lines.append("| --- | --- | ---: |")
    for artifact, counts in sorted(report["artifacts"].items()):
        for rule, count in sorted(counts.items()):
            lines.append(f"| `{artifact}` | {rule} | {count} |")
    lines.append("")
    return "\n".join(lines)


def write_sanitize_report(outdir: Path, report: dict) -> None:
    serializable = {
        "session_id": report["session_id"],
        "generated_at": report["generated_at"],
        "redaction_token_format": "[REDACTED:<rule>]",
        "totals": dict(report["totals"]),
        "artifacts": report["artifacts"],
    }
    (outdir / "sanitize-report.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    (outdir / "sanitize-report.md").write_text(render_sanitize_markdown(serializable), encoding="utf-8")


# --------------------------------------------------------------------------- #
# licenses: collect the source's licenses into <session>/licenses/
# --------------------------------------------------------------------------- #
LICENSE_FILE_RE = re.compile(r"^(licen[cs]e|copying|copyright|notice|unlicense)([.\-].*)?$", re.IGNORECASE)

LICENSE_SIGNATURES: list[tuple[str, list[str]]] = [
    ("MIT", ["Permission is hereby granted, free of charge"]),
    ("Apache-2.0", ["Apache License", "Version 2.0"]),
    ("AGPL-3.0", ["GNU AFFERO GENERAL PUBLIC LICENSE", "Version 3"]),
    ("LGPL-3.0", ["GNU LESSER GENERAL PUBLIC LICENSE", "Version 3"]),
    ("GPL-3.0", ["GNU GENERAL PUBLIC LICENSE", "Version 3"]),
    ("GPL-2.0", ["GNU GENERAL PUBLIC LICENSE", "Version 2"]),
    ("BSD-3-Clause", ["Redistribution and use in source and binary forms", "Neither the name"]),
    ("BSD-2-Clause", ["Redistribution and use in source and binary forms"]),
    ("MPL-2.0", ["Mozilla Public License Version 2.0"]),
    ("ISC", ["Permission to use, copy, modify, and/or distribute this software"]),
    ("Unlicense", ["free and unencumbered software released into the public domain"]),
    ("CC0-1.0", ["CC0 1.0 Universal"]),
]

SPDX_RE = re.compile(r"SPDX-License-Identifier:\s*([^\n\r*/]+)")


def is_license_filename(path: str) -> bool:
    parts = PurePosixPath(path).parts
    if any(part.upper() == "LICENSES" for part in parts[:-1]):
        return True
    return bool(LICENSE_FILE_RE.match(PurePosixPath(path).name))


def detect_license_text(text: str) -> list[str]:
    low = text.lower()
    return [name for name, phrases in LICENSE_SIGNATURES if all(p.lower() in low for p in phrases)]


def detect_spdx(text: str) -> list[str]:
    seen: list[str] = []
    for match in SPDX_RE.findall(text[:4096]):
        value = match.strip()
        if value and value not in seen:
            seen.append(value)
    return seen


def render_licenses_markdown(index: dict) -> str:
    lines = [f"# Licenses - {index['session_id']}", ""]
    lines.append(f"- **Generated:** {index['generated_at']}")
    lines.append("")
    if index["summary"]:
        lines.append("## Detected summary")
        lines.append("")
        lines.append("| License | Occurrences |")
        lines.append("| --- | ---: |")
        for name, count in sorted(index["summary"].items()):
            lines.append(f"| {name} | {count} |")
        lines.append("")
    if index["license_files"]:
        lines.append("## License files")
        lines.append("")
        for entry in index["license_files"]:
            detected = ", ".join(entry["detected"]) or "unrecognized"
            lines.append(f"- `{entry['path']}` - {detected}")
        lines.append("")
    if index["spdx_identifiers"]:
        lines.append("## SPDX identifiers in source")
        lines.append("")
        for entry in index["spdx_identifiers"]:
            lines.append(f"- `{entry['path']}` - {entry['license']}")
        lines.append("")
    return "\n".join(lines)


def collect_licenses(zip_path: Path, outdir: Path, session_id: str) -> dict | None:
    """Scan a source zip for license files and SPDX headers, copying them into licenses/."""
    license_files: list[dict] = []
    spdx_identifiers: list[dict] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            raw = zf.read(name)
            text = decode_text(raw)
            if text is None:
                continue
            if is_license_filename(name):
                dest = outdir / "licenses" / "files" / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(raw)
                license_files.append({"path": name, "detected": detect_license_text(text)})
            else:
                for lic in detect_spdx(text):
                    spdx_identifiers.append({"path": name, "license": lic})

    if not license_files and not spdx_identifiers:
        return None

    summary: Counter = Counter()
    for entry in license_files:
        for lic in entry["detected"] or ["unknown"]:
            summary[lic] += 1
    for entry in spdx_identifiers:
        summary[entry["license"]] += 1

    index = {
        "session_id": session_id,
        "generated_at": now_iso(),
        "license_files": license_files,
        "spdx_identifiers": spdx_identifiers,
        "summary": dict(summary),
    }
    licdir = outdir / "licenses"
    (licdir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (licdir / "LICENSES.md").write_text(render_licenses_markdown(index), encoding="utf-8")
    return index


# --------------------------------------------------------------------------- #
# end: export session + copy artifacts + label
# --------------------------------------------------------------------------- #
def resolve_session(repo: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    proc = run(["opencode", "session", "list"], repo)
    if proc.returncode != 0:
        die(f"`opencode session list` failed: {proc.stderr.strip()}")
    for line in proc.stdout.splitlines():
        m = re.match(r"^(ses_[A-Za-z0-9]+)", line.strip())
        if m:
            return m.group(1)
    die("no sessions found for this project; pass --session <id>")


def export_session(repo: Path, session_id: str) -> dict:
    cmd = ["opencode", "export", session_id]
    # opencode is a Bun binary: it exits before flushing a *piped* stdout, which
    # truncates large exports. Redirect to a real file so the write flushes.
    with tempfile.TemporaryFile("w+", encoding="utf-8") as tmp:
        proc = subprocess.run(cmd, cwd=repo, stdout=tmp, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            die(f"`opencode export {session_id}` failed: {proc.stderr.strip()}")
        tmp.seek(0)
        raw = tmp.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        die(f"could not parse export output as JSON: {exc}")


def fence(text: str) -> str:
    ticks = "```"
    while ticks in text:
        ticks += "`"
    return f"{ticks}\n{text.rstrip()}\n{ticks}"


def render_part(part: dict) -> str:
    kind = part.get("type")
    if kind == "text":
        return part.get("text", "").strip()
    if kind == "reasoning":
        body = part.get("text", "").strip()
        return f"### Reasoning\n\n{body}" if body else ""
    if kind == "tool":
        tool = part.get("tool", "tool")
        state = part.get("state", {}) or {}
        status = state.get("status", "")
        chunks = [f"**Tool: `{tool}`** _({status})_"]
        if state.get("input") is not None:
            chunks.append("Input:\n\n" + fence(json.dumps(state["input"], indent=2, ensure_ascii=False)))
        if state.get("output") is not None:
            out = state["output"]
            if not isinstance(out, str):
                out = json.dumps(out, indent=2, ensure_ascii=False)
            chunks.append("Output:\n\n" + fence(out))
        return "\n\n".join(chunks)
    if kind in ("step-start", "step-finish", "snapshot", "patch"):
        return ""
    text = part.get("text")
    return text.strip() if isinstance(text, str) else ""


def render_markdown(data: dict) -> str:
    info = data.get("info", {}) or {}
    lines: list[str] = [f"# Session {info.get('id', '')}", ""]
    meta = {
        "Title": info.get("title"),
        "Directory": info.get("directory") or info.get("path"),
        "Agent": info.get("agent"),
        "Version": info.get("version"),
    }
    created = (info.get("time") or {}).get("created")
    if created:
        meta["Created"] = ts_from_ms(created)
    for key, value in meta.items():
        if value:
            lines.append(f"- **{key}:** {value}")
    lines.append("")

    for message in data.get("messages", []):
        mi = message.get("info", {}) or {}
        role = (mi.get("role") or "?").capitalize()
        if mi.get("modelID"):
            role += f" ({mi.get('providerID', '')}/{mi.get('modelID')})"
        lines.append(f"## {role}")
        lines.append("")
        for part in message.get("parts", []) or []:
            rendered = render_part(part)
            if rendered:
                lines.append(rendered)
                lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def ask_yes_no(question: str) -> bool:
    while True:
        answer = input(f"{question} [y/n]: ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  please answer y or n")


def ask_multiselect(items: list[tuple[str, str]]) -> list[str]:
    for i, (_, label) in enumerate(items, 1):
        print(f"  {i}. {label}")
    while True:
        raw = input("  select by number (comma separated), blank = none: ").strip()
        if not raw:
            return []
        try:
            nums = {int(x) for x in re.split(r"[,\s]+", raw) if x}
        except ValueError:
            print("  invalid input")
            continue
        if not nums or any(n < 1 or n > len(items) for n in nums):
            print("  number out of range")
            continue
        return [items[n - 1][0] for n in sorted(nums)]


def ask_text(prompt: str) -> str:
    while True:
        print(prompt)
        answer = input("> ").strip()
        if answer:
            return answer
        print("  a description is required")


def collect_section(title: str, question: str, items: list[tuple[str, str]]) -> dict:
    print(f"\n== {title} ==")
    flagged = ask_yes_no(question)
    selected: list[str] = []
    descriptions: dict[str, str] = {}
    if flagged:
        selected = ask_multiselect(items)
        for key, label in items:
            if key in selected:
                descriptions[key] = ask_text(f"  describe: {label}")

    section: dict = {"flagged": flagged, "items": {}}
    for key, label in items:
        if flagged and key in selected:
            section["items"][key] = {"label": label, "marked": "yes", "description": descriptions[key]}
        else:
            section["items"][key] = {"label": label, "marked": "no", "description": None}
    return section


def render_rubric_markdown(rubric: dict) -> str:
    labels = rubric["labels"]
    lines = [f"# Rubric - {rubric['session']['id']}", ""]
    lines.append(f"- **Directory:** {rubric['session'].get('directory', '')}")
    lines.append(f"- **Model:** {rubric['session'].get('model', '')}")
    lines.append(f"- **Labeled at:** {rubric['labeled_at']}")
    lines.append("")

    for section_key, heading in (("problems", "Any problems?"), ("exceptional", "Anything done exceptionally well?")):
        section = labels[section_key]
        verdict = "YES" if section["flagged"] else "NO"
        lines.append(f"## {heading} - {verdict}")
        lines.append("")
        for item in section["items"].values():
            mark = item["marked"].upper()
            entry = f"- [{mark}] {item['label']}"
            if item.get("description"):
                entry += f" - {item['description']}"
            lines.append(entry)
        lines.append("")

    lines.append("## Overall description")
    lines.append("")
    lines.append(labels["overall_description"])
    lines.append("")
    return "\n".join(lines)


def copy_tool_outputs(data: dict, outdir: Path) -> dict[str, str]:
    """Copy full text of truncated tool outputs.

    `opencode export` clips long tool outputs in the part and records the full
    text at `state.metadata.outputPath`; that file lives outside the export and
    would otherwise be lost.
    """
    copied: dict[str, str] = {}
    todir = outdir / "tool-output"
    for message in data.get("messages", []):
        for part in message.get("parts", []) or []:
            if part.get("type") != "tool":
                continue
            metadata = (part.get("state") or {}).get("metadata") or {}
            output_path = metadata.get("outputPath")
            if not output_path:
                continue
            source = Path(output_path)
            if not source.is_file():
                continue
            todir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, todir / source.name)
            copied[part.get("callID") or source.name] = f"tool-output/{source.name}"
    if copied:
        (todir / "index.json").write_text(json.dumps(copied, indent=2), encoding="utf-8")
    return copied


def copy_capture_artifacts(session_id: str, outdir: Path) -> dict[str, str]:
    """Copy the rendered system prompt and tool definitions captured by the plugin.

    The plugin snapshots tool definitions per session (tools-<id>.json).
    """
    copied: dict[str, str] = {}
    system_file = CAPTURE_DIR / f"system-{session_id}.json"
    if system_file.is_file():
        shutil.copy2(system_file, outdir / "system-prompt.json")
        copied["system_prompt"] = "system-prompt.json"
    tool_file = CAPTURE_DIR / f"tools-{session_id}.json"
    if tool_file.is_file():
        shutil.copy2(tool_file, outdir / "tool-definitions.json")
        copied["tool_definitions"] = "tool-definitions.json"
    return copied


def cmd_end(args: argparse.Namespace) -> None:
    repo = Path(args.repo).expanduser().resolve()
    session_id = resolve_session(repo, args.session)

    print(f"exporting session {session_id} ...")
    data = export_session(repo, session_id)

    stem = f"session-{session_id[:8]}"
    outdir = DATA_ROOT / stem
    outdir.mkdir(parents=True, exist_ok=True)

    sanitize_report: dict = {
        "session_id": session_id,
        "generated_at": now_iso(),
        "artifacts": {},
        "totals": Counter(),
    }

    def record(artifact: str, counts: Counter) -> None:
        if counts:
            sanitize_report["artifacts"][artifact] = dict(counts)
            sanitize_report["totals"].update(counts)

    # Copy artifacts while the session data still holds the real (unredacted)
    # paths, then sanitize the copies.
    src_zip = repo / ZIP_NAME
    src_copied = src_zip.exists()
    if src_copied:
        shutil.copy2(src_zip, outdir / ZIP_NAME)
        print(f"copied {ZIP_NAME}")
    else:
        print(f"warning: {ZIP_NAME} not found in {repo}; run `aidata.py start` first")

    tool_outputs = copy_tool_outputs(data, outdir)
    if tool_outputs:
        print(f"copied {len(tool_outputs)} truncated tool output(s)")

    capture = copy_capture_artifacts(session_id, outdir)
    if capture:
        print("copied " + ", ".join(capture.values()))
    else:
        print("note: no system-prompt/tool-definitions found (is the aidata plugin installed and loaded?)")

    if args.sanitize:
        data, counts = sanitize_data(data, TRANSCRIPT)
        record(f"{stem}.json", counts)
        if src_copied:
            for name, entry_counts in sanitize_zip(outdir / ZIP_NAME, SOURCE).items():
                record(f"{ZIP_NAME}:{name}", entry_counts)
        todir = outdir / "tool-output"
        if todir.is_dir():
            for path in sorted(todir.iterdir()):
                text = decode_text(path.read_bytes())
                if text is None:
                    continue
                new, counts = redact(text, TRANSCRIPT)
                if counts:
                    path.write_text(new, encoding="utf-8")
                    record(f"tool-output/{path.name}", counts)
        for fname in capture.values():
            path = outdir / fname
            new, counts = redact(path.read_text(encoding="utf-8"), TRANSCRIPT)
            path.write_text(new, encoding="utf-8")
            record(fname, counts)

    json_path = outdir / f"{stem}.json"
    md_path = outdir / f"{stem}.md"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(data), encoding="utf-8")
    print(f"wrote {md_path.name} and {json_path.name}")

    license_info = None
    if src_copied:
        license_info = collect_licenses(outdir / ZIP_NAME, outdir, session_id)
        if license_info:
            print(f"collected {len(license_info['license_files'])} license file(s) -> licenses/")
        else:
            print("no license files or SPDX identifiers found in source")

    if args.sanitize:
        write_sanitize_report(outdir, sanitize_report)
        total = sum(sanitize_report["totals"].values())
        if total:
            print(f"sanitized {total} item(s) -> sanitize-report.md")
        else:
            print("sanitize: nothing sensitive found")

    info = data.get("info", {}) or {}
    messages = data.get("messages", []) or []
    model = ""
    for message in messages:
        mi = message.get("info", {}) or {}
        if mi.get("modelID"):
            model = f"{mi.get('providerID', '')}/{mi.get('modelID')}"
            break

    labels = {
        "problems": collect_section("Problems", "Any problems?", PROBLEM_ITEMS),
        "exceptional": collect_section("Exceptional", "Anything done exceptionally well?", EXCEPTIONAL_ITEMS),
    }
    print("\n== Overall description ==")
    labels["overall_description"] = ask_text("  describe the overall outcome of the session")

    rubric = {
        "session": {
            "id": info.get("id", session_id),
            "title": info.get("title", ""),
            "directory": info.get("directory") or info.get("path") or str(repo),
            "agent": info.get("agent", ""),
            "model": model,
            "version": info.get("version", ""),
        },
        "repo": str(repo),
        "artifacts": {
            "session_md": md_path.name,
            "session_json": json_path.name,
            "source_zip": ZIP_NAME if src_copied else None,
            "system_prompt": capture.get("system_prompt"),
            "tool_definitions": capture.get("tool_definitions"),
            "tool_outputs": tool_outputs or None,
            "licenses": "licenses" if license_info else None,
            "sanitize_report": "sanitize-report.json" if args.sanitize else None,
        },
        "labels": labels,
        "labeled_at": now_iso(),
    }

    (outdir / "rubric.json").write_text(json.dumps(rubric, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / "rubric.md").write_text(render_rubric_markdown(rubric), encoding="utf-8")
    print(f"\nsaved rubric.json and rubric.md to {outdir}")


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aidata.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=".", help="repo to operate on (default: current directory)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_start = sub.add_parser("start", help="build src.zip from .ai_data_config")
    p_start.add_argument("--output", default=ZIP_NAME, help=f"zip file to write (default: {ZIP_NAME})")
    p_start.set_defaults(func=cmd_start)

    p_end = sub.add_parser("end", help="export the session, copy artifacts and label them")
    p_end.add_argument("--session", help="opencode session id (default: most recent for this project)")
    p_end.add_argument("--no-sanitize", dest="sanitize", action="store_false", help="keep secrets/PII in the captured data (not recommended)")
    p_end.set_defaults(func=cmd_end, sanitize=True)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\naborted", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
