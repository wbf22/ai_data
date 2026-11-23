#!/usr/bin/env python3
"""Capture an opencode session as AI training data.

Usage:
    python3 aidata.py start            # zip the repo per .ai_data_config -> src.zip
    python3 aidata.py end [--session ID] [--no-sanitize]
                                       # export the session, copy src.zip, label it, save labels
    python3 aidata.py rate <session-dir> --rater NAME [--all] [--force]
                                       # add a second/later rater's labels to a session
    python3 aidata.py agreement [--output PREFIX]
                                       # inter-rater agreement (Cohen's kappa)
    python3 aidata.py catalog [--output PREFIX]
                                       # rewrite the whole-dataset overview (rerunnable)

The "start" step reads .ai_data_config (gitignore-style) from the current repo
and writes src.zip. The "end" step uses `opencode export` (CLI, no TUI needed)
to grab the current project's most recent session, copies it and src.zip into
data/<session>/, sanitizes secrets/PII out of every captured artifact, collects
the repo's license files into data/<session>/licenses/, then runs an interactive
labeling step.

Sanitization is on by default and reports what it redacted (sanitize-report.*).
Pass --no-sanitize to keep the raw data.
"""

from __future__ import annotations

import argparse
import difflib
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
LABELS_JSON = "labels.json"
LABELS_MD = "labels.md"
LABELS_VERSION = 5
LEGACY_LABELS_JSON = "rubric.json"  # captures labeled before the rubric->labels rename
# where the .opencode/plugin/aidata-capture.ts plugin drops system prompt + tool defs
CAPTURE_DIR = Path(os.environ.get("AIDATA_CAPTURE_DIR", Path.home() / ".local" / "share" / "opencode" / "aidata-capture"))
# session categories and issue tags live with the tool, so the taxonomy is shared
# across every repo. Both are plain text, one label per line, edited by hand.
CATEGORIES_FILE = REPO_ROOT / "categories.txt"
ISSUES_FILE = REPO_ROOT / "issues.txt"

OUTCOME_CHOICES = [
    ("success", "succeeded - task fully accomplished"),
    ("partial", "partial - some of the task done, some not"),
    ("fail", "failed - task not accomplished"),
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


def warn(msg: str) -> None:
    print(_tint(f"warning: {msg}", "yellow", stream=sys.stderr), file=sys.stderr)


def note(msg: str) -> None:
    print(_tint(f"note: {msg}", "dim"))


def die(msg: str) -> None:
    print(_tint(f"error: {msg}", "red", stream=sys.stderr), file=sys.stderr)
    raise SystemExit(1)


def unique_dir(parent: Path, stem: str) -> Path:
    """Return a fresh `<parent>/<stem>` path, appending -2, -3, ... on collision.

    Each `end` run gets its own folder so one opencode session can yield several
    training examples.
    """
    candidate = parent / stem
    suffix = 2
    while candidate.exists():
        candidate = parent / f"{stem}-{suffix}"
        suffix += 1
    return candidate


# --------------------------------------------------------------------------- #
# start: build src.zip from .ai_data_config
# --------------------------------------------------------------------------- #
DEFAULT_CONFIG = """\
# .ai_data_config - what to include in src.zip (gitignore-style globs)
#
# `aidata start` zips every file matched here into src.zip; that snapshot is the
# "before" state the session diff is measured against. Patterns use gitignore
# syntax, matched against paths relative to the repo root:
#
#   - a pattern with no slash matches that name at any depth
#   - a plain line INCLUDES matching files; a leading '!' EXCLUDES them
#   - blank lines and lines starting with '#' are ignored
#
# Comments must be on their own lines (a trailing comment becomes part of the
# pattern). This default includes the whole repo; narrow it to keep snapshots
# small and to leave out secrets or noise.

*

# Exclude things you never want captured in the snapshot:
!data/
!.git/
!node_modules/
!__pycache__/
!*.zip
!*.log
"""


def ensure_config(repo: Path) -> Path:
    """Return the repo's .ai_data_config, writing a documented default if missing."""
    config = repo / CONFIG_NAME
    if not config.exists():
        config.write_text(DEFAULT_CONFIG, encoding="utf-8")
        warn(f"no {CONFIG_NAME} found in {repo}")
        print(f"         wrote a default that includes everything to {config}")
        print("         edit it to control what goes into src.zip, then re-run if you like.")
    return config


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


def write_source_zip(
    repo: Path, rules: list[tuple[bool, GitIgnoreSpecPattern]], out_path: Path
) -> list[str]:
    """Zip the repo per `rules` to `out_path`; returns the included relative paths."""
    # never include our own artifacts (the start snapshot or this output)
    exclude_files = {out_path.resolve(), (repo / ZIP_NAME).resolve()}
    exclude_dirs: set[Path] = set()
    if DATA_ROOT.resolve().is_relative_to(repo):
        exclude_dirs.add(DATA_ROOT.resolve())
    files = collect_files(repo, rules, exclude_files, exclude_dirs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in files:
            zf.write(repo / rel, arcname=rel)
    return files


LANGUAGE_BY_EXT = {
    ".py": "Python", ".js": "JavaScript", ".mjs": "JavaScript", ".cjs": "JavaScript",
    ".ts": "TypeScript", ".tsx": "TypeScript", ".jsx": "JavaScript",
    ".rs": "Rust", ".go": "Go", ".java": "Java", ".kt": "Kotlin", ".kts": "Kotlin",
    ".c": "C", ".h": "C", ".cc": "C++", ".cpp": "C++", ".hpp": "C++", ".cs": "C#",
    ".rb": "Ruby", ".php": "PHP", ".swift": "Swift", ".m": "Objective-C", ".scala": "Scala",
    ".sh": "Shell", ".bash": "Shell", ".zsh": "Shell", ".ps1": "PowerShell",
    ".html": "HTML", ".css": "CSS", ".scss": "CSS", ".vue": "Vue", ".svelte": "Svelte",
    ".sql": "SQL", ".r": "R", ".jl": "Julia", ".lua": "Lua", ".dart": "Dart",
    ".ex": "Elixir", ".exs": "Elixir", ".clj": "Clojure", ".hs": "Haskell",
    ".md": "Markdown", ".json": "JSON", ".yaml": "YAML", ".yml": "YAML",
    ".toml": "TOML", ".xml": "XML", ".tex": "TeX", ".ipynb": "Jupyter", ".txt": "Text",
}


def repo_stats(zip_path: Path) -> dict:
    """Summarize a source zip: file count, bytes, lines, and language mix."""
    files = 0
    total_bytes = 0
    lines = 0
    langs: Counter = Counter()
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.filename.endswith("/"):
                continue
            files += 1
            total_bytes += info.file_size
            ext = PurePosixPath(info.filename).suffix.lower()
            langs[LANGUAGE_BY_EXT.get(ext, "other")] += 1
            text = decode_text(zf.read(info.filename))
            if text is not None:
                lines += text.count("\n") + (0 if (not text or text.endswith("\n")) else 1)
    ordered = dict(langs.most_common())
    return {"files": files, "bytes": total_bytes, "lines": lines, "languages": ordered}


def cmd_start(args: argparse.Namespace) -> None:
    repo = Path(args.repo).expanduser().resolve()
    if not repo.is_dir():
        die(f"no such directory: {repo}")
    config = ensure_config(repo)

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
    # PII applies to the transcript *and* the captured source: code, license
    # files, package manifests, and fixtures routinely embed author emails,
    # home-directory paths, and contact numbers, so the src.zip pass needs them
    # too. Patterns are kept conservative to avoid mangling ordinary code.
    {"name": "email", "pattern": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "scopes": BOTH, "pii": True},
    {"name": "home_path", "pattern": re.compile(r"/(?:home|Users)/[^/\s\"'\\]+"), "scopes": BOTH, "pii": True},
    {"name": "windows_home_path", "pattern": re.compile(r"(?i)\b[A-Z]:\\Users\\[^\\\s\"']+"), "scopes": BOTH, "pii": True},
    {"name": "ssn", "pattern": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "scopes": BOTH, "pii": True},
    {"name": "phone_e164", "pattern": re.compile(r"\+\d{7,15}\b"), "scopes": BOTH, "pii": True},
]


def redact(text: str, scopes: frozenset[str], include_pii: bool = True) -> tuple[str, Counter]:
    """Apply every rule matching `scopes`, returning the new text and per-rule counts.

    Set `include_pii=False` to apply only secret rules -- used for license files,
    whose copyright notices (author emails) must be retained for attribution.
    """
    counts: Counter = Counter()
    for rule in REDACTION_RULES:
        if not (rule["scopes"] & scopes):
            continue
        if rule.get("pii") and not include_pii:
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
    """Rewrite a zip in place, redacting text entries. Returns per-entry counts.

    License files keep their PII (copyright notices and author emails must be
    retained for attribution); they still get secret redaction.
    """
    per_entry: dict[str, Counter] = {}
    tmp = zip_path.with_name(zip_path.name + ".tmp")
    with zipfile.ZipFile(zip_path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            raw = zin.read(info.filename)
            text = decode_text(raw)
            if text is None:
                zout.writestr(info, raw)
                continue
            # is_license_filename is defined below; resolved at call time.
            keep_pii = is_license_filename(info.filename)
            new, counts = redact(text, scopes, include_pii=not keep_pii)
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


# --------------------------------------------------------------------------- #
# license risk scan: flag terms that may be incompatible with AI training data
# --------------------------------------------------------------------------- #
# Names (from signature detection or SPDX ids) that imply real obligations.
LICENSE_NAME_NOTES: dict[str, str] = {
    "AGPL-3.0": "network copyleft; strong obligations, frequently excluded from datasets",
    "GPL-3.0": "strong copyleft; derivative works must remain GPL",
    "GPL-2.0": "strong copyleft; derivative works must remain GPL",
    "LGPL-3.0": "weak copyleft; check how the library is linked/used",
    "MPL-2.0": "file-level copyleft; check redistribution terms",
}

# Keyword/prose patterns that suggest restrictive terms in license text.
LICENSE_TEXT_RULES: list[tuple[str, re.Pattern]] = [
    ("non-commercial restriction", re.compile(r"(?i)non[- ]?commercial")),
    ("no-derivatives restriction", re.compile(r"(?i)no[- ]?derivatives?")),
    ("share-alike obligation", re.compile(r"(?i)share[- ]?alike")),
    (
        "AI/ML/TDM restriction",
        re.compile(
            r"(?i)(?:no|not|prohibit|forbid|without|restrict)\w*[^.\n]{0,60}"
            r"(?:\bai\b|a\.i\.|machine[- ]?learning|\bml\b|model training|text[- ]?(?:and|&)[- ]?data[- ]?mining|\btdm\b)"
        ),
    ),
    (
        "research/academic/evaluation-only",
        re.compile(r"(?i)(?:research|academic|evaluation|personal|non-?profit)(?:\s+\w+){0,2}\s+only|only for (?:research|academic|evaluation|personal)"),
    ),
    ("redistribution prohibited", re.compile(r"(?i)\bno (?:re)?distribution\b|may not (?:be )?(?:re)?distribut")),
    ("copyleft", re.compile(r"(?i)\bcopyleft\b")),
]

UNKNOWN_LICENSE_NOTE = "unrecognized license text; review manually"


def scan_license_text(text: str) -> list[tuple[str, str]]:
    """Return (kind, matched snippet) for every restrictive term found in `text`."""
    found: list[tuple[str, str]] = []
    for kind, pattern in LICENSE_TEXT_RULES:
        match = pattern.search(text)
        if match:
            snippet = " ".join(match.group(0).split())
            found.append((kind, snippet[:80]))
    return found


def name_notes(names: list[str]) -> list[str]:
    """Return compatibility notes for detected license / SPDX names.

    Matching is boundary-aware so e.g. "GPL-3.0" does not also fire for
    "AGPL-3.0-only" or "LGPL-3.0".
    """
    notes: list[str] = []
    for key, note in LICENSE_NAME_NOTES.items():
        pattern = re.compile(rf"(?<![A-Za-z]){re.escape(key)}", re.IGNORECASE)
        if any(pattern.search(name) for name in names):
            notes.append(f"{key}: {note}")
    return notes


def scan_licenses(license_entries: list[dict], spdx_identifiers: list[dict]) -> list[dict]:
    """Combine name- and text-based checks into a flat, de-duplicated list of findings.

    `license_entries` items are {path, detected, text}.
    """
    findings: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    def add(path: str, kind: str, detail: str) -> None:
        key = (path, kind, detail)
        if key not in seen:
            seen.add(key)
            findings.append({"path": path, "kind": kind, "detail": detail})

    for entry in license_entries:
        path = entry["path"]
        names = entry.get("detected") or []
        for note in name_notes(names):
            add(path, "restrictive license", note)
        if not names:
            add(path, "unknown license", UNKNOWN_LICENSE_NOTE)
        for kind, snippet in scan_license_text(entry.get("text", "")):
            add(path, kind, snippet)

    for note in name_notes([e["license"] for e in spdx_identifiers]):
        add("SPDX identifiers", "restrictive license", note)
    return findings


def _tint(text: str, style: str, stream=None) -> str:
    """Wrap `text` in an ANSI style when the terminal supports color."""
    return f"{ANSI[style]}{text}{ANSI['reset']}" if color_enabled(stream=stream) else text


def print_license_scan(findings: list[dict]) -> None:
    """Green when clean, yellow for each potential AI-training incompatibility."""
    if not findings:
        print(_tint("license scan: clean - detected licenses look compatible with AI training data", "green"))
        return
    print(_tint(f"license scan: {len(findings)} potential incompatibility(ies) - review before selling", "yellow"))
    for item in findings:
        print(_tint(f"  ! {item['path']}: {item['kind']} - {item['detail']}", "yellow"))


def render_license_findings_markdown(findings: list[dict]) -> list[str]:
    lines = ["## Compatibility notes (AI training)", ""]
    if not findings:
        lines.append("No restrictive terms found; detected licenses look compatible.")
        lines.append("")
        return lines
    lines.append(f"{len(findings)} potential issue(s) found - review before selling:")
    lines.append("")
    for item in findings:
        lines.append(f"- `{item['path']}` - **{item['kind']}**: {item['detail']}")
    lines.append("")
    return lines


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
    lines.extend(render_license_findings_markdown(index.get("risk_findings", [])))
    return "\n".join(lines)


def collect_licenses(zip_path: Path, outdir: Path, session_id: str) -> dict | None:
    """Scan a source zip for license files and SPDX headers, copying them into licenses/."""
    license_files: list[dict] = []
    spdx_identifiers: list[dict] = []
    scan_entries: list[dict] = []
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
                detected = detect_license_text(text)
                license_files.append({"path": name, "detected": detected})
                scan_entries.append({"path": name, "detected": detected, "text": text})
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

    risk_findings = scan_licenses(scan_entries, spdx_identifiers)

    index = {
        "session_id": session_id,
        "generated_at": now_iso(),
        "license_files": license_files,
        "spdx_identifiers": spdx_identifiers,
        "summary": dict(summary),
        "risk_findings": risk_findings,
    }
    licdir = outdir / "licenses"
    (licdir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    (licdir / "LICENSES.md").write_text(render_licenses_markdown(index), encoding="utf-8")
    return index


# --------------------------------------------------------------------------- #
# training release: an explicit grant so every capture states its AI-use terms
# --------------------------------------------------------------------------- #
TRAINING_RELEASE_NAME = "AI-TRAINING-RELEASE.md"
TRAINING_RELEASE_OVERRIDE = ".ai_data_release.md"
DEFAULT_GRANTOR = "William B Fowler"

DEFAULT_TRAINING_RELEASE = """\
# AI Training Release

The copyright holder releases the source code and development-session artifacts
in this folder for use as machine-learning **training, fine-tuning, and
evaluation data**, including for training models that may compete with any model
used to produce the session.

This grant covers the accompanying `src.zip` and the captured session
transcript, system prompt, tool definitions, and tool outputs.

- **Session:** {session_id}
- **Granted by:** {grantor}
- **Generated:** {generated_at}

Third-party licenses detected in `LICENSES.md` (if any) still apply to their
respective files and are not superseded by this release.
"""


def training_release_text(repo: Path, session_id: str, grantor: str) -> str:
    """Use the repo's `.ai_data_release.md` if present, else the default grant."""
    override = repo / TRAINING_RELEASE_OVERRIDE
    if override.is_file():
        return override.read_text(encoding="utf-8")
    return DEFAULT_TRAINING_RELEASE.format(
        session_id=session_id, grantor=grantor, generated_at=now_iso()
    )


def write_training_release(repo: Path, outdir: Path, session_id: str, grantor: str) -> str:
    licdir = outdir / "licenses"
    licdir.mkdir(parents=True, exist_ok=True)
    (licdir / TRAINING_RELEASE_NAME).write_text(
        training_release_text(repo, session_id, grantor), encoding="utf-8"
    )
    return f"licenses/{TRAINING_RELEASE_NAME}"


# --------------------------------------------------------------------------- #
# third-party notices: attribution for opencode (whose prompt/tools we capture)
# --------------------------------------------------------------------------- #
THIRD_PARTY_NOTICES_NAME = "THIRD-PARTY-NOTICES.md"

OPENCODE_LICENSE_TEXT = """\
MIT License

Copyright (c) 2025 opencode

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

THIRD_PARTY_NOTICES_TEMPLATE = """\
# Third-party notices

This capture was produced with **opencode**, the open-source coding agent
(<https://github.com/anomalyco/opencode>). The captured system prompt and tool
definitions originate from opencode and are redistributed here under the MIT
License below. Any redistribution of those artifacts must retain this notice.

"opencode" is a trademark of its owners. This dataset is neither affiliated
with nor endorsed by the opencode project.

## opencode

{opencode_license}
{deepseek_section}"""

DEEPSEEK_PROVENANCE_TEMPLATE = """
## DeepSeek

Session outputs were generated by a DeepSeek model (`{model}`) through the
DeepSeek Open Platform. Under section 4.2 of the DeepSeek Open Platform Terms
of Service, DeepSeek assigns any rights in the Outputs to the user and expressly
permits applying them to train other models (including model distillation). This
dataset relies on that grant.

- Source: DeepSeek Open Platform Terms of Service, section 4.2
  <https://cdn.deepseek.com/policies/en-US/deepseek-open-platform-terms-of-service.html>

"DeepSeek" is a trademark of Hangzhou DeepSeek Artificial Intelligence Co.,
Ltd. This dataset is neither affiliated with nor endorsed by DeepSeek.

Note: that grant is governed by the laws of the People's Republic of China and
the terms may be revised over time.
"""


def deepseek_provenance(model: str) -> str:
    """DeepSeek notice, but only when the session actually used a DeepSeek model."""
    if "deepseek" not in model.lower():
        return ""
    return DEEPSEEK_PROVENANCE_TEMPLATE.format(model=model or "DeepSeek")


def third_party_notices_text(model: str = "") -> str:
    return THIRD_PARTY_NOTICES_TEMPLATE.format(
        opencode_license=OPENCODE_LICENSE_TEXT,
        deepseek_section=deepseek_provenance(model),
    )


def write_third_party_notices(outdir: Path, model: str = "") -> str:
    licdir = outdir / "licenses"
    licdir.mkdir(parents=True, exist_ok=True)
    (licdir / THIRD_PARTY_NOTICES_NAME).write_text(third_party_notices_text(model), encoding="utf-8")
    return f"licenses/{THIRD_PARTY_NOTICES_NAME}"


def session_model(data: dict) -> str:
    """First `<provider>/<modelID>` seen in an exported session, else ''."""
    for message in data.get("messages", []) or []:
        mi = message.get("info", {}) or {}
        if mi.get("modelID"):
            return f"{mi.get('providerID', '')}/{mi.get('modelID')}"
    return ""


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
    if kind == "patch":
        files = part.get("files") or []
        h = (part.get("hash") or "")[:12]
        header = f"**Patch** `{h}`" if h else "**Patch**"
        body = "\n".join(f"- `{f}`" for f in files) or "_(no files listed)_"
        return f"{header} - {len(files)} file(s) changed:\n\n{body}"
    if kind == "snapshot":
        snap = (part.get("snapshot") or "")[:12]
        return f"*Snapshot:* `{snap}`" if snap else ""
    if kind == "step-finish":
        tokens = part.get("tokens") or {}
        cost = part.get("cost")
        bits: list[str] = []
        if tokens:
            bits.append(
                f"tokens in={tokens.get('input', 0)} out={tokens.get('output', 0)} "
                f"reasoning={tokens.get('reasoning', 0)}"
            )
            cache = tokens.get("cache") or {}
            if cache:
                bits.append(f"cache read={cache.get('read', 0)} write={cache.get('write', 0)}")
        if cost is not None:
            bits.append(f"cost=${cost}")
        return f"_step finished: {', '.join(bits)}_" if bits else ""
    if kind == "step-start":
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


def render_final_state_markdown(final_state: dict, outdir: Path) -> str:
    """Append the end-of-session state: the start-vs-end snapshot diff."""
    if not final_state:
        return ""
    lines = ["", "---", "", "# Final repository state", ""]

    if final_state.get("session_diff"):
        diff_path = outdir / final_state["session_diff"]
        if diff_path.is_file():
            diff_text = diff_path.read_text(encoding="utf-8").strip()
            lines.append("## Session changes (start -> end)")
            lines.append("")
            lines.append(fence(diff_text) if diff_text else "_(no changes)_")
            lines.append("")
    elif final_state.get("state_zip"):
        lines.append("Full start/end snapshots are in `src.zip` and `final-state/state.zip`.")
        lines.append("")
    return "\n".join(lines)


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


def ask_choice(question: str, items: list[tuple[str, str]]) -> tuple[str, str]:
    print(question)
    for i, (_, label) in enumerate(items, 1):
        print(f"  {i}. {label}")
    while True:
        raw = input(f"  select [1-{len(items)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(items):
            return items[int(raw) - 1]
        print("  invalid input")


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-")
    return slug or "uncategorized"


def _load_label_lines(path: Path, fallback: list[str]) -> list[str]:
    """Read one-label-per-line text; ignore blanks and '#' comments."""
    labels: list[str] = []
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                labels.append(line)
    return labels or fallback


def _append_label(path: Path, label: str, header: str, existing: list[str]) -> bool:
    """Append a label if it isn't already there (case-insensitive). Never raises.

    An unwritable install dir (read-only, permissions) just means the label isn't
    remembered.
    """
    label = " ".join(label.split())
    if not label or any(c.lower() == label.lower() for c in existing):
        return False
    try:
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            prefix = "" if (not text or text.endswith("\n")) else "\n"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(f"{prefix}{label}\n")
        else:
            path.write_text(header + label + "\n", encoding="utf-8")
        return True
    except OSError as exc:  # noqa: BLE001
        warn(f"could not update {path.name}: {exc}")
        return False


def load_categories() -> list[str]:
    """Read categories.txt (one label per line); fall back to a minimal default."""
    return _load_label_lines(CATEGORIES_FILE, ["Coding", "Non-coding"])


def add_category(label: str) -> bool:
    return _append_label(
        CATEGORIES_FILE, label, "# Session categories for aidata. One per line.\n", load_categories()
    )


def load_issues() -> list[str]:
    """Read issues.txt (one tag per line). Empty is fine; issues are optional."""
    return _load_label_lines(ISSUES_FILE, [])


def add_issue(label: str) -> bool:
    return _append_label(
        ISSUES_FILE, label, "# Session issue tags for aidata. One per line.\n", load_issues()
    )


def collect_category() -> dict:
    print("\n== Category ==")
    categories = load_categories()
    items = [(slugify(c), c) for c in categories] + [("__other__", "other (describe)")]
    key, label = ask_choice("What best describes this session?", items)
    if key == "__other__":
        custom = ask_text("  describe the category")
        if add_category(custom):
            print(f"  added '{custom}' to {CATEGORIES_FILE.name} (now offered in future sessions)")
        return {"value": slugify(custom), "label": custom, "custom": True}
    return {"value": key, "label": label, "custom": False}


def collect_issues() -> list[dict]:
    """Tag every issue that applies. Tags come from issues.txt (like categories)."""
    print("\n== Issues ==")
    print("  select every option that applies (blank = none)")
    issues = load_issues()
    items = [(slugify(t), t) for t in issues] + [("__other__", "other (describe)")]
    selected = ask_multiselect(items)
    labels_by_key = dict(items)
    tags: list[dict] = []
    for key in selected:
        if key == "__other__":
            custom = ask_text("  describe the issue")
            if add_issue(custom):
                print(f"  added '{custom}' to {ISSUES_FILE.name} (now offered in future sessions)")
            tags.append({"value": slugify(custom), "label": custom, "custom": True})
        else:
            tags.append({"value": key, "label": labels_by_key[key], "custom": False})
    return tags


def collect_difficulty() -> dict:
    print("\n== Difficulty ==")
    while True:
        raw = input("  rate difficulty 1-10 (1 = trivial, 10 = very hard): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= 10:
            return {"value": int(raw)}
        print("  enter a number from 1 to 10")


def collect_outcome() -> dict:
    print("\n== Outcome ==")
    key, label = ask_choice("What was the outcome?", OUTCOME_CHOICES)
    print()
    evidence = ask_text("  provide proof (e.g. file, command, test name, tool call)")
    return {"value": key, "label": label, "evidence": evidence}


def issue_values(section) -> set[str]:
    """Issue tag values from either the new list form or a legacy structured dict."""
    if isinstance(section, list):
        return {t.get("value") for t in section if t.get("value")}
    return {k for k, v in (section or {}).get("items", {}).items() if v.get("marked") == "yes"}


def render_labels_markdown(labels_doc: dict) -> str:
    labels = labels_doc["labels"]
    lines = [f"# Labels - {labels_doc['session']['id']}", ""]
    lines.append(f"- **Directory:** {labels_doc['session'].get('directory', '')}")
    lines.append(f"- **Model:** {labels_doc['session'].get('model', '')}")
    lines.append(f"- **Labeled at:** {labels_doc.get('labeled_at', '')}")
    category = labels.get("category")
    if category:
        lines.append(f"- **Category:** {category.get('label', category.get('value', ''))}")
    difficulty = labels.get("difficulty")
    if difficulty:
        lines.append(f"- **Difficulty:** {difficulty.get('value')}/10")
    stats = labels_doc.get("source_stats")
    if stats:
        langs = ", ".join(list(stats.get("languages", {}))[:5])
        lines.append(f"- **Source:** {stats.get('files', 0)} files, {stats.get('bytes', 0):,} bytes, {stats.get('lines', 0):,} lines")
        if langs:
            lines.append(f"- **Languages:** {langs}")
    lines.append("")

    outcome = labels["outcome"]
    lines.append(f"## Outcome - {outcome['label']}")
    lines.append("")
    evidence = outcome.get("evidence")
    if evidence:
        lines.append(f"- **Evidence:** {evidence}")
    lines.append("")

    issues = labels.get("issues") or []
    if isinstance(issues, list):
        tags = [t.get("label", t.get("value", "")) for t in issues]
    else:  # legacy structured dict
        tags = [v.get("label", k) for k, v in issues.get("items", {}).items() if v.get("marked") == "yes"]
    lines.append("## Issues")
    lines.append("")
    if tags:
        lines.extend(f"- {t}" for t in tags)
    else:
        lines.append("_none_")
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
    params_file = CAPTURE_DIR / f"params-{session_id}.json"
    if params_file.is_file():
        shutil.copy2(params_file, outdir / "sampling-params.json")
        copied["sampling_params"] = "sampling-params.json"
    return copied


def _zip_entries(zip_path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(zip_path) as zf:
        return {i.filename: zf.read(i.filename) for i in zf.infolist() if not i.filename.endswith("/")}


def session_diff(start_zip: Path, end_zip: Path, max_diff_bytes: int = 2_000_000) -> str:
    """Unified diff between two source snapshots, independent of git.

    Text files get a real unified diff; binary or very large changes are noted
    by name only.
    """
    start = _zip_entries(start_zip)
    end = _zip_entries(end_zip)
    added = sorted(set(end) - set(start))
    removed = sorted(set(start) - set(end))
    changed = sorted(n for n in set(start) & set(end) if start[n] != end[n])

    header = [
        "# Session changes (start -> end)",
        "",
        f"- added: {len(added)}, removed: {len(removed)}, changed: {len(changed)}",
        "",
    ]
    if not (added or removed or changed):
        return "\n".join(header + ["_no changes_", ""])

    lines: list[str] = []
    for name in added:
        lines.append(f"+++ added: {name}")
    for name in removed:
        lines.append(f"--- removed: {name}")
    for name in changed:
        ta = decode_text(start[name])
        tb = decode_text(end[name])
        if ta is None or tb is None:
            lines.append(f"### binary changed: {name}")
            continue
        if len(start[name]) + len(end[name]) > max_diff_bytes:
            lines.append(f"### changed (too large to diff): {name}")
            continue
        lines.append("")
        lines.extend(
            difflib.unified_diff(
                ta.splitlines(),
                tb.splitlines(),
                fromfile=f"a/{name}",
                tofile=f"b/{name}",
                lineterm="",
            )
        )
    return "\n".join(header + lines) + "\n"


def capture_final_state(repo: Path, outdir: Path, start_zip: Path | None = None) -> dict:
    """Record the repo state at `end`: a final snapshot and a start-vs-end diff."""
    final = outdir / "final-state"
    final.mkdir(parents=True, exist_ok=True)
    result: dict = {"dir": "final-state"}

    config = repo / CONFIG_NAME
    if config.is_file():
        rules = load_rules(config)
        files = write_source_zip(repo, rules, final / "state.zip")
        result["state_zip"] = "final-state/state.zip"
        result["state_files"] = len(files)

    if start_zip and start_zip.is_file() and (final / "state.zip").is_file():
        (final / "session.diff.patch").write_text(session_diff(start_zip, final / "state.zip"), encoding="utf-8")
        result["session_diff"] = "final-state/session.diff.patch"
    return result


def cmd_end(args: argparse.Namespace) -> None:
    repo = Path(args.repo).expanduser().resolve()
    session_id = resolve_session(repo, args.session)

    print(f"exporting session {session_id} ...")
    data = export_session(repo, session_id)

    stem = unique_dir(DATA_ROOT, f"session-{session_id[:8]}").name
    outdir = DATA_ROOT / stem
    outdir.mkdir(parents=True)
    print(f"capturing into data/{stem}/")

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
        warn(f"{ZIP_NAME} not found in {repo}; run `aidata.py start` first")

    tool_outputs = copy_tool_outputs(data, outdir)
    if tool_outputs:
        print(f"copied {len(tool_outputs)} truncated tool output(s)")

    capture = copy_capture_artifacts(session_id, outdir)
    if capture:
        print("copied " + ", ".join(capture.values()))
    else:
        note("no system-prompt/tool-definitions found (is the aidata plugin installed and loaded?)")

    final_state = capture_final_state(repo, outdir, start_zip=outdir / ZIP_NAME if src_copied else None)
    if final_state.get("session_diff"):
        print("captured final state -> final-state/ (snapshot + session diff)")
    else:
        print("captured final state -> final-state/ (snapshot only; no start src.zip)")

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
        final_dir = outdir / "final-state"
        if final_dir.is_dir():
            diff_path = final_dir / "session.diff.patch"
            if diff_path.is_file():
                new, counts = redact(diff_path.read_text(encoding="utf-8"), SOURCE)
                diff_path.write_text(new, encoding="utf-8")
                record("final-state/session.diff.patch", counts)
            state_zip = final_dir / "state.zip"
            if state_zip.is_file():
                for name, entry_counts in sanitize_zip(state_zip, SOURCE).items():
                    record(f"final-state/state.zip:{name}", entry_counts)

    json_path = outdir / f"{stem}.json"
    md_path = outdir / f"{stem}.md"
    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(data) + render_final_state_markdown(final_state, outdir), encoding="utf-8")
    print(f"wrote {md_path.name} and {json_path.name}")

    license_info = None
    if src_copied:
        license_info = collect_licenses(outdir / ZIP_NAME, outdir, session_id)
        if license_info:
            print(f"collected {len(license_info['license_files'])} license file(s) -> licenses/")
        else:
            print("no license files or SPDX identifiers found in source")

    print_license_scan(license_info.get("risk_findings", []) if license_info else [])

    release_rel = write_training_release(repo, outdir, session_id, args.grantor)
    print(f"wrote {release_rel}")
    notices_rel = write_third_party_notices(outdir, session_model(data))
    print(f"wrote {notices_rel}")

    if args.sanitize:
        write_sanitize_report(outdir, sanitize_report)
        total = sum(sanitize_report["totals"].values())
        if total:
            print(f"sanitized {total} item(s) -> sanitize-report.md")
        else:
            print("sanitize: nothing sensitive found")

    info = data.get("info", {}) or {}
    model = session_model(data)

    labels = {
        "category": collect_category(),
        "difficulty": collect_difficulty(),
        "outcome": collect_outcome(),
        "issues": collect_issues(),
    }

    source_stats = None
    if src_copied:
        try:
            source_stats = repo_stats(outdir / ZIP_NAME)
        except Exception as exc:  # noqa: BLE001
            warn(f"could not compute source stats: {exc}")

    labels_doc = {
        "labels_version": LABELS_VERSION,
        "session": {
            "id": info.get("id", session_id),
            "title": info.get("title", ""),
            "directory": info.get("directory") or info.get("path") or str(repo),
            "agent": info.get("agent", ""),
            "model": model,
            "version": info.get("version", ""),
        },
        "repo": str(repo),
        "source_stats": source_stats,
        "final_state": final_state,
        "artifacts": {
            "session_md": md_path.name,
            "session_json": json_path.name,
            "source_zip": ZIP_NAME if src_copied else None,
            "system_prompt": capture.get("system_prompt"),
            "tool_definitions": capture.get("tool_definitions"),
            "sampling_params": capture.get("sampling_params"),
            "tool_outputs": tool_outputs or None,
            "licenses": "licenses" if license_info else None,
            "training_release": release_rel,
            "third_party_notices": notices_rel,
            "final_state": "final-state" if final_state else None,
            "sanitize_report": "sanitize-report.json" if args.sanitize else None,
        },
        "labels": labels,
        "labeled_at": now_iso(),
    }

    (outdir / LABELS_JSON).write_text(json.dumps(labels_doc, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir / LABELS_MD).write_text(render_labels_markdown(labels_doc), encoding="utf-8")
    print(f"\nsaved {LABELS_JSON} and {LABELS_MD} to {outdir}")


# --------------------------------------------------------------------------- #
# rate: add a second (or later) rater's labels to a captured session
# --------------------------------------------------------------------------- #
RATINGS_DIR = "ratings"
PRIMARY_RATER = "rater1"


def labels_path(sdir: Path) -> Path:
    """The labels file for a session folder, tolerating pre-rename captures."""
    new = sdir / LABELS_JSON
    return new if new.is_file() else sdir / LEGACY_LABELS_JSON


def session_dirs() -> list[Path]:
    if not DATA_ROOT.is_dir():
        return []
    return sorted(
        p for p in DATA_ROOT.iterdir()
        if p.is_dir() and ((p / LABELS_JSON).is_file() or (p / LEGACY_LABELS_JSON).is_file())
    )


def rating_view(labels: dict) -> dict:
    """Reduce a label set to what agreement needs: category, outcome, issue tags.

    `category` and `issues` are read leniently so captures from older versions
    still load.
    """
    return {
        "category": (labels.get("category") or {}).get("value"),
        "outcome": labels["outcome"]["value"],
        "issues": issue_values(labels.get("issues")),
    }


def load_session_ratings(sdir: Path) -> dict[str, dict]:
    """Return {rater_name: rating_view} for a session folder.

    The main labels file is the primary rating; extra raters live in
    `<session>/ratings/<name>.json` so one session can accumulate several.
    """
    ratings: dict[str, dict] = {}
    main_path = labels_path(sdir)
    if main_path.is_file():
        doc = json.loads(main_path.read_text(encoding="utf-8"))
        ratings[doc.get("rater") or PRIMARY_RATER] = rating_view(doc["labels"])
    rdir = sdir / RATINGS_DIR
    if rdir.is_dir():
        for path in sorted(rdir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            ratings[path.stem] = rating_view(data["labels"])
    return ratings


# --------------------------------------------------------------------------- #
# markdown rendering: colorize a transcript for the terminal
# --------------------------------------------------------------------------- #
ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "italic": "\033[3m",
    "underline": "\033[4m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "bright_cyan": "\033[96m",
}

INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
BOLD_RE = re.compile(r"\*\*([^*\n]+)\*\*")
ITALIC_RE = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\n]+)\)")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
LIST_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
QUOTE_RE = re.compile(r"^(\s*>+\s?)(.*)$")
RULE_RE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")


def color_enabled(force_off: bool = False, stream=None) -> bool:
    """Color only on a TTY, and respect the NO_COLOR convention."""
    if force_off or os.environ.get("NO_COLOR"):
        return False
    return (stream if stream is not None else sys.stdout).isatty()


def _inline(text: str) -> str:
    """Style inline spans without letting later patterns match inserted escapes.

    Replacements are stashed as NUL-delimited placeholders first (ANSI escapes
    contain `[`, which the link pattern would otherwise swallow), then expanded.
    """
    tokens: list[str] = []

    def stash(styled: str) -> str:
        tokens.append(styled)
        return f"\x00{len(tokens) - 1}\x00"

    text = INLINE_CODE_RE.sub(lambda m: stash(f"{ANSI['yellow']}{m.group(1)}{ANSI['reset']}"), text)
    text = BOLD_RE.sub(lambda m: stash(f"{ANSI['bold']}{m.group(1)}{ANSI['reset']}"), text)
    text = ITALIC_RE.sub(lambda m: stash(f"{ANSI['italic']}{m.group(1)}{ANSI['reset']}"), text)
    text = LINK_RE.sub(
        lambda m: stash(
            f"{ANSI['underline']}{ANSI['blue']}{m.group(1)}{ANSI['reset']}{ANSI['dim']} ({m.group(2)}){ANSI['reset']}"
        ),
        text,
    )
    expand = re.compile(r"\x00(\d+)\x00")
    previous = None
    while previous != text:
        previous = text
        text = expand.sub(lambda m: tokens[int(m.group(1))], text)
    return text


def colorize_markdown(text: str) -> str:
    """Return `text` with ANSI colors for headings, code, lists, quotes, and tables."""
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            out.append(f"{ANSI['dim']}{line}{ANSI['reset']}")
            in_fence = not in_fence
            continue
        if in_fence:
            out.append(f"{ANSI['green']}{line}{ANSI['reset']}")
            continue
        if RULE_RE.match(line):
            out.append(f"{ANSI['dim']}{line}{ANSI['reset']}")
            continue
        heading = HEADING_RE.match(line)
        if heading:
            level = len(heading.group(1))
            color = ANSI["bright_cyan"] if level == 1 else ANSI["cyan"] if level == 2 else ANSI["blue"]
            out.append(f"{ANSI['bold']}{color}{_inline(heading.group(2))}{ANSI['reset']}")
            continue
        quote = QUOTE_RE.match(line)
        if quote:
            out.append(f"{ANSI['dim']}{quote.group(1)}{ANSI['reset']}{ANSI['italic']}{_inline(quote.group(2))}{ANSI['reset']}")
            continue
        if line.lstrip().startswith("|"):
            out.append(f"{ANSI['dim']}{_inline(line)}{ANSI['reset']}")
            continue
        item = LIST_RE.match(line)
        if item:
            out.append(f"{item.group(1)}{ANSI['magenta']}{item.group(2)}{ANSI['reset']} {_inline(item.group(3))}")
            continue
        out.append(_inline(line))
    return "\n".join(out)


def print_markdown(text: str, color: bool = True) -> None:
    print(colorize_markdown(text) if color else text)


def cmd_rate(args: argparse.Namespace) -> None:
    if args.all:
        targets = session_dirs()
        if not targets:
            die(f"no captured sessions under {DATA_ROOT}")
    elif args.session_dir:
        targets = [Path(args.session_dir).expanduser().resolve()]
    else:
        die("pass a session folder or --all")

    for sdir in targets:
        main_path = labels_path(sdir)
        if not main_path.is_file():
            warn(f"skip {sdir}: no labels file")
            continue
        labels_doc = json.loads(main_path.read_text(encoding="utf-8"))
        out = sdir / RATINGS_DIR / f"{args.rater}.json"
        if out.exists() and not args.force:
            warn(f"skip {sdir}: {out.name} already exists (use --force to overwrite)")
            continue

        session = labels_doc.get("session", {})
        transcript = (labels_doc.get("artifacts") or {}).get("session_md")
        print(f"\n=== Rater '{args.rater}' on {sdir.name} ===")
        print(f"  session : {session.get('title', '')}")
        print(f"  model   : {session.get('model', '')}")
        if transcript and not args.no_transcript:
            tpath = sdir / transcript
            print(f"  review  : {tpath}")
            if tpath.is_file():
                print("\n" + "=" * 72)
                print_markdown(tpath.read_text(encoding="utf-8"), color=color_enabled(args.no_color))
                print("=" * 72)
            else:
                warn(f"transcript {tpath} not found")
        elif transcript:
            print(f"  review  : {sdir / transcript} (not printed; --no-transcript)")
        print("\nRead the transcript above, then label it (the primary labels are hidden to avoid bias).")

        labels = {
            "category": collect_category(),
            "difficulty": collect_difficulty(),
            "outcome": collect_outcome(),
            "issues": collect_issues(),
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "labels_version": LABELS_VERSION,
                    "rater": args.rater,
                    "session": session,
                    "labels": labels,
                    "labeled_at": now_iso(),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"  saved {out}")


# --------------------------------------------------------------------------- #
# agreement: inter-rater agreement (Cohen's kappa)
# --------------------------------------------------------------------------- #
def cohen_kappa(a: list, b: list) -> float | None:
    n = len(a)
    if n == 0:
        return None
    cats = set(a) | set(b)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def kappa_label(k: float | None) -> str:
    if k is None:
        return "n/a"
    if k < 0:
        return "poor"
    if k < 0.21:
        return "slight"
    if k < 0.41:
        return "fair"
    if k < 0.61:
        return "moderate"
    if k < 0.81:
        return "substantial"
    return "almost perfect"


def pair_agreement(sessions: list[dict], r1: str, r2: str) -> dict:
    """`sessions` is a list of {rater_name: rating_view}; compare two raters."""
    paired = [s for s in sessions if r1 in s and r2 in s]
    if not paired:
        return {"n": 0}

    a = [s[r1]["outcome"] for s in paired]
    b = [s[r2]["outcome"] for s in paired]
    outcome = {
        "n": len(paired),
        "agree": sum(x == y for x, y in zip(a, b)),
        "kappa": cohen_kappa(a, b),
    }

    cats = [(s[r1]["category"], s[r2]["category"]) for s in paired]
    cats = [(x, y) for x, y in cats if x is not None and y is not None]
    category = {
        "n": len(cats),
        "agree": sum(x == y for x, y in cats),
        "kappa": cohen_kappa([x for x, _ in cats], [y for _, y in cats]) if cats else None,
    }

    # Issues are open tags, so compare over the union of every tag either rater
    # used. Pooled binary kappa mirrors the old flag-based measure; the mean
    # Jaccard is a natural set-overlap companion.
    a_issues = [s[r1]["issues"] for s in paired]
    b_issues = [s[r2]["issues"] for s in paired]
    vocab = sorted({tag for st in a_issues + b_issues for tag in st})
    a_bits: list[int] = []
    b_bits: list[int] = []
    per_item: dict[str, dict] = {}
    for tag in vocab:
        ia = [1 if tag in st else 0 for st in a_issues]
        ib = [1 if tag in st else 0 for st in b_issues]
        a_bits.extend(ia)
        b_bits.extend(ib)
        per_item[tag] = {
            "agree": sum(x == y for x, y in zip(ia, ib)),
            "n": len(paired),
            "kappa": cohen_kappa(ia, ib),
        }
    overlaps = [len(x & y) / len(x | y) for x, y in zip(a_issues, b_issues) if (x | y)]
    issues = {
        "n": len(a_bits),
        "agree": sum(x == y for x, y in zip(a_bits, b_bits)),
        "kappa": cohen_kappa(a_bits, b_bits) if (any(a_bits) or any(b_bits)) else None,
        "jaccard": (sum(overlaps) / len(overlaps)) if overlaps else None,
        "per_item": per_item,
    }
    return {"n": len(paired), "outcome": outcome, "category": category, "issues": issues}


def fmt_pct(hit: int, total: int) -> str:
    return f"{100 * hit / total:.1f}%" if total else "n/a"


def fmt_kappa(k: float | None) -> str:
    return f"{k:.3f} ({kappa_label(k)})" if k is not None else "n/a"


def render_agreement_markdown(report: dict) -> str:
    lines = ["# Inter-rater agreement", ""]
    lines.append(f"- **Generated:** {report['generated_at']}")
    lines.append(f"- **Sessions scanned:** {report['sessions_scanned']}")
    lines.append(f"- **Raters:** {', '.join(report['raters']) or 'none'}")
    lines.append("")
    lines.append("> Cohen's kappa: >0.8 almost perfect, 0.6-0.8 substantial, 0.4-0.6 moderate, 0.2-0.4 fair, <0.2 slight.")
    lines.append("> Items that neither rater ever flagged still read as 1.000; judge overall label quality by the pooled kappa.")
    lines.append("")
    if not report["pairs"]:
        lines.append("No session has ratings from two or more raters yet.")
        lines.append("")
        lines.append("Add one with: `aidata rate <session-dir> --rater NAME`")
        lines.append("")
        return "\n".join(lines)

    for pair in report["pairs"]:
        res = pair["result"]
        lines.append(f"## {pair['rater_a']} vs {pair['rater_b']}")
        lines.append("")
        lines.append(f"Compared on {res['n']} session(s).")
        lines.append("")
        cat = res.get("category") or {"n": 0, "agree": 0, "kappa": None}
        if cat["n"]:
            lines.append("### Category")
            lines.append("")
            lines.append(f"- agreement: {fmt_pct(cat['agree'], cat['n'])} ({cat['agree']}/{cat['n']})")
            lines.append(f"- Cohen's kappa: {fmt_kappa(cat['kappa'])}")
            lines.append("")
        o = res["outcome"]
        lines.append("### Outcome (success / partial / fail)")
        lines.append("")
        lines.append(f"- agreement: {fmt_pct(o['agree'], o['n'])} ({o['agree']}/{o['n']})")
        lines.append(f"- Cohen's kappa: {fmt_kappa(o['kappa'])}")
        lines.append("")
        it = res["issues"]
        lines.append("### Issue tags (pooled over all tags either rater used)")
        lines.append("")
        lines.append(f"- agreement: {fmt_pct(it['agree'], it['n'])} ({it['agree']}/{it['n']})")
        lines.append(f"- Cohen's kappa: {fmt_kappa(it['kappa'])}")
        if it.get("jaccard") is not None:
            lines.append(f"- mean Jaccard: {it['jaccard']:.3f}")
        lines.append("")
        if it["per_item"]:
            lines.append("| tag | agreement | kappa |")
            lines.append("| --- | ---: | ---: |")
            for key, stat in it["per_item"].items():
                lines.append(f"| {key} | {fmt_pct(stat['agree'], stat['n'])} | {fmt_kappa(stat['kappa'])} |")
            lines.append("")
    return "\n".join(lines)


def cmd_agreement(args: argparse.Namespace) -> None:
    dirs = session_dirs()
    if not dirs:
        die(f"no captured sessions under {DATA_ROOT}")

    sessions: list[dict] = []
    all_raters: set[str] = set()
    for sdir in dirs:
        ratings = load_session_ratings(sdir)
        if ratings:
            sessions.append(ratings)
            all_raters.update(ratings)

    raters = sorted(all_raters)
    pairs = []
    for i in range(len(raters)):
        for j in range(i + 1, len(raters)):
            result = pair_agreement(sessions, raters[i], raters[j])
            if result["n"]:
                pairs.append({"rater_a": raters[i], "rater_b": raters[j], "result": result})

    report = {
        "generated_at": now_iso(),
        "sessions_scanned": len(dirs),
        "raters": raters,
        "pairs": pairs,
    }

    outbase = Path(args.output).expanduser().resolve() if args.output else DATA_ROOT / "agreement"
    outbase.parent.mkdir(parents=True, exist_ok=True)
    (outbase.with_suffix(".json")).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_agreement_markdown(report)
    (outbase.with_suffix(".md")).write_text(md, encoding="utf-8")
    print(md)
    print(f"\nwrote {outbase.with_suffix('.md')} and {outbase.with_suffix('.json')}")


# --------------------------------------------------------------------------- #
# catalog: rerunnable dataset overview across every captured session
# --------------------------------------------------------------------------- #
def catalog_rows() -> list[dict]:
    rows: list[dict] = []
    for sdir in session_dirs():
        labels_doc = json.loads(labels_path(sdir).read_text(encoding="utf-8"))
        labels = labels_doc.get("labels", {})
        session = labels_doc.get("session", {})
        stats = labels_doc.get("source_stats") or {}
        rdir = sdir / RATINGS_DIR
        ratings = sorted(p.stem for p in rdir.glob("*.json")) if rdir.is_dir() else []
        rows.append(
            {
                "dir": sdir.name,
                "id": session.get("id", ""),
                "title": session.get("title", ""),
                "category": (labels.get("category") or {}).get("value"),
                "difficulty": (labels.get("difficulty") or {}).get("value"),
                "outcome": (labels.get("outcome") or {}).get("value"),
                "model": session.get("model", ""),
                "files": stats.get("files"),
                "bytes": stats.get("bytes"),
                "lines": stats.get("lines"),
                "languages": list(stats.get("languages", {}).keys()),
                "issues": sorted(issue_values(labels.get("issues"))),
                "raters": 1 + len(ratings),
            }
        )
    return rows


def render_catalog_markdown(rows: list[dict]) -> str:
    lines = ["# Dataset catalog", ""]
    lines.append(f"- **Sessions:** {len(rows)}")
    lines.append(f"- **Generated:** {now_iso()}")
    if not rows:
        lines.append("")
        lines.append("No captured sessions yet.")
        return "\n".join(lines) + "\n"

    def tally(key: str) -> str:
        counts = Counter(r[key] for r in rows if r.get(key))
        return ", ".join(f"{k}: {v}" for k, v in counts.most_common()) or "n/a"

    def tally_all(key: str) -> str:
        counts = Counter(tag for r in rows for tag in (r.get(key) or []))
        return ", ".join(f"{k}: {v}" for k, v in counts.most_common()) or "n/a"

    lines.append(f"- **By category:** {tally('category')}")
    lines.append(f"- **By outcome:** {tally('outcome')}")
    lines.append(f"- **By model:** {tally('model')}")
    lines.append(f"- **By issue:** {tally_all('issues')}")
    lines.append(f"- **Total source:** {sum(r['bytes'] or 0 for r in rows):,} bytes, {sum(r['lines'] or 0 for r in rows):,} lines")
    diffs = [r["difficulty"] for r in rows if r.get("difficulty")]
    if diffs:
        lines.append(f"- **Difficulty:** avg {sum(diffs) / len(diffs):.1f}/10 (n={len(diffs)})")
    lines.append("")
    lines.append("| dir | title | category | diff | outcome | model | files | lines | languages | issues | raters |")
    lines.append("| --- | --- | --- | ---: | --- | --- | ---: | ---: | --- | ---: | ---: |")
    for r in sorted(rows, key=lambda x: x["dir"]):
        diff = r["difficulty"] if r["difficulty"] is not None else ""
        lines.append(
            f"| {r['dir']} | {r['title'][:40]} | {r['category'] or ''} | {diff} | {r['outcome'] or ''} | "
            f"{r['model'] or ''} | {r['files'] if r['files'] is not None else ''} | "
            f"{r['lines'] if r['lines'] is not None else ''} | {', '.join(r['languages'][:3])} | {', '.join(r['issues'])} | {r['raters']} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def cmd_catalog(args: argparse.Namespace) -> None:
    unpack_sessions(DATA_ROOT)
    rows = catalog_rows()
    outbase = Path(args.output).expanduser().resolve() if args.output else DATA_ROOT / "CATALOG"
    outbase.parent.mkdir(parents=True, exist_ok=True)
    (outbase.with_suffix(".json")).write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    md = render_catalog_markdown(rows)
    (outbase.with_suffix(".md")).write_text(md, encoding="utf-8")
    print(md)
    print(f"wrote {outbase.with_suffix('.md')} and {outbase.with_suffix('.json')}")


# --------------------------------------------------------------------------- #
# pack / unpack: archive captured sessions for cheap storage
# --------------------------------------------------------------------------- #
def _safe_extract(zip_path: Path, dest: Path) -> None:
    dest = dest.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest):
                die(f"refusing unsafe path in {zip_path}: {member.filename}")
        zf.extractall(dest)


def cmd_pack(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_dir():
        die(f"no such directory: {data_dir}")

    dirs = sorted(p for p in data_dir.glob("session-*") if p.is_dir())
    if not dirs:
        print(f"no session-* directories found in {data_dir}")
        return

    packed = 0
    for session_dir in dirs:
        name = session_dir.name
        zip_path = data_dir / f"{name}.zip"
        if zip_path.exists() and not args.force:
            warn(f"skip {name} (archive exists; use --force to overwrite)")
            continue
        print(f"packing {name} ...")
        zip_path.unlink(missing_ok=True)
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(session_dir.rglob("*")):
                if file.is_file():
                    zf.write(file, arcname=file.relative_to(data_dir).as_posix())
        bad = zipfile.ZipFile(zip_path).testzip()
        if bad:
            die(f"archive failed verification: {zip_path} ({bad})")
        if not args.keep:
            shutil.rmtree(session_dir)
        packed += 1
    print(f"done: {packed} archive(s) written to {data_dir}")


def unpack_sessions(data_dir: Path, force: bool = False) -> int:
    """Extract every session-*.zip in `data_dir` back to a folder.

    Each archive is removed once it has been safely unpacked. Returns the number
    of archives extracted.
    """
    if not data_dir.is_dir():
        return 0

    unpacked = 0
    for zip_path in sorted(data_dir.glob("session-*.zip")):
        name = zip_path.stem
        target = data_dir / name
        if target.exists() and not force:
            warn(f"skip {name} (directory exists; use --force to overwrite)")
            continue
        print(f"unpacking {name} ...")
        if target.exists():
            shutil.rmtree(target)
        _safe_extract(zip_path, data_dir)
        zip_path.unlink()
        unpacked += 1
    if unpacked:
        print(f"done: {unpacked} archive(s) extracted to {data_dir}")
    return unpacked


def cmd_unpack(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir).expanduser()
    if not data_dir.is_dir():
        die(f"no such directory: {data_dir}")
    if not any(data_dir.glob("session-*.zip")):
        print(f"no session-*.zip archives found in {data_dir}")
        return
    unpack_sessions(data_dir, force=args.force)


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
    p_end.add_argument("--grantor", default=DEFAULT_GRANTOR, help=f'name credited in licenses/AI-TRAINING-RELEASE.md (default: "{DEFAULT_GRANTOR}")')
    p_end.add_argument("--no-sanitize", dest="sanitize", action="store_false", help="keep secrets/PII in the captured data (not recommended)")
    p_end.set_defaults(func=cmd_end, sanitize=True)

    p_rate = sub.add_parser("rate", help="add a second (or later) rater's labels to a captured session")
    p_rate.add_argument("session_dir", nargs="?", help="a data/session-* folder (omit with --all)")
    p_rate.add_argument("--rater", default="rater2", help="rater name, e.g. alice or me-week2 (default: rater2)")
    p_rate.add_argument("--all", action="store_true", help="rate every captured session under data/")
    p_rate.add_argument("--force", action="store_true", help="overwrite an existing rating for this rater")
    p_rate.add_argument("--no-transcript", action="store_true", help="do not print the session transcript before labeling")
    p_rate.add_argument("--no-color", action="store_true", help="disable ANSI colors in the printed transcript")
    p_rate.set_defaults(func=cmd_rate)

    p_agree = sub.add_parser("agreement", help="compute inter-rater agreement (Cohen's kappa) across ratings")
    p_agree.add_argument("--output", help="report path prefix (default: data/agreement)")
    p_agree.set_defaults(func=cmd_agreement)

    p_catalog = sub.add_parser("catalog", help="write a rerunnable overview of every captured session")
    p_catalog.add_argument("--output", help="report path prefix (default: data/CATALOG)")
    p_catalog.set_defaults(func=cmd_catalog)

    p_pack = sub.add_parser("pack", help="archive each data/session-* folder into session-*.zip")
    p_pack.add_argument("-d", "--dir", dest="data_dir", default="data", help="directory holding session-* folders (default: data)")
    p_pack.add_argument("-f", "--force", action="store_true", help="overwrite existing archives")
    p_pack.add_argument("-k", "--keep", action="store_true", help="keep the session folder after packing")
    p_pack.set_defaults(func=cmd_pack)

    p_unpack = sub.add_parser("unpack", help="extract each data/session-*.zip back to a folder (then delete the archive)")
    p_unpack.add_argument("-d", "--dir", dest="data_dir", default="data", help="directory holding session-*.zip archives (default: data)")
    p_unpack.add_argument("-f", "--force", action="store_true", help="overwrite existing session-* folders")
    p_unpack.set_defaults(func=cmd_unpack)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print(_tint("\naborted", "red"), file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
