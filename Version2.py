#!/usr/bin/env python3
"""
cpp_inspect.py

Text-based C++ repo inspector:
  1) Build file include dependency graph and print include trees.
  2) Build class inheritance graph (heuristic regex) and print base-class trees.

Examples:
  python cpp_inspect.py deps --root /path/to/repo --start src/foo.cpp
  python cpp_inspect.py deps --root . --start include/foo.h --max-depth 8

  python cpp_inspect.py inherits --root . --class MyNamespace::Foo
  python cpp_inspect.py inherits --root . --class Foo --prefer-namespace MyNamespace

Tips:
  - Add include dirs if needed: --inc include --inc src --inc third_party/somelib/include
"""

from __future__ import annotations
import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# ---------- Parsing includes ----------

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"])\s*([^">]+?)\s*[>"]')
# Very light stripping of comments (good enough for includes in most codebases)
LINE_COMMENT_RE = re.compile(r'//.*$')

CPP_EXTS = {".h", ".hpp", ".hh", ".hxx", ".inl", ".ipp", ".c", ".cc", ".cpp", ".cxx"}


@dataclass(frozen=True)
class IncludeRef:
    raw: str           # the string inside include, e.g. "foo/bar.h"
    kind: str          # "quote" or "angle"
    resolved: Optional[Path]  # resolved path within repo, if found


def strip_line_comment(s: str) -> str:
    return LINE_COMMENT_RE.sub("", s)


def find_all_source_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in CPP_EXTS:
            files.append(p)
    return files


def parse_includes(file_path: Path) -> List[Tuple[str, str]]:
    """
    Returns list of (kind, include_string) where kind in {"quote","angle"}.
    """
    includes: List[Tuple[str, str]] = []
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = strip_line_comment(line)
                m = INCLUDE_RE.match(line)
                if not m:
                    continue
                delim, inc = m.group(1), m.group(2).strip()
                kind = "quote" if delim == '"' else "angle"
                includes.append((kind, inc))
    except OSError:
        pass
    return includes


# ---------- Resolving include paths ----------

def build_resolution_index(root: Path, all_files: List[Path]) -> Dict[str, List[Path]]:
    """
    Maps "path as posix string relative to root" and also basenames to candidate files.
    """
    by_rel: Dict[str, List[Path]] = {}
    by_base: Dict[str, List[Path]] = {}
    for p in all_files:
        rel = p.relative_to(root).as_posix()
        by_rel.setdefault(rel, []).append(p)
        by_base.setdefault(p.name, []).append(p)
    # store both kinds in one dict with different keys
    index: Dict[str, List[Path]] = {}
    for k, v in by_rel.items():
        index[f"REL::{k}"] = v
    for k, v in by_base.items():
        index[f"BASE::{k}"] = v
    return index


def resolve_include(
    *,
    including_file: Path,
    include_str: str,
    root: Path,
    index: Dict[str, List[Path]],
    include_dirs: List[Path],
) -> Optional[Path]:
    """
    Heuristic resolution:
      1) relative to including file directory
      2) relative to each include_dir (relative to root)
      3) direct rel match from repo root
      4) basename match if unique
    """
    # 1) relative to current file dir
    cand = (including_file.parent / include_str).resolve()
    try:
        cand_rel = cand.relative_to(root.resolve())
        if cand.exists() and cand.is_file():
            return root.resolve() / cand_rel
    except Exception:
        pass

    # 2) relative to include dirs
    for incdir in include_dirs:
        cand2 = (root / incdir / include_str).resolve()
        if cand2.exists() and cand2.is_file():
            return cand2

    # 3) direct repo-relative path string match
    rel_key = f"REL::{include_str.replace(os.sep, '/')}"
    if rel_key in index and len(index[rel_key]) == 1:
        return index[rel_key][0]
    if rel_key in index and len(index[rel_key]) > 1:
        # ambiguous (same rel appears multiple times) - pick first deterministically
        return sorted(index[rel_key])[0]

    # 4) basename match if unique
    base = Path(include_str).name
    base_key = f"BASE::{base}"
    if base_key in index and len(index[base_key]) == 1:
        return index[base_key][0]

    return None


def build_include_graph(root: Path, include_dirs: List[Path]) -> Tuple[Dict[Path, List[IncludeRef]], List[Path]]:
    all_files = find_all_source_files(root)
    index = build_resolution_index(root, all_files)

    graph: Dict[Path, List[IncludeRef]] = {}
    for f in all_files:
        refs: List[IncludeRef] = []
        for kind, inc in parse_includes(f):
            resolved = resolve_include(
                including_file=f,
                include_str=inc,
                root=root,
                index=index,
                include_dirs=include_dirs,
            )
            refs.append(IncludeRef(raw=inc, kind=kind, resolved=resolved))
        graph[f] = refs

    return graph, all_files


# ---------- Printing file dependency trees ----------

def print_dep_tree(
    root: Path,
    graph: Dict[Path, List[IncludeRef]],
    start: Path,
    max_depth: int,
    show_unresolved: bool,
    skip_system_angle: bool,
):
    start = (root / start).resolve() if not start.is_absolute() else start.resolve()

    if start not in graph:
        # try to locate by repo-relative string
        rel = start.relative_to(root.resolve()).as_posix() if start.is_absolute() else str(start)
        candidates = [p for p in graph.keys() if p.relative_to(root).as_posix() == rel]
        if candidates:
            start = candidates[0]

    if start not in graph:
        print(f"[!] Start file not found in indexed sources: {start}")
        return

    root_res = root.resolve()

    def label(p: Path) -> str:
        try:
            return p.relative_to(root_res).as_posix()
        except Exception:
            return str(p)

    seen_stack: Set[Path] = set()

    def rec(file: Path, depth: int, prefix: str, is_last: bool):
        branch = "└── " if is_last else "├── "
        print(prefix + branch + label(file))

        if depth >= max_depth:
            return

        new_prefix = prefix + ("    " if is_last else "│   ")

        refs = graph.get(file, [])
        # Filter system includes if requested (angle includes that don't resolve)
        children: List[Tuple[str, Optional[Path], IncludeRef]] = []
        for r in refs:
            if skip_system_angle and r.kind == "angle" and r.resolved is None:
                continue
            if r.resolved is not None:
                children.append((label(r.resolved), r.resolved, r))
            elif show_unresolved:
                # show unresolved as pseudo-nodes
                children.append((f"[unresolved] {r.raw}", None, r))

        # deterministic order
        children.sort(key=lambda t: t[0].lower())

        if file in seen_stack:
            # cycle
            if children:
                print(new_prefix + "└── " + "[cycle detected: stopping expansion]")
            return

        seen_stack.add(file)
        for i, (_, child_path, r) in enumerate(children):
            last = (i == len(children) - 1)
            if child_path is None:
                print(new_prefix + ("└── " if last else "├── ") + f"[unresolved] {r.raw}")
            else:
                rec(child_path, depth + 1, new_prefix, last)
        seen_stack.remove(file)

    # root header without branch
    print(label(start))
    refs0 = graph.get(start, [])
    # Kick recursion for children of root
    children0: List[Tuple[str, Optional[Path], IncludeRef]] = []
    for r in refs0:
        if skip_system_angle and r.kind == "angle" and r.resolved is None:
            continue
        if r.resolved is not None:
            children0.append((r.resolved.relative_to(root_res).as_posix(), r.resolved, r))
        elif show_unresolved:
            children0.append((f"[unresolved] {r.raw}", None, r))
    children0.sort(key=lambda t: t[0].lower())

    for i, (_, child_path, r) in enumerate(children0):
        last = (i == len(children0) - 1)
        if child_path is None:
            print(("└── " if last else "├── ") + f"[unresolved] {r.raw}")
        else:
            rec(child_path, 1, "", last)


# ---------- Parsing class inheritance (heuristic) ----------

# Capture namespace starts: namespace foo {  OR namespace foo::bar {
NS_START_RE = re.compile(r'^\s*namespace\s+([a-zA-Z_]\w*(?:::\w+)*)\s*\{')
# Detect class/struct definition lines (heuristic, allows templates above but not required)
# Excludes forward decl "class X;" by requiring "{" somewhere after optional inheritance.
CLASS_DEF_RE = re.compile(
    r'^\s*(class|struct)\s+([a-zA-Z_]\w*)\s*'
    r'(?:\:\s*([^\\{;]+))?\s*\{'
)
# Split base list by commas, then strip access specifiers / virtual / final keywords
BASE_SPLIT_RE = re.compile(r'\s*,\s*')

# crude brace tracking (won't handle braces in strings/comments perfectly, but ok for a first pass)
BRACE_OPEN_RE = re.compile(r'\{')
BRACE_CLOSE_RE = re.compile(r'\}')


def normalize_base_token(tok: str) -> str:
    """
    Turn 'public virtual Foo<int>' into 'Foo<int>' (keeps templates).
    """
    tok = tok.strip()
    tok = re.sub(r'\b(public|protected|private)\b', '', tok)
    tok = re.sub(r'\bvirtual\b', '', tok)
    tok = re.sub(r'\bfinal\b', '', tok)
    tok = re.sub(r'\boverride\b', '', tok)
    tok = re.sub(r'\s+', ' ', tok).strip()
    return tok


def parse_classes_in_file(file_path: Path) -> List[Tuple[List[str], str, List[str]]]:
    """
    Returns list of (namespace_stack, class_name, base_tokens)
    namespace_stack: list like ["MyNS", "Inner"]
    class_name: "Foo"
    base_tokens: ["Bar", "Baz<int>", "ns::Qux"]
    """
    results: List[Tuple[List[str], str, List[str]]] = []
    ns_stack: List[str] = []
    brace_depth = 0
    ns_depth_markers: List[Tuple[int, List[str]]] = []  # (brace_depth_at_entry, namespaces_added)

    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            for raw_line in f:
                line = strip_line_comment(raw_line).strip()
                if not line:
                    # still track braces below
                    pass

                # namespace start?
                mns = NS_START_RE.match(line)
                if mns:
                    ns_name = mns.group(1)
                    parts = ns_name.split("::")
                    ns_stack.extend(parts)
                    ns_depth_markers.append((brace_depth, parts))

                # class definition?
                mcl = CLASS_DEF_RE.match(line)
                if mcl:
                    class_name = mcl.group(2)
                    bases_raw = mcl.group(3)
                    bases: List[str] = []
                    if bases_raw:
                        for tok in BASE_SPLIT_RE.split(bases_raw.strip()):
                            bt = normalize_base_token(tok)
                            if bt:
                                bases.append(bt)
                    results.append((list(ns_stack), class_name, bases))

                # update brace depth after parsing constructs on this line
                opens = len(BRACE_OPEN_RE.findall(raw_line))
                closes = len(BRACE_CLOSE_RE.findall(raw_line))
                brace_depth += opens - closes

                # pop namespaces when leaving their brace scope
                while ns_depth_markers and brace_depth < ns_depth_markers[-1][0] + 1:
                    _, added = ns_depth_markers.pop()
                    # remove the namespaces we added
                    for _ in added:
                        if ns_stack:
                            ns_stack.pop()

    except OSError:
        pass

    return results


def build_class_graph(root: Path) -> Tuple[Dict[str, List[str]], Dict[str, Set[Path]]]:
    """
    class_graph: fully_qualified_class -> list of base class tokens (as written)
    decl_files: class -> set(files where declared)
    """
    all_files = find_all_source_files(root)
    class_graph: Dict[str, List[str]] = {}
    decl_files: Dict[str, Set[Path]] = {}

    for f in all_files:
        for ns_stack, cname, bases in parse_classes_in_file(f):
            fq = "::".join(ns_stack + [cname]) if ns_stack else cname
            # keep first seen, but merge bases if multiple partial definitions appear
            if fq not in class_graph:
                class_graph[fq] = bases
            else:
                # merge bases conservatively
                existing = set(class_graph[fq])
                for b in bases:
                    if b not in existing:
                        class_graph[fq].append(b)
                        existing.add(b)
            decl_files.setdefault(fq, set()).add(f)

    return class_graph, decl_files


def resolve_base_name(
    base_token: str,
    *,
    current_ns: str,
    known_classes: Set[str],
    prefer_namespace: Optional[str],
) -> Optional[str]:
    """
    Try to map a base token to a known fully qualified class name.

    Rules:
      - If token already contains '::', try exact match.
      - Else try current namespace + token, then prefer_namespace + token (if provided),
        then global token.
      - If still not found, return None.
    """
    # strip leading '::'
    bt = base_token.strip()
    if bt.startswith("::"):
        bt = bt[2:].strip()

    # remove trailing qualifiers after templates? (we keep templates, but graph keys won't include template args)
    # heuristic: drop template args for lookup
    bt_key = re.sub(r'<.*>$', '', bt).strip()

    if "::" in bt_key:
        return bt_key if bt_key in known_classes else None

    candidates = []
    if current_ns:
        candidates.append(f"{current_ns}::{bt_key}")
    if prefer_namespace:
        candidates.append(f"{prefer_namespace}::{bt_key}")
    candidates.append(bt_key)

    for c in candidates:
        if c in known_classes:
            return c
    return None


def print_inheritance_tree(
    class_graph: Dict[str, List[str]],
    decl_files: Dict[str, Set[Path]],
    root: Path,
    start_class: str,
    max_depth: int,
    prefer_namespace: Optional[str],
    show_decl_files: bool,
):
    known = set(class_graph.keys())

    # If user gave unqualified name, try prefer_namespace first
    start = start_class.strip()
    if start.startswith("::"):
        start = start[2:]

    if start not in known:
        if "::" not in start and prefer_namespace and f"{prefer_namespace}::{start}" in known:
            start = f"{prefer_namespace}::{start}"
        elif "::" not in start:
            # try unique match by suffix
            matches = [c for c in known if c.endswith(f"::{start}") or c == start]
            if len(matches) == 1:
                start = matches[0]

    if start not in known:
        print(f"[!] Class not found: {start_class}")
        if "::" not in start_class:
            # helpful nearby suggestions
            matches = sorted([c for c in known if c.endswith(f"::{start_class}") or start_class in c])[:30]
            if matches:
                print("    Similar classes (subset):")
                for m in matches:
                    print(f"      - {m}")
        return

    def rel_files(c: str) -> str:
        if not show_decl_files:
            return ""
        files = decl_files.get(c, set())
        if not files:
            return ""
        rels = sorted([p.relative_to(root).as_posix() for p in files])
        # show up to 3 to keep it readable
        shown = rels[:3]
        extra = f" (+{len(rels)-3} more)" if len(rels) > 3 else ""
        return "  [" + ", ".join(shown) + extra + "]"

    visited_stack: Set[str] = set()

    def rec(cls: str, depth: int, prefix: str, is_last: bool, current_ns: str):
        branch = "└── " if is_last else "├── "
        print(prefix + branch + cls + rel_files(cls))
        if depth >= max_depth:
            return

        bases = class_graph.get(cls, [])
        # resolve bases to known classes when possible; keep unresolved as text
        resolved: List[Tuple[str, Optional[str]]] = []
        for b in bases:
            rb = resolve_base_name(
                b,
                current_ns=current_ns,
                known_classes=known,
                prefer_namespace=prefer_namespace,
            )
            if rb is not None:
                resolved.append((rb, rb))
            else:
                resolved.append((f"[unresolved] {b}", None))

        resolved.sort(key=lambda t: t[0].lower())
        new_prefix = prefix + ("    " if is_last else "│   ")

        if cls in visited_stack:
            if resolved:
                print(new_prefix + "└── " + "[cycle detected: stopping expansion]")
            return

        visited_stack.add(cls)
        # compute namespace for children: namespace of cls
        next_ns = "::".join(cls.split("::")[:-1])
        for i, (label, rcls) in enumerate(resolved):
            last = i == len(resolved) - 1
            if rcls is None:
                print(new_prefix + ("└── " if last else "├── ") + label)
            else:
                rec(rcls, depth + 1, new_prefix, last, next_ns)
        visited_stack.remove(cls)

    print(start + rel_files(start))
    ns0 = "::".join(start.split("::")[:-1])
    bases0 = class_graph.get(start, [])
    resolved0: List[Tuple[str, Optional[str]]] = []
    for b in bases0:
        rb = resolve_base_name(
            b,
            current_ns=ns0,
            known_classes=known,
            prefer_namespace=prefer_namespace,
        )
        if rb is not None:
            resolved0.append((rb, rb))
        else:
            resolved0.append((f"[unresolved] {b}", None))
    resolved0.sort(key=lambda t: t[0].lower())

    for i, (label, rcls) in enumerate(resolved0):
        last = i == len(resolved0) - 1
        if rcls is None:
            print(("└── " if last else "├── ") + label)
        else:
            rec(rcls, 1, "", last, ns0)


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Text-based C++ dependency + inheritance inspector")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_deps = sub.add_parser("deps", help="Print include dependency tree from a start file")
    ap_deps.add_argument("--root", required=True, help="Repo root")
    ap_deps.add_argument("--start", required=True, help="Start file path (relative to root or absolute)")
    ap_deps.add_argument("--inc", action="append", default=[], help="Extra include dir (relative to root). Repeatable.")
    ap_deps.add_argument("--max-depth", type=int, default=12)
    ap_deps.add_argument("--show-unresolved", action="store_true", help="Show unresolved #includes in the tree")
    ap_deps.add_argument("--no-skip-system", action="store_true",
                         help="Do NOT skip system angle includes that don't resolve into repo")

    ap_inh = sub.add_parser("inherits", help="Print inheritance tree (base classes) from a start class")
    ap_inh.add_argument("--root", required=True, help="Repo root")
    ap_inh.add_argument("--class", dest="cls", required=True, help="Start class, e.g. Foo or ns::Foo")
    ap_inh.add_argument("--max-depth", type=int, default=12)
    ap_inh.add_argument("--prefer-namespace", default=None,
                        help="If class/base is unqualified, prefer this namespace for resolution (e.g. MyNS)")
    ap_inh.add_argument("--show-decl-files", action="store_true", help="Show up to 3 files declaring each class")

    args = ap.parse_args()
    root = Path(args.root).resolve()

    if args.cmd == "deps":
        include_dirs = [Path(p) for p in args.inc]
        graph, _all = build_include_graph(root, include_dirs)
        print_dep_tree(
            root=root,
            graph=graph,
            start=Path(args.start),
            max_depth=args.max_depth,
            show_unresolved=args.show_unresolved,
            skip_system_angle=not args.no_skip_system,
        )

    elif args.cmd == "inherits":
        class_graph, decl_files = build_class_graph(root)
        print_inheritance_tree(
            class_graph=class_graph,
            decl_files=decl_files,
            root=root,
            start_class=args.cls,
            max_depth=args.max_depth,
            prefer_namespace=args.prefer_namespace,
            show_decl_files=args.show_decl_files,
        )


if __name__ == "__main__":
    main()
