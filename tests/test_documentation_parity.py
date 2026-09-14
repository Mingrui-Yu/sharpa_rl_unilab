from __future__ import annotations

import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ENGLISH_DOCS = REPOSITORY_ROOT / "docs" / "en"
CHINESE_DOCS = REPOSITORY_ROOT / "docs" / "zh_CN"


def _relative_files(root: Path) -> set[str]:
    return {path.relative_to(root).as_posix() for path in root.rglob("*.md")}


def _fenced_blocks(text: str) -> list[str]:
    return re.findall(r"```.*?```", text, flags=re.DOTALL)


def _inline_code(text: str) -> list[str]:
    prose_without_fences = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    return re.findall(r"`([^`]+)`", prose_without_fences)


def _heading_levels(text: str) -> list[int]:
    return [len(match.group(1)) for match in re.finditer(r"(?m)^(#+)\s+", text)]


def _table_shapes(text: str) -> list[tuple[int, ...]]:
    shapes = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        columns = max(line.count("|") - line.count(r"\|") - 1, 0)
        shapes.append((columns,))
    return shapes


def _list_markers(text: str) -> list[tuple[str, str, str]]:
    markers = []
    for match in re.finditer(r"(?m)^(\s*)(?:([-*])|(\d+\.))(?=\s)", text):
        indent, bullet, ordered = match.groups()
        marker = "bullet" if bullet is not None else "ordered"
        markers.append((indent, marker, ordered or ""))
    return markers


def _link_targets(text: str) -> list[str]:
    return [
        target
        for target in re.findall(r"\[[^\]]+\]\(([^)#]+?)(?:#[^)]*)?\)", text)
        if not target.startswith(("http:", "https:", "mailto:"))
    ]


def test_english_and_chinese_docs_have_identical_file_sets() -> None:
    english = _relative_files(ENGLISH_DOCS)
    chinese = _relative_files(CHINESE_DOCS)

    assert english
    assert english == chinese


def test_english_and_chinese_docs_have_parallel_structure() -> None:
    for relative_path in sorted(_relative_files(ENGLISH_DOCS)):
        english = (ENGLISH_DOCS / relative_path).read_text(encoding="utf-8")
        chinese = (CHINESE_DOCS / relative_path).read_text(encoding="utf-8")

        assert _heading_levels(english) == _heading_levels(chinese), relative_path
        assert _fenced_blocks(english) == _fenced_blocks(chinese), relative_path
        assert _inline_code(english) == _inline_code(chinese), relative_path
        assert _table_shapes(english) == _table_shapes(chinese), relative_path
        assert _list_markers(english) == _list_markers(chinese), relative_path
        assert _link_targets(english) == _link_targets(chinese), relative_path


def test_documentation_links_resolve() -> None:
    markdown_roots = (
        REPOSITORY_ROOT / "README.md",
        REPOSITORY_ROOT / "README_zh.md",
        *sorted((REPOSITORY_ROOT / "docs").rglob("*.md")),
    )
    broken: list[tuple[Path, str]] = []

    for markdown_path in markdown_roots:
        text = markdown_path.read_text(encoding="utf-8")
        for target in _link_targets(text):
            candidate = (markdown_path.parent / target).resolve()
            if not candidate.exists():
                broken.append((markdown_path, target))

    assert not broken
