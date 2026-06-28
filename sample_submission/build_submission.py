from __future__ import annotations

import argparse
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT_DIR / "submission_manifest.txt"
DEFAULT_OUTPUT = ROOT_DIR / "submission.tar.gz"
SKIP_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
}
SKIP_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
}


@dataclass(frozen=True)
class ManifestPaths:
    includes: list[Path]
    excludes: list[Path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Kaggle submission.tar.gz from a manifest file."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the include-manifest file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to the output .tar.gz archive.",
    )
    return parser.parse_args()


def resolve_manifest_path(manifest_path: Path, raw_entry: str) -> Path:
    relative_path = Path(raw_entry.rstrip("/\\"))
    resolved_path = (manifest_path.parent / relative_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Manifest entry does not exist: {relative_path.as_posix()}"
        )
    return resolved_path


def parse_manifest(manifest_path: Path) -> ManifestPaths:
    includes: list[Path] = []
    excludes: list[Path] = []
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("!"):
            excludes.append(resolve_manifest_path(manifest_path, line[1:].strip()))
            continue
        includes.append(resolve_manifest_path(manifest_path, line))
    if not includes:
        raise ValueError("The manifest is empty.")
    return ManifestPaths(includes=includes, excludes=excludes)


def should_skip(path: Path) -> bool:
    if any(part in SKIP_DIR_NAMES for part in path.parts):
        return True
    if path.suffix.lower() in SKIP_FILE_SUFFIXES:
        return True
    return False


def is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


def is_excluded(path: Path, excluded_paths: list[Path]) -> bool:
    return any(
        path == excluded_path or is_relative_to(path, excluded_path)
        for excluded_path in excluded_paths
    )


def iter_files(entry_path: Path, excluded_paths: list[Path]) -> list[Path]:
    if entry_path.is_file():
        if should_skip(entry_path) or is_excluded(entry_path, excluded_paths):
            return []
        return [entry_path]

    files: list[Path] = []
    for current_dir, dir_names, file_names in os.walk(entry_path):
        current_path = Path(current_dir)
        dir_names[:] = sorted(
            dir_name
            for dir_name in dir_names
            if dir_name not in SKIP_DIR_NAMES
            and not is_excluded(current_path / dir_name, excluded_paths)
        )
        for file_name in sorted(file_names):
            file_path = Path(current_dir) / file_name
            if should_skip(file_path) or is_excluded(file_path, excluded_paths):
                continue
            files.append(file_path)
    return files


def to_archive_name(path: Path) -> str:
    return path.relative_to(ROOT_DIR).as_posix()


def build_archive(manifest_path: Path, output_path: Path) -> list[str]:
    manifest_paths = parse_manifest(manifest_path.resolve())
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    archived_names: list[str] = []
    seen_names: set[str] = set()

    with tarfile.open(output_path, mode="w:gz") as archive:
        for entry_path in manifest_paths.includes:
            for file_path in iter_files(entry_path, manifest_paths.excludes):
                archive_name = to_archive_name(file_path)
                if archive_name in seen_names:
                    continue
                archive.add(file_path, arcname=archive_name)
                seen_names.add(archive_name)
                archived_names.append(archive_name)

    return archived_names


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    output_path = args.output.resolve()

    archived_names = build_archive(manifest_path, output_path)

    print(f"Created: {output_path}")
    print("Archived files:")
    for archive_name in archived_names:
        print(f" - {archive_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
