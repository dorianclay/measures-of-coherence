import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """The sha256 of a file, or of a directory as its files' relative paths and hashes, in sorted order."""
    path = Path(path)
    if path.is_file():
        return sha256(path)
    lines = [f"{f.relative_to(path).as_posix()} {sha256(f)}\n" for f in sorted(path.rglob("*")) if f.is_file()]
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def size(path: Path) -> int:
    path = Path(path)
    return path.stat().st_size if path.is_file() else sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
