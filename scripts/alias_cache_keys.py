"""Alias a cached artifact under the key a config computes with the current code.

Source fingerprints re-key the ``stats`` / ``rank_probe`` artifacts whenever ``core/importance.py`` /
``core/rank_probe.py`` change, even for math-preserving edits. ``ArtifactCache.load`` accepts a symlink
``<kind>/<new key>.pt -> <old key>.pt`` whose target carries the old key, so a 184 GB statistics file can be reused
without rewriting it. This script computes the new key for a config and creates that symlink.

Examples
--------
Alias the 14B 512-sample statistics for the production config, old key prefix taken from a previous run's
``[cache] hit stats 80831d36aeb6`` line::

    uv run python scripts/alias_cache_keys.py configs/qwen3_14b_best_stats512.json --kind stats --old 80831d36aeb6

Alias the same statistics into a separate cache directory (verification run)::

    uv run python scripts/alias_cache_keys.py configs/qwen3_14b_probe_verify_parallel4.json --kind stats \\
        --old 80831d36aeb6 --from-cache-dir ~/code/shaman/cache --cache-dir cache_verify
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nanoquant.main import build_quant_config, parse_arguments
from nanoquant.utils.cache import probe_key, stats_key

KEY_FUNCS = {"stats": stats_key, "rank_probe": probe_key}


def resolve_old(cache_dir: Path, kind: str, old: str) -> Path:
    """Return the existing artifact whose key starts with ``old`` (full key or unique prefix).

    Parameters
    ----------
    cache_dir : Path
        Cache root holding ``<kind>/<key>.pt`` files.
    kind : str
        Artifact kind (``stats`` or ``rank_probe``).
    old : str
        Full key or unique prefix.

    Returns
    -------
    Path
        The matching ``.pt`` path.
    """
    matches = sorted(p for p in (cache_dir / kind).glob(f"{old}*.pt") if not p.is_symlink())
    if len(matches) != 1:
        raise SystemExit(f"{len(matches)} artifacts match {kind}/{old}* in {cache_dir} (need exactly one)")
    return matches[0]


def main(argv: list[str] | None = None) -> None:
    """Create ``<cache_dir>/<kind>/<new key>.pt -> <old artifact>`` for the config's current key."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="pipeline config (.json)")
    ap.add_argument("--kind", choices=sorted(KEY_FUNCS), required=True)
    ap.add_argument("--old", required=True, help="old key or unique prefix (e.g. from a '[cache] hit' log line)")
    ap.add_argument("--cache-dir", default=None, help="where to create the alias (default: the config's cache_dir)")
    ap.add_argument("--from-cache-dir", default=None, help="where the old artifact lives (default: --cache-dir)")
    ap.add_argument("--dry-run", action="store_true", help="print the keys without creating the symlink")
    args = ap.parse_args(argv)

    model_args, quant_args, tune_args, _ = parse_arguments([args.config])
    cfg = build_quant_config(model_args, quant_args, tune_args).to_dict()
    new_key = KEY_FUNCS[args.kind](cfg)
    cache_dir = Path(args.cache_dir or cfg["cache_dir"]).expanduser()
    from_dir = Path(args.from_cache_dir).expanduser() if args.from_cache_dir else cache_dir
    old_path = resolve_old(from_dir, args.kind, args.old)
    new_path = cache_dir / args.kind / f"{new_key}.pt"
    print(f"{args.kind}: old {old_path.stem[:12]} ({old_path}) -> new {new_key[:12]} ({new_path})")
    if old_path.stem == new_key:
        print("keys are identical; nothing to alias")
        return
    if args.dry_run:
        return
    if new_path.exists() or new_path.is_symlink():
        raise SystemExit(f"refusing to overwrite {new_path}")
    new_path.parent.mkdir(parents=True, exist_ok=True)
    target = old_path.name if old_path.parent.resolve() == new_path.parent.resolve() else old_path.resolve()
    os.symlink(target, new_path)
    print(f"created {new_path} -> {target}")


if __name__ == "__main__":
    main()
