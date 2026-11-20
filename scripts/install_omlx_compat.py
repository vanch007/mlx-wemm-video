"""Install a non-invasive .pth hook into oMLX's bundled Python environment
so oMLX automatically recognizes wemm_qwen3_5_moe checkpoints without modifying oMLX source files."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_OMLX_SITE = Path(
    "/Applications/oMLX.app/Contents/Resources/Python/framework-mlx-base/lib/python3.11/site-packages"
)
REPO_ROOT = Path(__file__).resolve().parent.parent


def install(site_dir: Path, dry_run: bool = False) -> None:
    if not site_dir.is_dir():
        raise FileNotFoundError(f"oMLX site-packages directory not found: {site_dir}")

    pth_file = site_dir / "mlx_wemm_video_omlx.pth"
    pth_content = (
        f"{REPO_ROOT}\n"
        "import omlx_compat; omlx_compat.apply_mlx_vlm_wemm_compat_patch()\n"
    )
    if dry_run:
        print(f"[DRY RUN] Would write {pth_file}:\n{pth_content}")
        return
    pth_file.write_text(pth_content, encoding="utf-8")
    print(f"Installed oMLX compatibility hook -> {pth_file}")


def uninstall(site_dir: Path) -> None:
    pth_file = site_dir / "mlx_wemm_video_omlx.pth"
    if pth_file.exists():
        pth_file.unlink()
        print(f"Removed {pth_file}")
    else:
        print(f"Nothing to remove at {pth_file}")


def main():
    parser = argparse.ArgumentParser(description="Install/uninstall oMLX compatibility hook for WeMM-Video.")
    parser.add_argument("--site-dir", type=Path, default=DEFAULT_OMLX_SITE)
    parser.add_argument("--uninstall", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.uninstall:
        uninstall(args.site_dir)
    else:
        install(args.site_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
