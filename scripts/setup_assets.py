from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MENAGERIE_DIR = ROOT / "third_party" / "mujoco_menagerie"
MODEL_DIR = MENAGERIE_DIR / "franka_emika_panda"
UPSTREAM = "https://github.com/google-deepmind/mujoco_menagerie.git"


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def main() -> None:
    if not (MENAGERIE_DIR / ".git").exists():
        MENAGERIE_DIR.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, 4):
            shutil.rmtree(MENAGERIE_DIR, ignore_errors=True)
            try:
                run(
                    "git",
                    "-c",
                    "http.version=HTTP/1.1",
                    "clone",
                    "--depth",
                    "1",
                    "--filter=blob:none",
                    "--sparse",
                    UPSTREAM,
                    str(MENAGERIE_DIR),
                )
                break
            except subprocess.CalledProcessError:
                if attempt == 3:
                    raise
                print(f"Clone attempt {attempt} failed; retrying...")
                time.sleep(2 * attempt)

    run("git", "-C", str(MENAGERIE_DIR), "sparse-checkout", "set", "franka_emika_panda")
    shutil.copy2(ROOT / "assets" / "pick_place.xml", MODEL_DIR / "pick_place.xml")
    print(f"Panda assets ready: {MODEL_DIR}")


if __name__ == "__main__":
    main()

