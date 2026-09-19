from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MENAGERIE_DIR = ROOT / "third_party" / "mujoco_menagerie"
MODEL_DIR = MENAGERIE_DIR / "franka_emika_panda"
UPSTREAM = "https://github.com/google-deepmind/mujoco_menagerie.git"
WRIST_CAMERA = (
    '                      <camera name="wrist_rgbd" pos="0.055 0 0.045" '
    'quat="0 1 0 0" fovy="75"/>\n'
)


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
    panda_xml = MODEL_DIR / "panda.xml"
    panda_text = panda_xml.read_text(encoding="utf-8")
    if 'camera name="wrist_rgbd"' not in panda_text:
        marker = '                      <site name="grasp_site" pos="0 0 0.103" size="0.005" rgba="1 0.8 0.1 0.6"/>\n'
        if marker not in panda_text:
            raise RuntimeError("Could not locate the Panda grasp_site for wrist-camera installation")
        panda_xml.write_text(panda_text.replace(marker, marker + WRIST_CAMERA), encoding="utf-8")
    shutil.copy2(ROOT / "assets" / "pick_place.xml", MODEL_DIR / "pick_place.xml")
    print(f"Panda assets ready: {MODEL_DIR}")


if __name__ == "__main__":
    main()

