"""Windows launcher for cal_and_map/tello_map.py: map the room's AprilTags without flying.

Hold the drone and walk it around; the map is built from what its camera sees. Uses the
repo's own pieces (map_loop, TagDetector, MapBuilder, VideoReceiver, and TelloLink with its
read-only allowlist: takeoff/rc/land raise before anything is sent). Only the Linux-only
parts are replaced: nmcli/ip for joining the WiFi and checking the drone's identity.

    conda activate tello
    python windows\\tello_map_win.py --ssid TELLO-1
    python windows\\tello_map_win.py --ssid TELLO-1 --tags-list 10,1,2,3,4 --out config\\map.yaml

Show tag 10 together with the other tags (two tags in one frame links them, a chain
10-6-5 works too). Green = placed in the tag-10 frame, yellow = seen but not linked yet.
q / s / Esc in the window, or Ctrl+C here, saves.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAL_AND_MAP = HERE.parent / "Action_Space" / "drone_deployment_infra" / "cal_and_map"
sys.path.insert(0, str(CAL_AND_MAP))
sys.path.insert(0, str(HERE))

from tello_calibration.calib.io import camera_from_dict, load_yaml  # noqa: E402
from tello_calibration.comms.tello_link import TelloLink  # noqa: E402
from tello_calibration.session import SessionError, video_stream  # noqa: E402
from tello_calibration.tags.detect import TagDetector, load_tag_config  # noqa: E402
from tello_calibration.utils.config import CONFIG_DIR  # noqa: E402
from tello_check import join, local_ip_toward_tello  # noqa: E402
from tello_map import map_loop  # noqa: E402


def associated_bssid(iface: str) -> str | None:
    """BSSID the adapter is associated with, from netsh (the nmcli check's stand-in)."""
    out = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True).stdout
    block = out.split(f"Name                   : {iface}\n")[-1].split("\n\n")[0]
    for line in block.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "BSSID" or key.strip() == "AP BSSID":
            return value.strip().lower()
    return None


def expected_bssid(ssid: str) -> str | None:
    dongles = json.loads((CONFIG_DIR / "dongles.json").read_text())["dongles"]
    return next((d["bssid"].lower() for d in dongles if d.get("ssid") == ssid and d.get("bssid")), None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ssid", required=True, help="drone WiFi, e.g. TELLO-1")
    parser.add_argument("--iface", default="Wi-Fi 2", help="dongle adapter name (netsh wlan show interfaces)")
    parser.add_argument("--tags-list", default="10,1,2,3,4,5,6,7,8,9", help="tag ids to map (10 = origin)")
    parser.add_argument("--camera", type=Path, help="default: config/camera/<drone>.yaml")
    parser.add_argument("--out", type=Path, help="default: config/map_<drone>_<time>.yaml (never overwrites map.yaml)")
    args = parser.parse_args()

    key = args.ssid.lower().replace("-", "_")  # TELLO-1 -> tello_1, the camera/fleet key
    out = args.out or CONFIG_DIR / f"map_{key}_{time.strftime('%Y%m%d_%H%M%S')}.yaml"
    designated = {int(x) for x in args.tags_list.split(",") if x.strip()}

    camera_path = args.camera or CONFIG_DIR / "camera" / f"{key}.yaml"
    if not camera_path.exists():
        print(f"no camera calibration at {camera_path}; pass --camera", flush=True)
        return 1
    camera = load_yaml(camera_path)
    if camera.get("drone") not in (None, key):
        print(f"[camera] NOTE: {camera_path.name} was calibrated on {camera['drone']}, not {key}; "
              "distances may be off by a few percent until this drone is calibrated", flush=True)
    K, dist = camera_from_dict(camera)
    tags = load_tag_config(CONFIG_DIR / "tags.yaml")
    designated.add(tags.origin_tag)
    detector = TagDetector(tags, K, dist)

    join(args.ssid, args.iface)
    got, want = associated_bssid(args.iface), expected_bssid(args.ssid)
    if want and got and got != want:
        print(f"[identity] {args.iface} is on BSSID {got}, but dongles.json has {want} for {args.ssid}: "
              "wrong drone, not mapping", flush=True)
        return 2
    print(f"[identity] {args.ssid} BSSID {got or '?'} " + ("matches dongles.json" if want else "(not in dongles.json)"))
    local_ip = local_ip_toward_tello()
    print(f"mapping {sorted(designated)} with {args.ssid}; origin = tag {tags.origin_tag}", flush=True)

    try:
        with TelloLink(local_ip) as link:
            reply = link.send("command", timeout=2.0, retries=3)
            print(f"[link] command -> {reply!r}, battery {link.send('battery?')}%", flush=True)
            with video_stream(link, local_ip) as video:
                import cv2
                try:
                    tag_map = map_loop(video, detector, tags.origin_tag, designated)
                finally:
                    cv2.destroyAllWindows()
    except SessionError as exc:
        print(f"video failed: {exc}", flush=True)
        return 1

    if tags.origin_tag not in tag_map.poses:
        print(f"origin tag {tags.origin_tag} never linked; no map written.", flush=True)
        return 1
    tag_map.save(out)
    for tag_id in tag_map.mapped():
        t = tag_map.poses[tag_id][:3, 3]
        print(f"tag {tag_id:2d}  x {t[0]:+.3f}  y {t[1]:+.3f}  z {t[2]:+.3f} m", flush=True)
    print(f"wrote {out} ({len(tag_map.mapped())} tags in tag-{tags.origin_tag} frame)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
