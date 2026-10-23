"""Windows check for one Tello on a USB WiFi dongle: link, status, and what the camera sees.

NEVER sends takeoff. The repo's own tools (core/tello_link.py, drone_gui) need Linux
(nmcli, SO_BINDTODEVICE); on Windows the dongle is just a second WiFi adapter, so one
drone at a time works with plain djitellopy.

    conda activate tello
    python windows\\tello_check.py --ssid TELLO-4               # join, status, live video
    python windows\\tello_check.py --ssid TELLO-4 --no_video    # join + status only
    python windows\\tello_check.py                              # already joined: skip netsh

Video window: green boxes are AprilTags (tag36h11) with their id; q or Esc quits.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import tempfile
import time

TELLO_IP = "192.168.10.1"

PROFILE_XML = """<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
  <name>{ssid}</name>
  <SSIDConfig><SSID><name>{ssid}</name></SSID></SSIDConfig>
  <connectionType>ESS</connectionType>
  <connectionMode>manual</connectionMode>
  <MSM><security><authEncryption>
    <authentication>open</authentication><encryption>none</encryption><useOneX>false</useOneX>
  </authEncryption></security></MSM>
</WLANProfile>
"""


def local_ip_toward_tello() -> str | None:
    """The address Windows would send from to reach the Tello, or None without a route."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.connect((TELLO_IP, 8889))  # UDP connect sends nothing; it only picks a route
            return s.getsockname()[0]
        except OSError:
            return None


def join(ssid: str, iface: str, timeout: float = 60.0) -> None:
    """Join the drone's open WiFi on the dongle (the laptop's main WiFi is untouched)."""
    ip = local_ip_toward_tello()
    if ip and ip.startswith("192.168.10."):
        # re-issuing netsh connect tears down a working link, and the Tello may not come back
        print(f"[wifi] already on a Tello network (laptop address {ip}), not reconnecting")
        return
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as f:
        f.write(PROFILE_XML.format(ssid=ssid))
    try:
        subprocess.run(["netsh", "wlan", "add", "profile", f"filename={f.name}",
                        f"interface={iface}", "user=current"], capture_output=True, check=True)
    finally:
        os.unlink(f.name)
    out = subprocess.run(["netsh", "wlan", "connect", f"name={ssid}", f"ssid={ssid}",
                          f"interface={iface}"], capture_output=True, text=True)
    print(f"[wifi] {out.stdout.strip() or out.stderr.strip()}")
    deadline = time.time() + timeout
    while time.time() < deadline:
        ip = local_ip_toward_tello()
        if ip and ip.startswith("192.168.10."):
            print(f"[wifi] {iface} joined {ssid}, laptop address {ip}")
            return
        time.sleep(1)
    status = subprocess.run(["netsh", "wlan", "show", "interfaces"], capture_output=True, text=True).stdout
    block = status.split(f"Name                   : {iface}")[-1].split("\n\n")[0]
    state = next((ln.split(":", 1)[1].strip() for ln in block.splitlines() if ln.strip().startswith("State")), "?")
    raise SystemExit(f"[wifi] no route to {TELLO_IP} after {timeout:.0f}s; {iface} is '{state}'. "
                     f"'disconnected' after joining usually means another device holds {ssid} "
                     "(a Tello takes one client) or the drone's WiFi timed out: power-cycle it and retry")


def print_status(tello) -> None:
    state = tello.get_current_state()
    print("[status]")
    print(f"  battery      {tello.get_battery()} %")
    print(f"  temperature  {state.get('templ')}-{state.get('temph')} C")
    print(f"  height       {tello.get_height()} cm (ToF {tello.get_distance_tof()} cm)")
    print(f"  attitude     pitch {state.get('pitch')}  roll {state.get('roll')}  yaw {state.get('yaw')}")
    print(f"  flight time  {tello.get_flight_time()} s")
    for name, query in (("wifi SNR", tello.query_wifi_signal_noise_ratio),
                        ("SDK", tello.query_sdk_version),
                        ("serial", tello.query_serial_number)):
        try:
            print(f"  {name:<12} {query()}")
        except Exception as exc:  # noqa: BLE001  (non-EDU firmware rejects some queries)
            print(f"  {name:<12} n/a ({exc})")


def show_video(tello) -> None:
    import cv2
    import pupil_apriltags

    detector = pupil_apriltags.Detector(families="tag36h11", quad_decimate=2.0)
    tello.streamon()
    reader = tello.get_frame_read()
    print("[video] window open: q or Esc to quit (allow python through Windows Firewall if asked)")
    start, frames, last_status = time.time(), 0, 0.0
    hud = ""
    while True:
        rgb = reader.frame
        if rgb is None or rgb.shape[0] < 400:  # djitellopy's placeholder until the first frame
            if time.time() - start > 10 and frames == 0:
                print("[video] no frames after 10 s: check Windows Firewall allows UDP 11111 for python")
                start = time.time()
            time.sleep(0.05)
            continue
        frames += 1
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        tags = detector.detect(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        for t in tags:
            pts = t.corners.astype(int)
            cv2.polylines(bgr, [pts.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
            cv2.putText(bgr, f"id {t.tag_id}", tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if time.time() - last_status > 1.0:
            last_status = time.time()
            hud = f"bat {tello.get_battery()}%  h {tello.get_height()}cm  tags {[t.tag_id for t in tags]}"
        cv2.putText(bgr, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow("Tello camera", bgr)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    tello.streamoff()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ssid", help="drone WiFi to join first, e.g. TELLO-4")
    parser.add_argument("--iface", default="Wi-Fi 2", help="dongle adapter name (netsh wlan show interfaces)")
    parser.add_argument("--no_video", action="store_true", help="status only")
    args = parser.parse_args()

    if args.ssid:
        join(args.ssid, args.iface)
    elif not (local_ip_toward_tello() or "").startswith("192.168.10."):
        raise SystemExit("not on a Tello network: pass --ssid TELLO-N, or join it in Windows WiFi settings")

    from djitellopy import Tello

    tello = Tello(TELLO_IP)
    tello.connect()  # SDK handshake ("command" -> "ok"); raises if the drone doesn't answer
    print("[link] drone answered the SDK handshake")
    # a garbled first reply makes connect() resend, and the late second "ok" would then
    # answer the next query: let it arrive and drop it so replies line up again
    time.sleep(0.5)
    tello.get_own_udp_object()["responses"].clear()
    try:
        print_status(tello)
        if not args.no_video:
            show_video(tello)
    finally:
        tello.end()


if __name__ == "__main__":
    main()
