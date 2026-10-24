"""Policy comparison page: breach table for every policy x scenario, then the same fast-attacker
episodes flown by each policy (static SVG, no script). Output: spoof_test.html."""
import json
from pathlib import Path

HERE = Path(__file__).parent
POLICIES = [("mecam", "Ours + cameras (new)", "defend_policy_me", "guard, camera positions; shipped default"),
            ("softcam", "Shibani soft + cameras (new)", "defend_policy_shib_soft", "camera positions; shipped default"),
            ("hardcam", "Shibani hard + cameras + chaser (new)", "defend_policy_shib_hard", "camera positions, Mo's chaser; shipped default"),
            ("me", "Ours, dead reckoning (before)", "defend_policy_me", "DEFEND_CAMERA=0"),
            ("soft", "Shibani soft, dead reckoning (before)", "defend_policy_shib_soft", "soft_contact_es_g10, DEFEND_CAMERA=0"),
            ("hard", "Shibani hard, dead reckoning (before)", "defend_policy_shib_hard", "normal_contact_es_g40, DEFEND_CAMERA=0"),
            ("roh", "Rohan", "defend_policy_roh", "B7200 MAPPO, 35 inputs"),
            ("mo", "Mo", "defend_policy_mo", "Rohan's B7200 + fallback chaser")]
COLS = ["Curved paths to tag 6, slow 0.3–0.5 m/s", "Same, fast 0.6–0.9 m/s", "Rohan's task: straight to the tag-6 line, 0.4–0.9 m/s",
        "Rohan's task, no noise or lag (training-like)", "Drones 33% slower than assumed, slow", "Same, fast",
        "Messy drones: slow", "Messy: fast", "Messy: Rohan's task"]
BREACH = {"mecam": [0.0, 2.5, 1.0, 0.0, 0.0, 2.5, 0.0, 1.0, 2.5], "softcam": [0.0, 5.0, 3.0, 0.0, 2.0, 11.5, 0.0, 3.5, 6.0],
          "hardcam": [0.5, 20.0, 35.5, 8.0, 7.5, 33.0, 1.0, 21.0, 27.5],
          "me": [0.0, 0.0, 1.0, 0.0, 7.0, 13.5, 23.5, 17.0, 18.5], "soft": [0.0, 5.0, 4.0, 0.5, 9.0, 19.5, 23.5, 19.0, 28.0],
          "hard": [0.5, 32.5, 40.5, 11.5, 5.5, 51.0, 24.0, 43.5, 48.5], "roh": [57.5, 66.0, 60.0, 36.0, 52.0, 70.5, 50.0, 61.5, 62.0],
          "mo": [1.5, 21.5, 0.5, 0.0, 1.0, 11.0, 1.5, 20.0, 0.0]}
EPISODES = [1, 4, 5]  # 0-based picks from the six recorded fast-attacker episodes

X0, X1, Z0, Z1 = -1.0, 3.5, 0.0, 7.2
W, H, PAD = 300, 440, 18
s = min((W - 2 * PAD) / (X1 - X0), (H - 2 * PAD) / (Z1 - Z0))
P = lambda x, z: (PAD + (x - X0) * s, PAD + (z - Z0) * s)


def svg(e, label):
    out = ['<rect class="room" x="%.1f" y="%.1f" width="%.1f" height="%.1f"/>' % (*P(X0, Z0), (X1 - X0) * s, (Z1 - Z0) * s)]
    for z in range(1, 8):
        a, b = P(X0, z), P(X1, z)
        out.append('<line class="grid" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/><text class="ax" x="%.1f" y="%.1f">z %d</text>' % (*a, *b, a[0] + 2, a[1] - 2, z))
    out.append('<line class="wall" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>' % (*P(X0, 0), *P(X1, 0)))
    t10 = P(0, 0)
    out.append('<rect class="t10" x="%.1f" y="%.1f" width="10" height="6"/><text class="ax" x="%.1f" y="%.1f">tag 10</text>' % (t10[0] - 5, t10[1], t10[0] + 8, t10[1] + 12))
    g = P(*e["goal"])
    out.append('<circle class="goal" cx="%.1f" cy="%.1f" r="%.1f"/><text class="ax" x="%.1f" y="%.1f">goal</text>' % (*g, 0.75 * s, g[0] + 0.8 * s, g[1] + 4))
    for cls, group in (("att", e["att"]), ("def", e["def"])):
        for pts in group.values():
            if pts:
                out.append('<polyline class="%s" points="%s"/><circle class="%sdot" cx="%.1f" cy="%.1f" r="3.5"/>'
                           % (cls, " ".join("%.1f,%.1f" % P(*p) for p in pts), cls, *P(*pts[0])))
    for kind, x, z in e["events"]:
        c = P(x, z)
        out.append('<g class="%s"><line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/><line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/></g>'
                   % (kind, c[0] - 6, c[1] - 6, c[0] + 6, c[1] + 6, c[0] - 6, c[1] + 6, c[0] + 6, c[1] - 6))
    return '<svg viewBox="0 0 %d %d" role="img" aria-label="%s">%s</svg>' % (W, H, label, "".join(out))


def cell(v):
    if v is None:
        return '<td class="num na">not run</td>'
    level = "good" if v <= 2 else "ok" if v <= 10 else "warn" if v <= 30 else "bad"
    return '<td class="num %s">%.1f%%</td>' % (level, v)


head = "".join("<th>%s</th>" % c for c in COLS)
rows = "".join('<tr><th scope="row">%s<span>%s · drive/%s</span></th>%s</tr>' % (name, note, folder, "".join(cell(v) for v in BREACH[k]))
               for k, name, folder, note in POLICIES)
table = '<div class="tablewrap"><table class="cmp"><thead><tr><th>Policy</th>%s</tr></thead><tbody>%s</tbody></table></div>' % (head, rows)

blocks = []
for k, name, folder, note in POLICIES:
    if not (HERE / ("tr_%s.json" % k)).exists():
        continue
    eps = json.loads((HERE / ("tr_%s.json" % k)).read_text())
    cards = []
    for i in EPISODES:
        e = eps[i]
        h = "Episode %d: %s, %d of 2 caught" % (i + 1, "BREACH" if e["result"] == "breach" else "held", e["caught"])
        cards.append('<div class="ep %s"><h3>%s</h3>%s</div>' % (e["result"], h, svg(e, name + " " + h)))
    blocks.append('<section class="pol"><h2>%s <span>%s</span></h2><div class="eps">%s</div></section>' % (name, note, "".join(cards)))

page = (HERE / "compare_tpl.html").read_text(encoding="utf-8").replace("__TABLE__", table).replace("__BLOCKS__", "\n".join(blocks))
(HERE / "spoof_test.html").write_text(page, encoding="utf-8")
print(len(page))
