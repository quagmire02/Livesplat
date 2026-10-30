"""Lab tab: scale, equipment, tests (transport, gripper fit, arm grasp), result cards and report download.

Transport and gripper fit run in the viewer (seconds). The arm grasp test exports metric sim packages and runs
many dynamic trials, so it runs as a separate process (python -m lab.grasp) and its JSON is loaded when done.
Cached results in <run>/lab/*.json are shown at once.
"""
import html
import json

import numpy as np
import subprocess
import sys
import threading
from pathlib import Path

ARMS = {"SO-100": "so100", "Panda": "panda", "xArm7": "xarm7",
        "ViperX 300 S (unverified)": "vx300s", "WidowX 250 S (unverified)": "wx250s"}   # Trossen specs: check vs sim.synthetic
GRIPPERS = {"SO-100 jaw": "so100_jaw", "Panda hand": "panda_hand", "Robotiq 2F-85": "robotiq_2f85",
            "Suction cup 30 mm": "suction_30mm"}
SURFACES = {"wood": "wood", "plastic tray": "plastic", "rubber mat": "rubber"}


class LabPanel:
    def __init__(self, server, sc, run_dir, src, ctrl, notify, panel_html):
        self.server, self.sc, self.run, self.src, self.ctrl = server, sc, Path(run_dir), src, ctrl
        self.notify, self.html = notify, panel_html
        self.busy = threading.Lock()
        self.last = None                                    # (kind, result dict) for the report
        gui = server.gui
        gui.add_markdown("**Lab bench**: test your equipment on these real objects. Each test runs many "
                         "simulated trials over what we don't know exactly (weight, friction, size).")
        self.scale_md = gui.add_markdown("")
        self.size = gui.add_text("known size", "", hint="e.g. 'can 15 cm tall' or 'bottle 7 cm wide': calibrates "
                                 "the whole scene. Without it, size-dependent results are tested at +-spread.")
        gui.add_button("calibrate").on_click(lambda _: self.calibrate(self.size.value))
        self.arm = gui.add_dropdown("robot arm", list(ARMS), initial_value="SO-100")
        self.gripper = gui.add_dropdown("gripper", list(GRIPPERS), initial_value="SO-100 jaw")
        self.surface = gui.add_dropdown("carrier surface", list(SURFACES), hint="for the transport test")
        tests = gui.add_button_group("run test", ["grasp (arm)", "gripper fit", "transport"])
        tests.on_click(lambda _: self.start(tests.value))
        from lab.camera import CAMERAS, load_camera
        self.cams = {load_camera(p.stem)["name"]: p.stem for p in sorted(CAMERAS.glob("*.yaml"))}
        self.camera = gui.add_dropdown("camera", list(self.cams), initial_value=next(iter(self.cams)))
        cam_btn = gui.add_button_group("camera test", ["best spot", "my view", "export dataset"])
        cam_btn.on_click(lambda _: self.start_camera(cam_btn.value))
        self.cam_view = None                                # (cam spec, c2w) of the last camera test
        self.status = gui.add_markdown("")
        gui.add_button("download report").on_click(lambda _: self.download())
        self.refresh_scale()

    # ---- scale -------------------------------------------------------------------------------
    def refresh_scale(self):
        from measure.scale import load
        s = load(self.run)
        if not s.get("metres_per_unit"):
            self.scale_md.content = "Scale: **unknown** (type a known size)."
            return
        cal = str(s.get("source", "")).startswith("user")
        self.scale_md.content = (f"Scale: 1 unit = {s['metres_per_unit'] * 100:.1f} cm, "
                                 + ("**calibrated**" if cal else f"estimated ±{int(s['spread'] * 100)}%")
                                 + f" ({html.escape(str(s.get('source', '')))})")

    def calibrate(self, text):
        from measure.scale import calibrate, parse_size
        from viewer.route import find_object
        got = parse_size(text)
        k = find_object(text, self.ctrl.names)
        if not got or k is None:
            self.notify("Say which object and its size, e.g. 'can 15 cm tall'.")
            return
        dim, metres = got
        o = self.ctrl.obj(k)
        pts = o["capture_means"][o["body_ids"]] if "body_ids" in o else o["capture_means"]
        s = calibrate(self.run, pts, dim, metres, self.ctrl.label(k))
        self.refresh_scale()
        self.notify(f"Calibrated: {self.ctrl.label(k)} {dim} = {metres * 100:.1f} cm -> 1 unit = "
                    f"{s['metres_per_unit'] * 100:.1f} cm. Re-run size-dependent tests.")

    # ---- tests -------------------------------------------------------------------------------
    def _names(self):
        return {k: v[0] for k, v in self.ctrl.names.items()}

    def start(self, kind):
        if not self.busy.acquire(blocking=False):
            self.notify("A lab test is already running.")
            return
        self.status.content = f"Running {kind}…"
        self.notify(f"Running the {kind} test…")

        def work():
            try:
                if kind == "transport":
                    from lab.transport import run, summary
                    res = run(self.sc, self._names(), SURFACES[self.surface.value], self.run)
                    self._save(f"transport_{res['surface']}", res)
                    self.show("transport", res, summary(res))
                elif kind == "gripper fit":
                    from lab.gripper import run, summary
                    props = json.loads((self.run / "props.json").read_text())
                    res = run(self.sc, self.run, GRIPPERS[self.gripper.value], self._names(), props)
                    self._save(f"gripper_{GRIPPERS[self.gripper.value]}", res)
                    self.show("gripper fit", res, summary(res))
                else:
                    robot = ARMS[self.arm.value]
                    cached = self.run / "lab" / f"grasp_{robot}.json"
                    self.status.content = (f"Running the {self.arm.value} grasp test in the background "
                                           "(exports metric scenes, many trials; a few minutes)…")
                    p = subprocess.run([sys.executable, "-m", "lab.grasp", str(self.run), "--robot", robot],
                                       capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
                    if p.returncode != 0 or not cached.exists():
                        raise RuntimeError((p.stderr or p.stdout)[-300:])
                    from lab.grasp import summary
                    res = json.loads(cached.read_text())
                    self.show("grasp", res, summary(res))
                self.status.content = "Done. The card is on the left; 'download report' saves it."
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.status.content = f"Test failed: {html.escape(str(e))[:300]}"
                self.notify("Lab test failed (details in the Lab tab)")
            finally:
                self.busy.release()
        threading.Thread(target=work, daemon=True).start()

    # ---- camera ------------------------------------------------------------------------------
    def _client(self):
        cl = list(self.server.get_clients().values())
        return cl[0] if cl else None

    def _render(self, cam, c2w, width=640):
        """Photoreal RGB of the live scene from this camera (the browser renders the splats)."""
        from scipy.spatial.transform import Rotation
        from lab.camera import intrinsics
        cl = self._client()
        if cl is None:
            return None
        K, W, H, f = intrinsics(cam)
        x, y, z, w = Rotation.from_matrix(c2w[:3, :3]).as_quat()
        h = int(width * H / W)
        try:
            return cl.get_render(height=h, width=width, wxyz=(w, x, y, z), position=tuple(c2w[:3, 3]),
                                 fov=float(2 * np.arctan(H / 2 / f)), timeout=20)
        except Exception as e:                              # no render (tab hidden, closed): metrics still work
            print("render failed:", e, flush=True)
            return None

    def start_camera(self, what):
        if not self.busy.acquire(blocking=False):
            self.notify("A lab test is already running.")
            return

        def work():
            try:
                from lab.camera import auto_place, evaluate, load_camera, look_at, scene_points, suggest
                from measure.scale import load
                cam = load_camera(self.cams[self.camera.value])
                mpu = load(self.run).get("metres_per_unit") or 0.6
                if what == "export dataset":
                    self.export_dataset(cam, mpu)
                    return
                self.status.content = f"Testing {cam['name']}…"
                objs, bg = scene_points(self.sc, self.run)
                names = self._names()
                if what == "my view":
                    cl = self._client()
                    pos, look = np.asarray(cl.camera.position), np.asarray(cl.camera.look_at)
                    c2w = look_at(pos, look)
                    res = evaluate(cam, c2w, objs, bg, mpu, names)
                    pose = "your current view"
                else:
                    c2w, res, p = auto_place(cam, objs, bg, mpu, names, self.sc)
                    pose = (f"best spot: {p['distance_m']} m away, {p['elevation_deg']}° up, "
                            f"azimuth {p['azimuth_deg']}°")
                tips = suggest(res, cam)
                res.update(camera=cam["name"], pose=pose, c2w=np.round(c2w, 5).tolist(), suggestions=tips)
                self.cam_view = (cam, c2w)
                self._save(f"camera_{cam['id']}", res)
                self._frustum(cam, c2w)
                text = (f"{cam['name']} from {pose}: view score {res['score']} (weakest object).\n" +
                        "\n".join(f"- {v['name']}: {int(v['visible'] * 100)}% visible, {v['pixels']} px, "
                                  f"{v['distance_m']} m" + (f", depth ±{v['depth_sigma_mm']} mm"
                                                             if v['depth_sigma_mm'] is not None else "")
                                  + ("" if v["in_range"] else " (OUT OF RANGE)")
                                  for v in res["objects"].values()) +
                        "".join(f"\nSuggestion: {t}" for t in tips))
                self.show("camera", res, text)
                self.status.content = "Camera placed (orange frustum). 'export dataset' saves 12 labelled views."
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.status.content = f"Camera test failed: {html.escape(str(e))[:300]}"
            finally:
                self.busy.release()
        threading.Thread(target=work, daemon=True).start()

    def _frustum(self, cam, c2w):
        from scipy.spatial.transform import Rotation
        from lab.camera import intrinsics
        K, W, H, f = intrinsics(cam)
        x, y, z, w = Rotation.from_matrix(c2w[:3, :3]).as_quat()
        img = self._render(cam, c2w, 320)
        self.server.scene.add_camera_frustum("/lab/camera", fov=float(2 * np.arctan(H / 2 / f)), aspect=W / H,
                                             scale=0.08 * self.sc.radius, wxyz=(w, x, y, z),
                                             position=tuple(c2w[:3, 3]), color=(255, 140, 66), image=img)

    def export_dataset(self, cam, mpu, n=12):
        """n views on a ring around the objects (through the tested camera's height/distance): RGB from the
        browser render, sensor-model depth (uint16 mm), object-id mask, intrinsics + OpenCV c2w (metres)."""
        import io
        import zipfile
        import cv2
        from lab.camera import depth_and_mask, intrinsics, look_at, scene_points
        objs, bg = scene_points(self.sc, self.run)
        centre = np.mean([p.mean(0) for p, _ in objs.values()], 0)
        ref = self.cam_view[1][:3, 3] if self.cam_view else centre + np.array([0.6, 0, 0.5]) / mpu
        rel = ref - centre
        rad, height = np.linalg.norm(rel[:2]), rel[2]
        K, W, H, f = intrinsics(cam)
        rgb_w = min(W, 960)                                 # browser renders must fit the window
        K_rgb = np.diag([rgb_w / W, rgb_w / W, 1]) @ K
        buf = io.BytesIO()
        frames = []
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for i in range(n):
                self.status.content = f"Exporting view {i + 1}/{n}…"
                a = np.arctan2(rel[1], rel[0]) + 2 * np.pi * i / n
                pos = centre + np.array([rad * np.cos(a), rad * np.sin(a), height])
                c2w = look_at(pos, centre)
                rgb = self._render(cam, c2w, rgb_w)
                depth, mask = depth_and_mask(cam, c2w, objs, bg, mpu, np.random.default_rng(i))
                if rgb is not None:
                    zf.writestr(f"rgb/{i:03d}.png", cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))[1])
                if depth is not None:
                    zf.writestr(f"depth/{i:03d}.png", cv2.imencode(".png", (depth * 1000).astype(np.uint16))[1])
                zf.writestr(f"mask/{i:03d}.png", cv2.imencode(".png", mask)[1])
                c2w_m = c2w.copy()
                c2w_m[:3, 3] *= mpu
                frames.append({"file": f"{i:03d}", "c2w_opencv_m": np.round(c2w_m, 6).tolist()})
            zf.writestr("meta.json", json.dumps({
                "camera": cam["name"], "rgb_size": [rgb_w, int(rgb_w * H / W)], "K_rgb": np.round(K_rgb, 3).tolist(),
                "depth_mask_size": [W // 4, H // 4], "K_depth_mask": np.round(np.diag([.25, .25, 1]) @ K, 3).tolist(),
                "depth": "uint16 millimetres, sensor noise model, 0 = no return",
                "mask": "uint8 object id (0 = background)", "objects": self._names(),
                "frame": "table frame, z up, metres (scale " + str(round(mpu, 4)) + " m/unit)",
                "frames": frames}, indent=1))
        self.server.send_file_download(f"livesplat_dataset_{cam['id']}.zip", buf.getvalue())
        self.status.content = f"Dataset with {n} views downloading."
        self.notify(f"{n}-view labelled dataset downloading")

    def _save(self, name, res):
        (self.run / "lab").mkdir(exist_ok=True)
        (self.run / "lab" / f"{name}.json").write_text(json.dumps(res, indent=1))

    # ---- card + report -----------------------------------------------------------------------
    def show(self, kind, res, text):
        self.last = (kind, res, text)
        lines = text.split("\n")
        rows = "".join(f"<li>{html.escape(x[2:])}</li>" for x in lines[1:] if x.startswith("- "))
        tips = "".join(f"<p style='color:#ffb27a'>{html.escape(x)}</p>" for x in lines[1:] if x.startswith("Suggestion"))
        self.html.content = (
            '<aside style="position:fixed;left:12px;top:60px;width:340px;max-height:80vh;overflow:auto;'
            'background:rgba(30,30,30,.95);color:#eee;padding:14px;border:1px solid #4aa3ff;border-radius:10px;'
            'z-index:27;font-size:13px;line-height:1.45">'
            f'<b style="font-size:15px;color:#7cc0ff">Lab · {html.escape(kind)}</b>'
            f'<p>{html.escape(lines[0])}</p><ul style="padding-left:18px">{rows}</ul>{tips}'
            '<small>Simulated with estimated properties; results are ranges, not guarantees. '
            'Download the report for every trial.</small></aside>')
        self.notify(lines[0][:160])

    def download(self):
        if not self.last:
            self.notify("Run a lab test first.")
            return
        kind, res, text = self.last
        body = "".join(f"<p>{html.escape(x)}</p>" for x in text.split("\n"))
        page = (f"<!doctype html><meta charset='utf-8'><title>LiveSplat lab report: {html.escape(kind)}</title>"
                "<style>body{font-family:sans-serif;max-width:900px;margin:24px auto;padding:0 16px}"
                "pre{background:#f4f4f4;padding:12px;overflow:auto;font-size:12px}</style>"
                f"<h1>LiveSplat lab report: {html.escape(kind)}</h1><p>Scene: {html.escape(self.run.name)}</p>{body}"
                "<h2>All trials (JSON)</h2><pre>" + html.escape(json.dumps(res, indent=1)) + "</pre>"
                "<p><small>Simulated (MuJoCo) on a Gaussian-splat scan; mass/friction are model estimates; "
                "scale as stated above.</small></p>")
        self.server.send_file_download(f"lab_report_{kind.replace(' ', '_')}.html", page.encode())
        self.notify("Report downloading")
