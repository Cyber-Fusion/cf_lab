# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Side-view diagram of the D555 camera FoV on the AYG quadruped.

Renders the actual AYG silhouette by parsing the URDF, doing forward kinematics
for a nominal standing pose, loading each link's STL, projecting vertices to the
X-Z plane, and drawing per-link convex hulls. Overlays the D555 camera FoV wedge
for several mount configurations.

Run:
    python scripts/sanity/d555_fov_side_view.py
Output:
    /tmp/d555_fov_side_view.png
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

# --- Paths -----------------------------------------------------------------
PKG_ROOT = Path(
    "/home/miq-hovsepyan/Desktop/CyberFusion/cf_lab_root/cf_lab/source/cf_lab/data/Robots/ayg_description"
)
URDF_PATH = PKG_ROOT / "urdf/ayg.urdf"
OUT_PATH = "/tmp/d555_fov_side_view.png"

# --- D555 contract ---------------------------------------------------------
HFOV_RAD = 1.5184  # 87 deg
ASPECT_H_OVER_W = 45.0 / 80.0
VFOV_RAD = 2.0 * math.atan(math.tan(HFOV_RAD / 2.0) * ASPECT_H_OVER_W)  # ~56 deg

# --- Nominal standing pose (rad) -------------------------------------------
# Solved by numerical search (see scripts/sanity/d555_fov_side_view.py history)
# so that foot ends up directly under hip (offset < 1 mm) and all four feet are
# at the same z. Same angles on every leg because the AYG chain is identical
# per leg; mirroring front/rear angles produces asymmetric foot z and is wrong.
# Net result: base_z = 0.41 m (matches AYG nominal standing height).
JOINT_ANGLES = {
    "LF_HAA": 0.0, "LF_HFE": +0.95, "LF_KFE": -2.30,
    "RF_HAA": 0.0, "RF_HFE": +0.95, "RF_KFE": -2.30,
    "LH_HAA": 0.0, "LH_HFE": +0.95, "LH_KFE": -2.30,
    "RH_HAA": 0.0, "RH_HFE": +0.95, "RH_KFE": -2.30,
}

# Camera mount (exact from URDF Base_2_Camera) — re-derived below from URDF too.
CAM_X = 0.2475
CAM_Z = 0.095


# --- URDF parsing + FK -----------------------------------------------------
def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split()]


def parse_urdf(urdf_path: Path):
    """Return (links_with_visual, joints_by_child)."""
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    links = {}
    for link in root.findall("link"):
        name = link.attrib["name"]
        visual = link.find("visual")
        if visual is None:
            continue
        mesh = visual.find("geometry/mesh")
        if mesh is None:
            continue
        origin = visual.find("origin")
        links[name] = {
            "mesh": mesh.attrib["filename"],
            "vis_xyz": _floats(origin.attrib.get("xyz", "0 0 0")) if origin is not None else [0, 0, 0],
            "vis_rpy": _floats(origin.attrib.get("rpy", "0 0 0")) if origin is not None else [0, 0, 0],
            "scale": _floats(mesh.attrib.get("scale", "1 1 1")),
        }

    joints = {}
    for j in root.findall("joint"):
        child = j.find("child").attrib["link"]
        parent = j.find("parent").attrib["link"]
        origin = j.find("origin")
        axis_el = j.find("axis")
        joints[child] = {
            "name": j.attrib["name"],
            "type": j.attrib["type"],
            "parent": parent,
            "xyz": _floats(origin.attrib.get("xyz", "0 0 0")) if origin is not None else [0, 0, 0],
            "rpy": _floats(origin.attrib.get("rpy", "0 0 0")) if origin is not None else [0, 0, 0],
            "axis": _floats(axis_el.attrib.get("xyz", "1 0 0")) if axis_el is not None else [1, 0, 0],
        }
    return links, joints


def rotmat_rpy(rpy: list[float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def rotmat_axis_angle(axis: list[float], angle: float) -> np.ndarray:
    axis = np.array(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-9 or abs(angle) < 1e-9:
        return np.eye(3)
    axis = axis / n
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def make_tf(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


def forward_kinematics(joints: dict, base_z: float, joint_angles: dict) -> dict:
    """Compute world 4x4 transforms for every link reachable from Base."""
    world = {"Base": make_tf(np.eye(3), [0.0, 0.0, base_z])}
    remaining = dict(joints)
    while remaining:
        progressed = False
        for child, j in list(remaining.items()):
            if j["parent"] not in world:
                continue
            T_joint = make_tf(rotmat_rpy(j["rpy"]), j["xyz"])
            angle = joint_angles.get(j["name"], 0.0)
            if j["type"] in ("revolute", "continuous") and abs(angle) > 0:
                T_motion = make_tf(rotmat_axis_angle(j["axis"], angle), [0, 0, 0])
                T = world[j["parent"]] @ T_joint @ T_motion
            else:
                T = world[j["parent"]] @ T_joint
            world[child] = T
            del remaining[child]
            progressed = True
        if not progressed:
            # Disconnected links (e.g., orphaned camera frames) — ok to drop.
            break
    return world


def settle_base_height(joints: dict, joint_angles: dict) -> float:
    """Find base z such that the lowest foot sits at z=0."""
    foot_links = [c for c, j in joints.items() if c.endswith("_Foot")]
    world = forward_kinematics(joints, base_z=0.0, joint_angles=joint_angles)
    feet_z = [world[fl][2, 3] for fl in foot_links if fl in world]
    # Want: base_z + feet_z_at_base0 == 0 -> base_z = -min(feet_z_at_base0)
    return -min(feet_z)


def resolve_mesh(uri: str) -> Path:
    # "package://ayg_description/meshes/Base.STL" -> PKG_ROOT/meshes/Base.STL
    m = re.match(r"package://([^/]+)/(.*)", uri)
    if m:
        return PKG_ROOT / m.group(2)
    return Path(uri)


def link_vertices_world(link_info: dict, link_world_tf: np.ndarray) -> np.ndarray | None:
    mesh_path = resolve_mesh(link_info["mesh"])
    if not mesh_path.exists():
        return None
    geom = trimesh.load(mesh_path, force="mesh")
    if not hasattr(geom, "vertices") or geom.vertices.shape[0] == 0:
        return None
    v = np.asarray(geom.vertices, dtype=float)
    scale = np.asarray(link_info["scale"], dtype=float)
    v = v * scale
    T_vis = make_tf(rotmat_rpy(link_info["vis_rpy"]), link_info["vis_xyz"])
    T_total = link_world_tf @ T_vis
    v_h = np.hstack([v, np.ones((v.shape[0], 1))])
    return (T_total @ v_h.T).T[:, :3]


def draw_ayg_side(ax: plt.Axes, links: dict, joints: dict, base_z: float, joint_angles: dict):
    world = forward_kinematics(joints, base_z, joint_angles)
    for name, info in links.items():
        if name not in world:
            continue
        verts = link_vertices_world(info, world[name])
        if verts is None or verts.shape[0] < 3:
            continue
        xz = verts[:, [0, 2]]
        # Per-link convex hull for a clean silhouette.
        try:
            hull = ConvexHull(xz)
            poly = xz[hull.vertices]
        except Exception:
            continue
        color = "#1976d2" if "Camera" in name else "#90a4ae"
        edge = "#0d47a1" if "Camera" in name else "#37474f"
        alpha = 0.85 if "Camera" in name else 0.55
        ax.add_patch(patches.Polygon(poly, closed=True,
                                     facecolor=color, edgecolor=edge,
                                     linewidth=0.6, alpha=alpha, zorder=2))
    return world


# --- FoV geometry ----------------------------------------------------------
def ground_x_of_ray(cam_x: float, cam_z: float, pitch_down: float) -> float:
    if pitch_down <= 1e-6:
        return math.inf
    return cam_x + cam_z / math.tan(pitch_down)


def draw_fov_and_blind(ax, cam_x, cam_z, tilt_down_rad, front_foot_x):
    half = VFOV_RAD / 2.0
    upper_pitch = tilt_down_rad - half  # negative -> ray pointing up
    lower_pitch = tilt_down_rad + half
    x_near = ground_x_of_ray(cam_x, cam_z, lower_pitch)
    x_far = ground_x_of_ray(cam_x, cam_z, upper_pitch)
    x_near_d = min(x_near, 5.0) if math.isfinite(x_near) else 5.0

    # FoV wedge to a 5 m horizon
    upper_end = (cam_x + 5.0 * math.cos(upper_pitch),
                 cam_z - 5.0 * math.sin(upper_pitch))
    lower_end = (x_near_d, 0.0) if math.isfinite(x_near) else (
        cam_x + 5.0 * math.cos(lower_pitch),
        cam_z - 5.0 * math.sin(lower_pitch),
    )
    ax.add_patch(patches.Polygon(
        [(cam_x, cam_z), upper_end, lower_end],
        closed=True, facecolor="#ffb300", alpha=0.22,
        edgecolor="#ff6f00", linewidth=1.0, zorder=1,
    ))

    if math.isfinite(x_near):
        ax.plot(x_near_d, 0, "o", color="#d32f2f", markersize=6, zorder=5)
        ax.annotate(f"first visible\nx={x_near:.2f} m",
                    (x_near_d, 0.02), fontsize=8, color="#d32f2f",
                    ha="center", va="bottom")
        if x_near > front_foot_x:
            blind = x_near - front_foot_x
            ax.annotate("", xy=(x_near_d, -0.08), xytext=(front_foot_x, -0.08),
                        arrowprops=dict(arrowstyle="<->", color="#d32f2f", lw=1.6))
            ax.text((front_foot_x + x_near_d) / 2.0, -0.16,
                    f"blind zone = {blind:.2f} m (in front of feet)",
                    fontsize=9, color="#d32f2f",
                    ha="center", va="top", fontweight="bold")
        return x_near, blind if x_near > front_foot_x else 0.0
    return math.inf, 0.0


# --- Main ------------------------------------------------------------------
def main():
    links, joints = parse_urdf(URDF_PATH)

    # Camera mount comes straight from URDF (sanity check):
    cam_joint = joints.get("Camera")
    if cam_joint is not None:
        urdf_cam_x, _, urdf_cam_z = cam_joint["xyz"]
        if abs(urdf_cam_x - CAM_X) > 1e-3 or abs(urdf_cam_z - CAM_Z) > 1e-3:
            print(f"[warn] camera mount in URDF ({urdf_cam_x:.4f}, {urdf_cam_z:.4f}) differs "
                  f"from script constants ({CAM_X}, {CAM_Z})")

    base_z = settle_base_height(joints, JOINT_ANGLES)
    print(f"[info] standing pose: base z = {base_z:.4f} m (feet on ground)")

    # Front foot x in the standing pose
    world = forward_kinematics(joints, base_z, JOINT_ANGLES)
    front_foot_x = float(np.mean([world["LF_Foot"][0, 3], world["RF_Foot"][0, 3]]))
    cam_world_z_base_mount = base_z + CAM_Z
    cam_world_x = CAM_X
    print(f"[info] front foot x = {front_foot_x:.4f} m, camera at ({cam_world_x:.4f}, {cam_world_z_base_mount:.4f})")

    scenarios = [
        ("1) CURRENT: horizontal mount",          0.0,  0.0),
        ("2) Same mount, tilt 15 deg down",       0.0, 15.0),
        ("3) Same mount, tilt 25 deg down",       0.0, 25.0),
        ("4) RAISE +0.10 m & tilt 20 deg down",   0.10, 20.0),
    ]

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(11, 14), sharex=True)
    for ax, (title, dz, tilt_deg) in zip(axes, scenarios):
        # Ground
        ax.fill_between([-0.7, 5.2], -0.2, 0, color="#d2b48c", alpha=0.45)
        ax.axhline(0, color="#3e2723", linewidth=1.0)

        # AYG silhouette
        draw_ayg_side(ax, links, joints, base_z, JOINT_ANGLES)

        # Effective camera height for THIS panel (raise extends mount up)
        cam_z_eff = cam_world_z_base_mount + dz
        cam_x_eff = cam_world_x
        # FoV + blind
        x_near, blind = draw_fov_and_blind(ax, cam_x_eff, cam_z_eff,
                                           math.radians(tilt_deg), front_foot_x)

        # Camera marker (small)
        ax.add_patch(patches.Rectangle((cam_x_eff - 0.02, cam_z_eff - 0.02),
                                       0.05, 0.05,
                                       linewidth=1, edgecolor="black",
                                       facecolor="#1976d2", zorder=5))

        ax.set_title(
            f"{title}\nh_cam={cam_z_eff:.2f} m  tilt={tilt_deg:+.0f}°  "
            f"VFoV={math.degrees(VFOV_RAD):.0f}°  "
            f"front_foot_x={front_foot_x:.2f} m",
            fontsize=10,
        )
        ax.set_xlim(-0.7, 5.2)
        ax.set_ylim(-0.25, 1.05)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.set_ylabel("z (m)")
    axes[-1].set_xlabel("x forward of Base center (m)")

    fig.suptitle(
        "AYG D555 side-view FoV — actual robot silhouette from URDF",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    print(f"[OK] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
