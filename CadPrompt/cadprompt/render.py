"""render.py — views of a CAD model around a sphere, by CPU ray casting (Open3D), no OpenGL needed.

Reads what trimesh reads: .obj/.mtl with a texture, .glb/.gltf, .ply with vertex colours or with a texture named in
a "comment TextureFile" header line (BOP models), .stl. Colour comes from the texture (UV
interpolated with the hit's barycentric coordinates) or the vertex colours, shaded with a head light.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


class Model:
    """triangles, and a colour function: colour(face ids, barycentric u, v) -> (N,3) float 0..1"""

    def __init__(self, path):
        path = Path(path)
        m = trimesh.load(path, force="mesh", process=False)
        self.V = np.asarray(m.vertices, np.float64)
        self.F = np.asarray(m.faces, np.int64)
        self.name = path.stem
        self.uv = self.tex = self.vcol = None
        vis = m.visual
        if getattr(vis, "kind", None) == "texture" and getattr(vis, "uv", None) is not None:
            img = getattr(vis.material, "image", None) or getattr(vis.material, "baseColorTexture", None)
            if img is not None:
                self.uv = np.asarray(vis.uv, np.float64)
                self.tex = np.asarray(img.convert("RGB"), np.float64) / 255
        if self.tex is None:
            try:
                c = np.asarray(vis.to_color().vertex_colors if vis.kind == "texture" else vis.vertex_colors, np.float64)
                if len(c) == len(self.V) and np.ptp(c[:, :3]) > 0:
                    self.vcol = c[:, :3] / 255
            except Exception:  # noqa: BLE001 - no usable colour: plain grey below
                pass
        self.size = float(np.linalg.norm(np.ptp(self.V, axis=0)))   # bounding-box diagonal, model units
        self.center = (self.V.min(0) + self.V.max(0)) / 2
        self.extent = np.ptp(self.V, axis=0)

    @property
    def coloured(self):
        return self.tex is not None or self.vcol is not None

    def colour(self, fid, u, v):
        w = np.column_stack([1 - u - v, u, v])
        tri = self.F[fid]
        if self.tex is not None:
            uv = np.einsum("nk,nkj->nj", w, self.uv[tri])
            h, wd = self.tex.shape[:2]
            x = np.clip((uv[:, 0] % 1.0) * (wd - 1), 0, wd - 1).astype(int)
            y = np.clip((1 - uv[:, 1] % 1.0) * (h - 1), 0, h - 1).astype(int)
            return self.tex[y, x]
        if self.vcol is not None:
            return np.einsum("nk,nkj->nj", w, self.vcol[tri])
        return np.full((len(fid), 3), 0.6)


def view_dirs(n):
    """n directions spread evenly over the sphere (Fibonacci), from where the camera looks at the object"""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    phi = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def render(model: Model, n_views=24, size=384, fov_deg=40.0, fill=0.8, up=(0, 0, 1)):
    """list of (rgb uint8 HxWx3, mask bool HxW, camera direction) for n_views around the object"""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(model.V.astype(np.float32)), o3d.core.Tensor(model.F.astype(np.uint32)))
    f = 0.5 * size / np.tan(np.radians(fov_deg) / 2)
    dist = 0.5 * model.size / np.tan(np.radians(fov_deg) / 2) / fill
    out = []
    for d in view_dirs(n_views):
        eye = model.center + d * dist
        fwd = -d
        upv = np.asarray(up, float)
        if abs(fwd @ upv) > 0.95:                              # looking straight up or down: any other up
            upv = np.array([0, 1.0, 0])
        right = np.cross(fwd, upv); right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        vv, uu = np.mgrid[0:size, 0:size]
        rays_d = ((uu - size / 2 + 0.5) / f)[..., None] * right + ((vv - size / 2 + 0.5) / f)[..., None] * down + fwd
        rays_d /= np.linalg.norm(rays_d, axis=-1, keepdims=True)
        rays = np.concatenate([np.broadcast_to(eye, rays_d.shape), rays_d], -1).reshape(-1, 6).astype(np.float32)
        hit = scene.cast_rays(o3d.core.Tensor(rays))
        t = hit["t_hit"].numpy()
        mask = np.isfinite(t)
        fid = hit["primitive_ids"].numpy()[mask].astype(np.int64)
        buv = hit["primitive_uvs"].numpy()[mask]
        nrm = hit["primitive_normals"].numpy()[mask]
        shade = 0.35 + 0.65 * np.abs(np.sum(nrm * rays[mask, 3:], 1))
        rgb = np.full((size * size, 3), 0.0)
        rgb[mask] = np.clip(model.colour(fid, buv[:, 0], buv[:, 1]) * shade[:, None], 0, 1)
        out.append(((rgb.reshape(size, size, 3) * 255).astype(np.uint8), mask.reshape(size, size), d))
    return out
