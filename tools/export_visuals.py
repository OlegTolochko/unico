#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


TYPE_NAMES = {
    0: "plane",
    1: "sphere",
    2: "cylinder",
    3: "cone",
}


@dataclass(frozen=True)
class Shape:
    pos: np.ndarray
    direction: np.ndarray
    color: tuple[int, int, int]
    r1: float
    r2: float
    type_id: int


@dataclass(frozen=True)
class SegData:
    path: Path
    xyz: np.ndarray
    normals: np.ndarray
    segments: np.ndarray
    shapes: list[Shape]


@dataclass
class MeshPart:
    name: str
    material: str
    color: tuple[int, int, int]
    vertices: list[np.ndarray]
    faces: list[tuple[int, ...]]


def _normalize(v: np.ndarray, fallback: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if not np.all(np.isfinite(v)):
        return np.asarray(fallback, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if not math.isfinite(norm) or norm < 1e-12:
        return np.asarray(fallback, dtype=np.float64)
    return v / norm


def _basis_from_axis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = _normalize(axis)
    helper = np.array([0.0, 0.0, 1.0]) if abs(float(w[2])) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _normalize(np.cross(helper, w), fallback=(1.0, 0.0, 0.0))
    v = _normalize(np.cross(w, u), fallback=(0.0, 1.0, 0.0))
    return u, v, w


def _palette(idx: int) -> tuple[int, int, int]:
    colors = [
        (230, 80, 70),
        (75, 160, 255),
        (80, 190, 120),
        (240, 180, 65),
        (180, 120, 240),
        (70, 205, 210),
        (245, 120, 175),
        (160, 170, 80),
    ]
    return colors[idx % len(colors)]


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


def read_seg(path: Path) -> SegData:
    lines = path.read_text(encoding="utf-8").splitlines()
    vertex_count = None
    shape_count = None
    header_end = None

    for idx, line in enumerate(lines):
        if line.startswith("element vertex "):
            vertex_count = int(line.split()[-1])
        elif line.startswith("element shapes "):
            shape_count = int(line.split()[-1])
        elif line == "end_header":
            header_end = idx
            break

    if vertex_count is None or shape_count is None or header_end is None:
        raise ValueError(f"{path}: not a UniCo SEG file")

    cursor = header_end + 1
    vertex_rows = []
    for line in lines[cursor : cursor + vertex_count]:
        fields = line.split()
        if len(fields) != 7:
            raise ValueError(f"{path}: expected 7 vertex fields, got {len(fields)} in {line!r}")
        vertex_rows.append([float(x) for x in fields[:6]] + [int(fields[6])])
    cursor += vertex_count

    vertex_array = np.asarray(vertex_rows, dtype=np.float64)
    normals = vertex_array[:, 0:3].astype(np.float64)
    xyz = vertex_array[:, 3:6].astype(np.float64)
    segments = vertex_array[:, 6].astype(np.int64)

    shapes: list[Shape] = []
    for line in lines[cursor : cursor + shape_count]:
        fields = line.split()
        if len(fields) != 12:
            raise ValueError(f"{path}: expected 12 shape fields, got {len(fields)} in {line!r}")
        values = [float(x) for x in fields]
        color = tuple(int(max(0, min(255, round(c)))) for c in values[6:9])
        shapes.append(
            Shape(
                pos=np.asarray(values[0:3], dtype=np.float64),
                direction=_normalize(np.asarray(values[3:6], dtype=np.float64)),
                color=color,
                r1=float(values[9]),
                r2=float(values[10]),
                type_id=int(round(values[11])),
            )
        )

    return SegData(path=path, xyz=xyz, normals=normals, segments=segments, shapes=shapes)


def _segment_color(seg_id: int, shapes: list[Shape]) -> tuple[int, int, int]:
    if 0 <= seg_id < len(shapes):
        return shapes[seg_id].color
    return _palette(seg_id)


def write_points_ply(data: SegData, out_path: Path) -> None:
    colors = np.asarray([_segment_color(int(seg), data.shapes) for seg in data.segments], dtype=np.uint8)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment UniCo visual export: predicted points colored by primitive segment\n")
        f.write(f"element vertex {data.xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float nx\n")
        f.write("property float ny\n")
        f.write("property float nz\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property int segment\n")
        f.write("end_header\n")
        for point, normal, color, seg in zip(data.xyz, data.normals, colors, data.segments):
            f.write(
                f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{normal[0]:.9g} {normal[1]:.9g} {normal[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} {int(seg)}\n"
            )


def _bbox_diag(points: np.ndarray) -> float:
    if points.size == 0:
        return 1.0
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))) or 1.0


def _points_for_shape(data: SegData, shape_id: int) -> np.ndarray:
    return data.xyz[data.segments == shape_id]


def _plane_part(data: SegData, shape_id: int, shape: Shape) -> MeshPart:
    points = _points_for_shape(data, shape_id)
    normal = shape.direction
    u, v, _ = _basis_from_axis(normal)
    origin = points.mean(axis=0) if len(points) else shape.pos

    if len(points) >= 3:
        rel = points - origin
        pu = rel @ u
        pv = rel @ v
        min_u, max_u = float(pu.min()), float(pu.max())
        min_v, max_v = float(pv.min()), float(pv.max())
        pad_u = max((max_u - min_u) * 0.08, _bbox_diag(data.xyz) * 0.01)
        pad_v = max((max_v - min_v) * 0.08, _bbox_diag(data.xyz) * 0.01)
        min_u -= pad_u
        max_u += pad_u
        min_v -= pad_v
        max_v += pad_v
    else:
        size = _bbox_diag(data.xyz) * 0.1
        min_u = min_v = -size
        max_u = max_v = size

    vertices = [
        origin + min_u * u + min_v * v,
        origin + max_u * u + min_v * v,
        origin + max_u * u + max_v * v,
        origin + min_u * u + max_v * v,
    ]
    return MeshPart(
        name=f"shape_{shape_id:03d}_plane",
        material=f"seg_{shape_id:03d}",
        color=shape.color,
        vertices=vertices,
        faces=[(0, 1, 2, 3)],
    )


def _sphere_part(data: SegData, shape_id: int, shape: Shape, segments: int) -> MeshPart:
    points = _points_for_shape(data, shape_id)
    radius = abs(shape.r1)
    if radius < 1e-6 and len(points):
        radius = float(np.mean(np.linalg.norm(points - shape.pos, axis=1)))
    radius = max(radius, _bbox_diag(data.xyz) * 0.01)

    lat_steps = max(6, segments // 2)
    lon_steps = max(12, segments)
    vertices = []
    faces = []
    for i in range(lat_steps + 1):
        theta = math.pi * i / lat_steps
        z = math.cos(theta)
        ring_radius = math.sin(theta)
        for j in range(lon_steps):
            phi = 2.0 * math.pi * j / lon_steps
            vertices.append(
                shape.pos
                + radius
                * np.array(
                    [
                        ring_radius * math.cos(phi),
                        ring_radius * math.sin(phi),
                        z,
                    ],
                    dtype=np.float64,
                )
            )

    for i in range(lat_steps):
        for j in range(lon_steps):
            a = i * lon_steps + j
            b = i * lon_steps + (j + 1) % lon_steps
            c = (i + 1) * lon_steps + (j + 1) % lon_steps
            d = (i + 1) * lon_steps + j
            if i == 0:
                faces.append((a, c, d))
            elif i == lat_steps - 1:
                faces.append((a, b, d))
            else:
                faces.append((a, b, c, d))

    return MeshPart(
        name=f"shape_{shape_id:03d}_sphere",
        material=f"seg_{shape_id:03d}",
        color=shape.color,
        vertices=vertices,
        faces=faces,
    )


def _axis_extents(points: np.ndarray, origin: np.ndarray, axis: np.ndarray, fallback_length: float) -> tuple[float, float]:
    if len(points):
        t = (points - origin) @ axis
        t_min = float(t.min())
        t_max = float(t.max())
        if t_max - t_min > 1e-6:
            pad = 0.05 * (t_max - t_min)
            return t_min - pad, t_max + pad
    half = max(fallback_length * 0.5, 1e-3)
    return -half, half


def _cylinder_part(data: SegData, shape_id: int, shape: Shape, segments: int) -> MeshPart:
    points = _points_for_shape(data, shape_id)
    axis = shape.direction
    u, v, _ = _basis_from_axis(axis)
    radius = abs(shape.r1)
    if radius < 1e-6 and len(points):
        rel = points - shape.pos
        lateral = rel - np.outer(rel @ axis, axis)
        radius = float(np.mean(np.linalg.norm(lateral, axis=1)))
    radius = max(radius, _bbox_diag(data.xyz) * 0.006)
    t_min, t_max = _axis_extents(points, shape.pos, axis, fallback_length=max(radius * 2.0, _bbox_diag(data.xyz) * 0.1))

    ring_steps = max(12, segments)
    vertices = []
    for t in (t_min, t_max):
        center = shape.pos + t * axis
        for j in range(ring_steps):
            phi = 2.0 * math.pi * j / ring_steps
            vertices.append(center + radius * (math.cos(phi) * u + math.sin(phi) * v))

    bottom_center = len(vertices)
    vertices.append(shape.pos + t_min * axis)
    top_center = len(vertices)
    vertices.append(shape.pos + t_max * axis)

    faces = []
    for j in range(ring_steps):
        a = j
        b = (j + 1) % ring_steps
        c = ring_steps + (j + 1) % ring_steps
        d = ring_steps + j
        faces.append((a, b, c, d))
        faces.append((bottom_center, b, a))
        faces.append((top_center, d, c))

    return MeshPart(
        name=f"shape_{shape_id:03d}_cylinder",
        material=f"seg_{shape_id:03d}",
        color=shape.color,
        vertices=vertices,
        faces=faces,
    )


def _cone_part(data: SegData, shape_id: int, shape: Shape, segments: int) -> MeshPart:
    points = _points_for_shape(data, shape_id)
    axis = shape.direction
    if len(points):
        t_probe = (points - shape.pos) @ axis
        if abs(float(t_probe.min())) > abs(float(t_probe.max())):
            axis = -axis
            t_probe = -t_probe
        height = max(float(t_probe.max()), _bbox_diag(data.xyz) * 0.05)
    else:
        height = _bbox_diag(data.xyz) * 0.2

    theta = abs(shape.r1)
    radius = height * math.tan(theta) if 1e-6 < theta < math.pi / 2 - 1e-6 else 0.0
    if (radius < 1e-6 or not math.isfinite(radius)) and len(points):
        rel = points - shape.pos
        lateral = rel - np.outer(rel @ axis, axis)
        radius = float(np.max(np.linalg.norm(lateral, axis=1)))
    radius = max(radius, _bbox_diag(data.xyz) * 0.006)

    u, v, _ = _basis_from_axis(axis)
    ring_steps = max(12, segments)
    apex = shape.pos
    base_center = shape.pos + height * axis
    vertices = [apex]
    for j in range(ring_steps):
        phi = 2.0 * math.pi * j / ring_steps
        vertices.append(base_center + radius * (math.cos(phi) * u + math.sin(phi) * v))
    center_index = len(vertices)
    vertices.append(base_center)

    faces = []
    for j in range(ring_steps):
        a = 1 + j
        b = 1 + (j + 1) % ring_steps
        faces.append((0, a, b))
        faces.append((center_index, b, a))

    return MeshPart(
        name=f"shape_{shape_id:03d}_cone",
        material=f"seg_{shape_id:03d}",
        color=shape.color,
        vertices=vertices,
        faces=faces,
    )


def build_primitive_parts(data: SegData, segments: int) -> list[MeshPart]:
    parts = []
    for shape_id, shape in enumerate(data.shapes):
        if shape.type_id == 0:
            parts.append(_plane_part(data, shape_id, shape))
        elif shape.type_id == 1:
            parts.append(_sphere_part(data, shape_id, shape, segments))
        elif shape.type_id == 2:
            parts.append(_cylinder_part(data, shape_id, shape, segments))
        elif shape.type_id == 3:
            parts.append(_cone_part(data, shape_id, shape, segments))
    return parts


def _sample_point_indices(data: SegData, sample_count: int) -> np.ndarray:
    if sample_count > 0 and data.xyz.shape[0] > sample_count:
        return np.linspace(0, data.xyz.shape[0] - 1, sample_count, dtype=np.int64)
    return np.arange(data.xyz.shape[0], dtype=np.int64)


def _write_mtl(materials: dict[str, tuple[int, int, int]], mtl_path: Path) -> None:
    with mtl_path.open("w", encoding="utf-8") as f:
        f.write("# UniCo visual export materials\n")
        for material, color in materials.items():
            r, g, b = [c / 255.0 for c in color]
            f.write(f"newmtl {material}\n")
            f.write(f"Kd {r:.6f} {g:.6f} {b:.6f}\n")
            f.write("Ka 0.050000 0.050000 0.050000\n")
            f.write("Ks 0.120000 0.120000 0.120000\n")
            f.write("Ns 16\n")
            f.write("illum 2\n\n")


def _write_mesh_parts(f, parts: list[MeshPart], vertex_offset: int) -> int:
    for part in parts:
        r, g, b = [c / 255.0 for c in part.color]
        f.write(f"\no {_safe_name(part.name)}\n")
        f.write(f"usemtl {part.material}\n")
        for vertex in part.vertices:
            f.write(
                f"v {vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g} "
                f"{r:.6f} {g:.6f} {b:.6f}\n"
            )
        for face in part.faces:
            f.write("f " + " ".join(str(vertex_offset + idx) for idx in face) + "\n")
        vertex_offset += len(part.vertices)
    return vertex_offset


def write_obj(parts: list[MeshPart], out_path: Path) -> None:
    mtl_path = out_path.with_suffix(".mtl")
    materials: dict[str, tuple[int, int, int]] = {}
    for part in parts:
        materials.setdefault(part.material, part.color)
    _write_mtl(materials, mtl_path)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# UniCo visual export\n")
        f.write(f"mtllib {mtl_path.name}\n")
        _write_mesh_parts(f, parts, vertex_offset=1)


def _default_point_radius(data: SegData) -> float:
    return max(_bbox_diag(data.xyz) * 0.0015, 1e-6)


def _add_point_glyph(
    vertices: list[tuple[np.ndarray, tuple[int, int, int]]],
    faces: list[tuple[int, ...]],
    center: np.ndarray,
    color: tuple[int, int, int],
    radius: float,
) -> None:
    offset = len(vertices)
    directions = (
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([-1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, -1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
        np.array([0.0, 0.0, -1.0], dtype=np.float64),
    )
    vertices.extend((center + radius * direction, color) for direction in directions)
    glyph_faces = (
        (0, 2, 4),
        (2, 1, 4),
        (1, 3, 4),
        (3, 0, 4),
        (2, 0, 5),
        (1, 2, 5),
        (3, 1, 5),
        (0, 3, 5),
    )
    faces.extend(tuple(offset + idx for idx in face) for face in glyph_faces)


def write_overlay_ply(
    data: SegData,
    primitive_parts: list[MeshPart],
    out_path: Path,
    point_sample: int,
    point_radius: float | None,
    point_offset_scale: float,
) -> None:
    point_indices = _sample_point_indices(data, sample_count=point_sample)
    radius = _default_point_radius(data) if point_radius is None else max(float(point_radius), 1e-9)
    normal_offset = max(float(point_offset_scale), 0.0) * radius

    vertices: list[tuple[np.ndarray, tuple[int, int, int]]] = []
    faces: list[tuple[int, ...]] = []
    for part in primitive_parts:
        face_offset = len(vertices)
        vertices.extend((vertex, part.color) for vertex in part.vertices)
        faces.extend(tuple(face_offset + idx for idx in face) for face in part.faces)

    for point_index in point_indices:
        seg_id = int(data.segments[point_index])
        normal = _normalize(data.normals[point_index], fallback=(0.0, 0.0, 0.0))
        center = data.xyz[point_index] + normal_offset * normal
        _add_point_glyph(
            vertices=vertices,
            faces=faces,
            center=center,
            color=_segment_color(seg_id, data.shapes),
            radius=radius,
        )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment UniCo visual export: primitive mesh faces plus predicted point glyph meshes\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for vertex, color in vertices:
            f.write(
                f"{vertex[0]:.9g} {vertex[1]:.9g} {vertex[2]:.9g} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )
        for face in faces:
            f.write(f"{len(face)} " + " ".join(str(idx) for idx in face) + "\n")


def write_overlay_obj(data: SegData, primitive_parts: list[MeshPart], out_path: Path, point_sample: int) -> None:
    point_indices = _sample_point_indices(data, sample_count=point_sample)
    mtl_path = out_path.with_suffix(".mtl")
    materials: dict[str, tuple[int, int, int]] = {}
    for part in primitive_parts:
        materials.setdefault(part.material, part.color)
    for seg_id in sorted({int(data.segments[i]) for i in point_indices}):
        materials.setdefault(f"points_seg_{seg_id:03d}", _segment_color(seg_id, data.shapes))
    _write_mtl(materials, mtl_path)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# UniCo visual export: primitive meshes plus OBJ point primitives\n")
        f.write("# Point vertices use the OBJ vertex-color extension: v x y z r g b\n")
        f.write(f"mtllib {mtl_path.name}\n")
        vertex_offset = _write_mesh_parts(f, primitive_parts, vertex_offset=1)

        f.write("\no predicted_points\n")
        for seg_id in sorted({int(data.segments[i]) for i in point_indices}):
            color = _segment_color(seg_id, data.shapes)
            r, g, b = [c / 255.0 for c in color]
            f.write(f"g points_segment_{seg_id:03d}\n")
            f.write(f"usemtl points_seg_{seg_id:03d}\n")
            for point_index in point_indices[data.segments[point_indices] == seg_id]:
                point = data.xyz[point_index]
                f.write(f"v {point[0]:.9g} {point[1]:.9g} {point[2]:.9g} {r:.6f} {g:.6f} {b:.6f}\n")
                f.write(f"p {vertex_offset}\n")
                vertex_offset += 1


def write_summary(data: SegData, out_path: Path, generated: dict[str, str]) -> None:
    shapes = []
    for shape_id, shape in enumerate(data.shapes):
        point_count = int(np.sum(data.segments == shape_id))
        shapes.append(
            {
                "id": shape_id,
                "type": shape.type_id,
                "type_name": TYPE_NAMES.get(shape.type_id, f"unknown_{shape.type_id}"),
                "point_count": point_count,
                "color": list(shape.color),
                "pos": shape.pos.tolist(),
                "direction": shape.direction.tolist(),
                "r1": shape.r1,
                "r2": shape.r2,
            }
        )

    payload = {
        "source": str(data.path),
        "point_count": int(data.xyz.shape[0]),
        "primitive_count": len(data.shapes),
        "types": {
            TYPE_NAMES.get(type_id, f"unknown_{type_id}"): sum(1 for shape in data.shapes if shape.type_id == type_id)
            for type_id in sorted({shape.type_id for shape in data.shapes})
        },
        "generated": generated,
        "shapes": shapes,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_one(
    path: Path,
    out_dir: Path,
    segments: int,
    point_sample: int,
    point_radius: float | None,
    point_offset_scale: float,
) -> dict[str, Path]:
    data = read_seg(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem

    points_ply = out_dir / f"{stem}_points.ply"
    primitives_obj = out_dir / f"{stem}_primitives.obj"
    overlay_ply = out_dir / f"{stem}_overlay.ply"
    summary_json = out_dir / f"{stem}_summary.json"

    primitive_parts = build_primitive_parts(data, segments=segments)
    for stale_path in (
        out_dir / f"{stem}_overlay.obj",
        out_dir / f"{stem}_overlay.mtl",
        out_dir / f"{stem}_points.obj",
        out_dir / f"{stem}_points.mtl",
    ):
        stale_path.unlink(missing_ok=True)

    write_points_ply(data, points_ply)
    write_obj(primitive_parts, primitives_obj)
    write_overlay_ply(
        data,
        primitive_parts,
        overlay_ply,
        point_sample=point_sample,
        point_radius=point_radius,
        point_offset_scale=point_offset_scale,
    )
    generated = {
        "points_ply": str(points_ply),
        "primitives_obj": str(primitives_obj),
        "overlay_ply": str(overlay_ply),
    }
    write_summary(data, summary_json, generated)

    return {
        "points_ply": points_ply,
        "primitives_obj": primitives_obj,
        "overlay_ply": overlay_ply,
        "summary_json": summary_json,
    }


def iter_inputs(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(input_path.glob("*.seg"))
    return [input_path]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export UniCo .seg predictions to standard PLY/OBJ visualizations.")
    parser.add_argument("input", type=Path, help="A .seg file or a directory containing .seg files.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <input-dir>/visual.",
    )
    parser.add_argument("--segments", type=int, default=32, help="Mesh radial resolution for spheres/cylinders/cones.")
    parser.add_argument(
        "--point-sample",
        type=int,
        default=0,
        help="Number of predicted points to include in overlay PLY output. Use 0 for all points.",
    )
    parser.add_argument(
        "--point-radius",
        type=float,
        default=None,
        help="Overlay point glyph radius. Defaults to 0.15%% of the prediction bounding-box diagonal.",
    )
    parser.add_argument(
        "--point-offset-scale",
        type=float,
        default=2.0,
        help="Offset overlay point glyphs along their normals by this many radii.",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    inputs = iter_inputs(input_path)
    if not inputs:
        raise SystemExit(f"No .seg files found in {input_path}")

    if args.out_dir is None:
        base = input_path if input_path.is_dir() else input_path.parent
        out_dir = base / "visual"
    else:
        out_dir = args.out_dir.resolve()

    for seg_path in inputs:
        outputs = export_one(
            seg_path,
            out_dir=out_dir,
            segments=max(8, int(args.segments)),
            point_sample=max(0, int(args.point_sample)),
            point_radius=args.point_radius,
            point_offset_scale=max(0.0, float(args.point_offset_scale)),
        )
        print(f"{seg_path} ->")
        for label, output_path in outputs.items():
            print(f"  {label}: {output_path}")


if __name__ == "__main__":
    main()
