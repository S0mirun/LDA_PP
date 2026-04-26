import os
import re

import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.ops import unary_union, polygonize, polygonize_full, snap
from tqdm import tqdm

JAPAN = True
BUOY = True
AIS = True

GRID = False
ZOOM = True


DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]


def dmm_to_decimal(s):
    s = s.strip().replace("’", "'").replace("′", "'")
    m = re.match(r"(\d+)°(\d+(?:\.\d+)?)'([NSEW])", s)
    if not m:
        raise ValueError(f"invalid format: {s}")

    deg = float(m.group(1))
    minute = float(m.group(2))
    direction = m.group(3)

    value = deg + minute / 60

    if direction in ["S", "W"]:
        value *= -1

    return value


def draw_Japan_poly(fig, ax):
    lines = _make_lines()
    polygons = _make_land_polygon(lines)
    _draw_polygons(fig, ax, polygons)
    _save_fig(fig, ax, "Japan")
    

def _make_lines():
    csv_path = glob.glob("raw_datas/国土交通省/*/*.csv")

    lines = []
    for path in tqdm(csv_path):
        df = pd.read_csv(path, usecols=["curve_id", "lat", "lon"])

        for curve_id, g in df.groupby("curve_id"):
            pts = list(zip(g["lon"], g["lat"]))

            cleaned = [pts[0]]
            for p in pts[1:]:
                if p != cleaned[-1]:
                    cleaned.append(p)

            if len(cleaned) >= 2:
                try:
                    lines.append(LineString(cleaned))
                except Exception:
                    pass
    
    print(f"\nlines: {len(lines)}")
    return lines


def _make_land_polygon(lines):
    print("\nmerging lines...")
    merged = unary_union(lines)
    check_polygon(merged, fig, ax)

    print("\nsnapping...")
    # snapped = _snap_lines(lines)
    # snapped = snap(merged, merged, 1e-4)
    # check_polygon(snapped)

    print("\npolygonizing...")
    polygons = list(polygonize(merged))
    # polygons = list(polygonize(snapped))

    print(f"\npolygons: {len(polygons)}")
    return polygons


def check_polygon(polygon, fig, ax):
    polys, cuts, dangles, invalids = polygonize_full(polygon)

    gdf_poly = gpd.GeoDataFrame(geometry=list(polys.geoms), crs="EPSG:4326")
    gdf_dng = gpd.GeoDataFrame(geometry=list(dangles.geoms), crs="EPSG:4326")

    gdf_poly.plot(ax=ax, color="lightgray", edgecolor="none")
    gdf_dng.plot(ax=ax, color="red", linewidth=1.0)

    print("polygons :", len(list(polys.geoms)))
    print("cuts     :", len(list(cuts.geoms)))
    print("dangles  :", len(list(dangles.geoms)))
    print("invalids :", len(list(invalids.geoms)))


def _snap_lines(lines):
    endpoint_records = []
    endpoint_xy = []
    for i, line in enumerate(lines):
        coords = list(line.coords)

        p0 = coords[0]
        p1 = coords[-1]

        endpoint_records.append((i, 0))
        endpoint_xy.append(p0)

        endpoint_records.append((i, -1))
        endpoint_xy.append(p1)

    endpoint_xy = np.array(endpoint_xy, dtype=float)

    tree = cKDTree(endpoint_xy)
    pairs = tree.query_pairs(r=1e-8)

    replacement = {}
    for a, b in pairs:
        xa, ya = endpoint_xy[a]
        xb, yb = endpoint_xy[b]

        xm = (xa + xb) / 2
        ym = (ya + yb) / 2

        replacement[a] = (xm, ym)
        replacement[b] = (xm, ym)

    snapped = []
    for i, line in enumerate(lines):
        coords = list(line.coords)
        if len(coords) < 2:
            snapped.append(line)
            continue

        # この line の始点・終点に対応する endpoint index を探す
        start_idx = 2 * i
        end_idx = 2 * i + 1

        if start_idx in replacement:
            coords[0] = replacement[start_idx]

        if end_idx in replacement:
            coords[-1] = replacement[end_idx]

        try:
            snapped.append(LineString(coords))
        except Exception:
            snapped.append(line)

    return snapped


def _draw_polygons(fig, ax, polygons):
    gdf = gpd.GeoDataFrame(geometry=polygons, crs="EPSG:4326")

    gdf.plot(ax=ax, color="lightgray", edgecolor="black", linewidth=0.2)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Japan Land Area")

    plt.tight_layout()


def _save_fig(fig, ax, name):
    save_dir = f"outputs/{dirname}"
    os.makedirs(save_dir, exist_ok=True)

    ax.set_aspect("equal", adjustable="box")
    fig.savefig(os.path.join(save_dir, f"{name}.png"), 
                dpi=400, bbox_inches="tight", pad_inches=0.05)
    print(f"\n{name} saved")


def draw_Buoy(fig, ax):
    csv_paths = glob.glob("outputs/data/buoy/*.csv")

    color_map = {"1": "white", "2": "black", "3": "red", 
                 "4": "green", "5": "blue", "6": "yellow"}

    for path in csv_paths:
        df = pd.read_csv(path)
        df = df.dropna(subset=["longitude", "latitude"]).copy()

        if "COLOUR" not in df.columns:
            continue

        gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")

        xs = gdf.geometry.x
        ys = gdf.geometry.y

        ax.scatter( xs, ys, s=20, c="orange",
                   edgecolors="black", linewidths=0.3, zorder=2)

        gdf["COLOUR"] = gdf["COLOUR"].astype(str).str.strip()
        for colour, g in gdf.groupby("COLOUR"):
            ax.scatter(g.geometry.x, g.geometry.y, s=10, c=color_map.get(colour), 
                       edgecolors="black", linewidths=0.2, zorder=3)
            
    _save_fig(fig, ax, "Buoy")


def draw_AIS(fig, ax):
    print("\nAIS reading...")
    csv_path = glob.glob("raw_datas/【秘密情報】航海記録/*/*.csv")
    lines = []
    source_paths = []
    source_dates = []

    for path in tqdm(csv_path):
        df = pd.read_csv(path, encoding="cp932")
        df.columns = df.columns.str.strip()

        df = df.dropna(subset=["日付", "時刻", "緯度", "経度"]).copy()
        df["date_dt"] = pd.to_datetime(df["日付"], format="%Y/%m/%d", errors="coerce")
        df["time_td"] = pd.to_timedelta(df["時刻"])
        df["latitude"] = df["緯度"].apply(dmm_to_decimal)
        df["longitude"] = df["経度"].apply(dmm_to_decimal)

        date_vals = df["date_dt"].dt.normalize().unique()
        date_val = pd.Timestamp(date_vals[0])

        df = df.sort_values(["time_td"])
        coords = list(zip(df["longitude"], df["latitude"]))

        cleaned = [coords[0]]
        for p in coords[1:]:
            if p != cleaned[-1]:
                cleaned.append(p)

        lines.append(LineString(cleaned))
        source_paths.append(path)
        source_dates.append(date_val)

    gdf_tracks = gpd.GeoDataFrame({"path": source_paths, "date": source_dates}, geometry=lines, crs="EPSG:4326")
    gdf_tracks.plot(ax=ax, color="black", linewidth=1.0, alpha=0.5, zorder=10)

    _save_fig(fig, ax, "AIS")


def save_grid_fig(fig, ax, xmin=127.5, xmax=146.5, ymin=27.0, ymax=46.0, step=2.5):
    print("grid fig saving...")
    xs = np.arange(xmin, xmax, step)
    ys = np.arange(ymin, ymax, step)

    save_dir = f"outputs/{dirname}/tile_{step}"
    os.makedirs(save_dir, exist_ok=True)
    count = 0
    for y0 in tqdm(ys):
        for x0 in xs:
            x1 = x0 + step
            y1 = y0 + step

            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal", adjustable="box")

            filename = f"tile_lat_{y0:.2f}_{y1:.2f}_lon_{x0:.2f}_{x1:.2f}.png"
            fig.savefig(
                os.path.join(save_dir, filename),
                dpi=400,
                bbox_inches="tight",
                pad_inches=0.05,
            )
            count += 1

    print("\ntiles saved")


def save_zoom_fig(fig, ax):
    port_path = glob.glob("raw_datas/tmp/coordinates_of_port/*.csv")
    margin = 0.02

    for path in port_path:
        df = pd.read_csv(path)

        if "Longitude" not in df.columns:
            continue
        if "Latitude" not in df.columns:
            continue

        lon = df["Longitude"].iloc[0]
        lat = df["Latitude"].iloc[0]

        ax.set_xlim(lon-margin, lon+margin)
        ax.set_ylim(lat-margin, lat+margin)
        

        port_name = os.path.splitext(os.path.basename(path))[0]
        save_dir = f"outputs/{dirname}/port"
        os.makedirs(save_dir, exist_ok=True)

        ax.set_aspect("equal", adjustable="box")
        fig.savefig(os.path.join(save_dir, f"{port_name}.png"), 
                    dpi=400, bbox_inches="tight", pad_inches=0.05)
        print(f"{port_name} saved")


if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(10, 10))

    if JAPAN:
        draw_Japan_poly(fig, ax)

    if BUOY:
        draw_Buoy(fig, ax)

    if AIS:
        draw_AIS(fig, ax)

    if GRID:
        save_grid_fig(fig, ax, step=1.5)

    if ZOOM:
        save_zoom_fig(fig, ax)