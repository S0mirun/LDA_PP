import os
import re

import glob
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

from shapely.geometry import LineString
from shapely.ops import unary_union, polygonize, polygonize_full, snap
from tqdm import tqdm

JAPAN = True
BUOY = True
AIS = True

ZOOM = False


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
    check_polygon(merged)

    print("\nsnapping...")
    snapped = snap(merged, merged, 1e-4)
    check_polygon(snapped)

    print("\npolygonizing...")
    # polygons = list(polygonize(merged))
    polygons = list(polygonize(snapped))

    print(f"\npolygons: {len(polygons)}")
    return polygons


def check_polygon(polygon):
    polys, cuts, dangles, invalids = polygonize_full(polygon)

    print("polygons :", len(list(polys.geoms)))
    print("cuts     :", len(list(cuts.geoms)))
    print("dangles  :", len(list(dangles.geoms)))
    print("invalids :", len(list(invalids.geoms)))


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

    color_map = {"0": "white", "1": "black", "2": "red", 
                 "3": "green", "4": "blue", "5": "yellow"}

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

        df["COLOUR"] = df["COLOUR"].astype(str).str.strip()
        for colour, g in gdf.groupby("COLOUR"):
            ax.scatter(g.geometry.x, g.geometry.y, s=10, c=color_map.get(colour), 
                       edgecolors="black", linewidths=0.2, zorder=3)
            
    _save_fig(fig, ax, "Buoy")


def draw_AIS(fig, ax):
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


def save_zoom_fig(fig, ax):
    pass


if __name__ == '__main__':
    fig, ax = plt.subplots(figsize=(10, 10))

    if JAPAN:
        draw_Japan_poly(fig, ax)

    if BUOY:
        draw_Buoy(fig, ax)

    if AIS:
        draw_AIS(fig, ax)

    if ZOOM:
        save_zoom_fig(fig, ax)