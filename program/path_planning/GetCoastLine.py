import os
import xml.etree.ElementTree as ET
import csv

DIR = os.path.dirname(__file__)
RAW_DATAS = f"{DIR}/../../raw_datas"

NUM = [f"{i:02d}" for i in range(1, 48)]
for num in NUM:
    gml_path = os.path.abspath(f"{RAW_DATAS}/国土交通省/C23-06_{num}_GML/C23-06_{num}-g.xml")
    if not os.path.exists(gml_path):
        continue
    SAVE_DIR = os.path.dirname(gml_path)

    tree = ET.parse(gml_path)
    root = tree.getroot()

    # 名前空間の指定（gml: ～ を探すため）
    ns = {"gml": "http://www.opengis.net/gml/3.2"}

    rows = []

    # すべての gml:Curve ごとに
    for curve in root.findall("gml:Curve", ns):
        curve_id = curve.attrib.get("{http://www.opengis.net/gml/3.2}id")

        # その中の gml:posList を探す
        poslist_elem = curve.find(".//gml:posList", ns)
        if poslist_elem is None:
            continue

        # 空白区切りで数値を取り出す
        nums = poslist_elem.text.split()

        # 2つずつ（lat, lon）にまとめる
        for i in range(0, len(nums), 2):
            lat = float(nums[i])
            lon = float(nums[i + 1])
            rows.append([curve_id, lat, lon])

    # CSV に書き出し
    with open(f"{SAVE_DIR}/C23-06_{num}-g.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["curve_id", "lat", "lon"])
        writer.writerows(rows)

    print(f"{num}   書き出し完了")
