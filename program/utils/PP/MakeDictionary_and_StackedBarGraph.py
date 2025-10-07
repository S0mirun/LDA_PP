"""
分類された要素の出現割合を最適化に用いるコストに使える形の辞書に格納する。
9.5kts以上のものはまとめて9.5kts以上で括るように変更した(11/10)
CMAで60度以上の分類を設けるためにangleに60も追加した(11/10)
CMAファイルとの整合性を加味しkeyは全て"数字"として扱う
縦長のヒストグラムの連続は見づらいので、積み重ね棒グラフとした（2025/1/24)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


DIR = os.path.dirname(__file__)
dirname = os.path.splitext(os.path.basename(__file__))[0]
SAVE_DIR = f"{DIR}/../../../outputs/{dirname}"
os.makedirs(SAVE_DIR, exist_ok=True)

# 積み上げ棒グラフを作成する
figure_plot_switch = True
#
df = pd.read_csv(f"{DIR}/../../../outputs/ClassifyElements/all_ports_classified_elements_fixed.csv")

# 任意の要素のみを含む新しいデータフレームを作成
selected_elements = [1, 3, 5]####
filtered_df = df[df['element'].isin(selected_elements)]

# 区切り幅と速度の範囲を設定(kts)(20240820にm/sからktsに変更)
interval = 1.0
speed_min = 1.5
speed_max = 9.5
# 速度域毎に分けるための数字を作成
speed_bins = np.arange(speed_min, speed_max + interval, interval)

#角度の下限上限インターバルを設定[deg]
angle_min = 5
angle_max = 60
angle_interval = 5
#角度リスト
angle_bins = np.arange(angle_min, angle_max, angle_interval)

# 速度域毎にリストを格納する辞書の初期化
dict_by_speed = {}

for bin_start in speed_bins:
    # キーを生成。ここは9.5も含めて作る
    #key = f'{round(bin_start, 1)}-{round(bin_start + interval, 1)} kts'
    key = bin_start
    
    # 辞書にキーと空のリストを追加
    dict_by_speed[key] = []

# データ一つ一つを速度域毎に分類して辞書に格納。
for index, row in filtered_df.iterrows():
    speed = row['knot']
    diff_psi = row['diff_psi_raw']
    element = row['element']
    
    # 該当する速度域のキーを見つける。まず、8.5~9.5まで分類して、そこで分類されなかったものを9.5~にまとめる
    for bin_start in speed_bins[:-1]:
        if bin_start <= speed < bin_start + interval:
            #
            #key = f'{round(bin_start, 1)}-{round(bin_start + interval, 1)} kts'
            key = bin_start
            dict_by_speed[key].append(row)
            break
        else:
            key = speed_max
            dict_by_speed[key].append(row)
            
# フィルタリングされた辞書を格納する新しい辞書を作成
filtered_dict = {}

# 長さが(選択された)総データ数の1%未満の辞書をフィルタリングし削除する
df_length = len(filtered_df)
threshold = round(df_length / 100)
#print(f'threshold : {threshold}')
for key, rows in dict_by_speed.items():
    if len(rows) >= threshold:
        filtered_dict[key] = rows

# ネストされた辞書を作成する
new_filtered_dict = {}
for key in filtered_dict.keys():
    for row in filtered_dict[key]:
        if row['element'] == 1 or row['element'] == 3:
            new_key = 0
            if key not in new_filtered_dict:
                new_filtered_dict[key] = {}
            if new_key not in new_filtered_dict:
                new_filtered_dict[key][new_key] = []
            #new_filtered_dict[key][new_key].append(row)
        else:
            for angle in angle_bins:
                if angle <= abs(row['diff_psi_raw']) < angle + angle_interval:
                    new_key = angle
                    if key not in new_filtered_dict:
                        new_filtered_dict[key] = {}
                    if new_key not in new_filtered_dict:
                        new_filtered_dict[key][new_key] = []
                    #new_filtered_dict[key][new_key].append(row)
                    break # 該当する角度範囲が見つかったらループを抜ける
        # CMAのために60度以上の分類を作っておく        
        new_filtered_dict[key][60] = 0
                
# 作成された辞書の内容を確認
# print(new_filtered_dict)

# 正規化を行う
normalization_standard = 100
#filtered_dictのキー（速度域）についてそれぞれのキーで中身が100になるように正規化する
for key, rows in filtered_dict.items():
    normalization_coefficient = len(rows) / normalization_standard
    
    angle_counts = {}
    # angle_countsの初期化（キー0も含める）
    angle_counts = {0: 0}  # まずキー0を初期化
    for angle in angle_bins:
        angle_counts[angle] = 0  # 各angle_binsの角度も初期化
    
    # 各角度域ごとにカウントを正規化係数で割って整数部分と小数部分を記録
    normalized_numbers = []
    total = 0
    for row in rows:
        if row['element'] == 1 or row['element'] == 3:
            angle_counts[0] += 1
        else:
            for angle in angle_bins:
                if angle <= abs(row['diff_psi_raw']) < angle + angle_interval:
                    angle_counts[angle] += 1
                    break # 該当する角度範囲が見つかったらループを抜ける
                
    for angle, count in angle_counts.items():
        normalized_value = count / normalization_coefficient
        integer_part = int(normalized_value)
        fractional_part = normalized_value - integer_part
        normalized_numbers.append((angle, integer_part, fractional_part))
        #まずは整数部分だけを考える
        total += integer_part
        
    #合計が100になるように小数部分が大きいものから+1していく
    normalized_numbers.sort(key=lambda x: x[2], reverse=True) # lambda式を用いてnormalized_numbersの3番目、つまり小数部分に対して降順にソートする
    while total < normalization_standard:
        angle, integer_part, fractional_part = normalized_numbers.pop(0) #小数部分が一番小さい物が0番目
        new_filtered_dict[key][angle] = integer_part + 1
        total += 1
        
    # popで抜かれなかった残りの値を辞書に設定
    for angle, integer_part, fractional_part in normalized_numbers:
        new_filtered_dict[key][angle] = integer_part
        
# 作成したnew_filtered_dictを表示する 
# print(new_filtered_dict)


if __name__ == '__main__':
    if figure_plot_switch:
        
        plt.rcParams['font.family'] = 'Times New Roman'  # フォントをTimes New Romanに設定
        
        # キー（速度域）のリストを取得
        speed_keys = list(new_filtered_dict.keys())

        # 速度域を1.5~2.5, 2.5~3.5...と表示
        # speed_labels = [
        #     f'{key:.1f}-{key + interval:.1f}' if key < speed_max else f'{key:.1f}-'
        #     for key in speed_keys
        # ]
        speed_labels = [
            '1.5-2.5', '2.5-3.5', '3.5-4.5', '4.5-5.5',
            '5.5-6.5', '6.5-7.5', '7.5-8.5', '8.5-9.5', '>=9.5'
        ]

        # 角度ごとのデータがある範囲をカウント
        angles_with_data = [angle for angle in range(0, 30, angle_interval) if angle in new_filtered_dict[speed_keys[0]]]
        angles_with_data.append(30)  # 30度以上を1つのカテゴリに統合

        # カラーマップを生成（角度ごとに異なる色を割り当て）
        cmap = plt.colormaps.get_cmap('tab20')

        # 積み上げ棒グラフ用のデータを作成
        bottom = np.zeros(len(speed_keys))

        # 図のサイズを 8:9 に調整# 図のサイズを調整（凡例のスペースを確保しつつバランスを取る）
        # 図のサイズを 8:12 に調整
        fig, ax = plt.subplots(figsize=(8, 12))  # (幅, 高さ) を 8:12 に

        # 積み上げ棒グラフ用のデータを作成
        bottom = np.zeros(len(speed_keys))

        # カラーマップを生成（角度ごとに異なる色を割り当て）
        cmap = plt.colormaps.get_cmap('tab20')

        # 凡例ラベルリスト（タイトルなし）
        # legend_labels = [
        #     "30 degrees or more" if angle == 30 else
        #     f"{angle}-{angle + angle_interval} degrees" if angle != 0 else "Course Keeping"
        #     for angle in angles_with_data
        # ]
        # legend_labels = [
        #     "30 degrees or more" if angle == 30 else
        #     f"{angle}-{angle + angle_interval} degrees" if angle != 0 else "Course Keeping"
        #     for angle in angles_with_data
        # ]
        # Course Keeping削除
        legend_labels = [
            "0-5 degrees" if angle == 0 else
            "30 degrees or more" if angle == 30 else
            f"{angle}-{angle + angle_interval} degrees"
            for angle in angles_with_data
        ]

        # 凡例の四角（タイトルなしでアイテムのみ）
        legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=cmap(idx)) for idx, _ in enumerate(angles_with_data)]

        for idx, angle in enumerate(angles_with_data):
            if angle == 30:
                counts = [sum(new_filtered_dict[key].get(a, 0) for a in range(30, angle_max + 1)) for key in speed_keys]
            else:
                counts = [new_filtered_dict[key].get(angle, 0) for key in speed_keys]

            bars = ax.bar(speed_keys, counts, bottom=bottom, label=legend_labels[idx], color=cmap(idx))
            bottom += np.array(counts)

        # 軸ラベルのフォントサイズを大きくする
        ax.set_xlabel('Speed Range [kts]', fontsize=16, labelpad=10)
        ax.set_ylabel('Percentage of Turning Angles', fontsize=16, labelpad=10)

        # 軸メモリのフォントサイズを大きくする
        ax.tick_params(axis='both', labelsize=14)

        # X軸の目盛りラベルを調整
        ax.set_xticks(speed_keys)
        ax.set_xticklabels(speed_labels, rotation=45, ha='right')

        # Y軸の最大値を100に固定（余白をなくす）
        ax.set_ylim(0, 100)

        # **凡例の列数を最大3列に制限**
        num_columns = min(3, len(legend_labels))  # 最大3列以内に制限
        num_rows = (len(legend_labels) + num_columns - 1) // num_columns  # 行数を計算

        # **凡例を作成（最大3列以内で調整）**
        legend = ax.legend(
            legend_patches, legend_labels,
            loc='upper center', bbox_to_anchor=(0.5, -0.25),  # 凡例の位置を微調整
            fontsize=16, ncol=num_columns, frameon=True  # 3列以内で調整
        )

        # **凡例と横軸ラベルの間を適切に開ける**
        plt.tight_layout(rect=[0, 0.3, 1, 1])  # 下の余白を適切に確保

        # 画像を保存
        output_file = f'output/elements/{today_str}/stacked_normalized_histogram_.png'
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
        print(f"Saved to '{output_file}'")
                
        # fig, ax = plt.subplots(figsize=(12, 6))
        # # fig, ax = plt.subplots()
        
        # for idx, angle in enumerate(angles_with_data):
        #     if angle == 30:
        #         # 30度以上の合計を計算
        #         counts = [sum(new_filtered_dict[key].get(a, 0) for a in range(30, angle_max + 1)) for key in speed_keys]
        #         label = '30 degrees or more'
        #     else:
        #         counts = [new_filtered_dict[key].get(angle, 0) for key in speed_keys]
        #         label = f'{angle}-{angle + angle_interval} degrees' if angle != 0 else 'Course Keeping'

        #     bars = ax.bar(speed_keys, counts, bottom=bottom, label=label, color=cmap(idx))
        #     bottom += np.array(counts)

        #     # データラベルを追加
        #     for bar in bars.patches:
        #         # height = bar.get_height()
        #         # if height > 0:
        #         #     ax.text(bar.get_x() + bar.get_width() / 2.0,
        #         #             bar.get_y() + height / 2.0,
        #         #             f'{int(height)}', ha='center', va='center', fontsize=9, color='white')
        #         height = bar.get_height()
        #         if height > 4:  # 4以下は表示しない
        #             ax.text(bar.get_x() + bar.get_width() / 2.0,
        #                     bar.get_y() + height / 2.0,
        #                     f'{int(height)}', ha='center', va='center', fontsize=11, color='white')

        # # ラベルとタイトルを設定
        # ax.set_xlabel('Speed Range [kts]', fontsize=12, labelpad=6)
        # ax.set_ylabel('Count', fontsize=12, labelpad=6)
        # ax.set_xticks(speed_keys)
        # ax.set_xticklabels(speed_labels, rotation=45, ha='right')

        # # 凡例を枠外に配置
        # ax.legend(title='Angles', fontsize=10, title_fontsize=12, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)

        # # レイアウト調整と保存
        # plt.tight_layout()
        # output_file = f'output/elements/{today_str}/stacked_normalized_histogram_.png'
        # plt.savefig(output_file, bbox_inches='tight', dpi=150)
        # print(f"Saved to '{output_file}'")