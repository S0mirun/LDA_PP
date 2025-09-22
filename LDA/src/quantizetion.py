
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import glob
from multiprocessing import Pool
from tqdm import tqdm
import cProfile

from utils.visualization import set_rcParams, Colors, save_fig, calc_fig_range_2D
from utils.time_series_figure import drawing_range_2D


#
PROFILE_FLAG = False
#
PRCSS_TYPE = "M"  # "M": multi process, "N": normal
N_PRCSS = 2
#
port_name = [
    "_Osaka_port1A",
    "_Osaka_port1B",
    "_Tokyo_port2B",
    "_Tokyo_port2C",
    "_Yokkaichi_port1A",
    "_Yokkaichi_port2B"
]
#ORIGINAL_TS_DIR = f"./LDA/src/ts_data/original/csv/{target_port}/"



class Setting:
    def __init__(self) -> None:
        # target
        self.TARGET_ELEMENT = [
            "u [m/s]",
            "vm [m/s]",
        ]
        self.dim_vec = len(self.TARGET_ELEMENT)
        # kmeans
        self.K_EACH = 5
        self.n_code = self.K_EACH ** self.dim_vec
        #
        self.trial_id = f"dim_vec_{str(self.dim_vec)}_n_code_{str(self.n_code)}"
        self.log_dir = f"./outputs/{target_port}/{self.trial_id}/"
        #
        self.save()
    
    def save(self):
        os.makedirs(f"{self.log_dir}setting/", exist_ok=True)
        with open(f"{self.log_dir}setting/setting.bf", 'wb') as f:
            pickle.dump(self, f)
        with open(f"{self.log_dir}setting/setting.txt", mode="w") as f:
            for index_name in self.__dict__:
                f.write('{} : {}, \n'.format(index_name, self.__dict__[index_name]))


class VectorQuantization:
    def __init__(self, ps):
        self.ps = ps

    def main(self):
        self.make_codebook()
        self.quantize_ts()

    def make_codebook(self):
        self.load_vecs()
        self.kmeans()
        self.calc_centroid()
        self.save_codebook()
    
    def quantize_ts(self):
        os.makedirs(f"{self.ps.log_dir}encoded_ts/", exist_ok=True)
        pool = Pool(processes=N_PRCSS)
        paths = list(glob.glob(f"{ORIGINAL_TS_DIR}*.csv"))
        print("\nEncode time series with vectors to sequence of words\n")
        with tqdm(total=len(paths)) as pbar:
            for _ in pool.imap_unordered(self.quantize_ts_from_path, paths):
                pbar.update(1)
        print("\nAll time series encoded.\n")

    def quantize_ts_from_path(self, path):
        original_fname = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, index_col=0)
        df_target = df[self.ps.TARGET_ELEMENT]
        doc = np.empty(len(df_target))
        for i in range(len(doc)):
            vec = df_target.iloc[i].values
            code = self.quantize_vec(vec)
            doc[i] = code
        df["code"] = doc
        df.to_csv(f"{self.ps.log_dir}encoded_ts/{original_fname}_encoded.csv")

    def quantize_vec(self, vec):
        d_min = 1e10
        code = 0
        for label in self.labels:
            centroid = self.codebook_dict[label]
            d = np.linalg.norm(vec - centroid)
            if d < d_min:
                d_min = d
                code = label
        return code

    def load_vecs(self):
        #
        vecs = []
        pool = Pool(processes=N_PRCSS)
        paths = list(glob.glob(f"{ORIGINAL_TS_DIR}*.csv"))
        #
        print("\nLoad vectors from original time series\n")
        for result in tqdm(
            pool.imap_unordered(self._load_vecs, paths),
            total=len(paths),
        ):
            vecs += result
        self.vecs = np.array(vecs)
        #
        if self.ps.dim_vec == 2:
            self.visualize_vecs_2D()
        elif self.ps.dim_vec == 3:
            self.visualize_vecs_3D()

    def _load_vecs(self, path):
        df = pd.read_csv(path, index_col=0)
        df_target = df[self.ps.TARGET_ELEMENT]
        vecs = []
        for i in range(len(df_target)):
            vec = df_target.iloc[i].values
            vecs.append(vec)
        return vecs
    
    def visualize_vecs_2D(self):
        set_rcParams()
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1, 1, 1)
        # ax setting
        ax.set_xlabel("First element")
        ax.set_ylabel("Second element")
        h_ax_range, v_ax_range =  \
            calc_fig_range_2D(self.vecs[:, 0], self.vecs[:, 1], 0.1, False)
        ax.set_xlim(h_ax_range[0], h_ax_range[1])
        ax.set_ylim(v_ax_range[0], v_ax_range[1])
        # ax.set_aspect('equal')
        # plot
        ax.scatter(
            self.vecs[:, 0],
            self.vecs[:, 1],
            color = Colors.black,
            s = 0.3,
        )
        #
        fig.align_labels()
        fig.tight_layout()
        save_fig(f"{self.ps.log_dir}fig/", f"scatter_{self.ps.trial_id}")
    
    def visualize_vecs_3D(self):
        set_rcParams()
        fig = plt.figure(figsize=(15,5))
        #
        ax12 = fig.add_subplot(131)
        ax12.set_xlabel("First element")
        ax12.set_ylabel("Second element")
        h_ax_range, v_ax_range =  \
            calc_fig_range_2D(self.vecs[:, 0], self.vecs[:, 1], 0.1, False)
        ax12.set_xlim(h_ax_range[0], h_ax_range[1])
        ax12.set_ylim(v_ax_range[0], v_ax_range[1])
        # ax12.set_aspect('equal')
        ax12.scatter(
            self.vecs[:, 0],
            self.vecs[:, 1],
            color = Colors.black,
            s = 0.3,
        )
        #
        ax23 = fig.add_subplot(132)
        ax23.set_xlabel("Second element")
        ax23.set_ylabel("Third element")
        h_ax_range, v_ax_range =  \
            calc_fig_range_2D(self.vecs[:, 1], self.vecs[:, 2], 0.1, False)
        ax23.set_xlim(h_ax_range[0], h_ax_range[1])
        ax23.set_ylim(v_ax_range[0], v_ax_range[1])
        # ax23.set_aspect('equal')
        ax23.scatter(
            self.vecs[:, 1],
            self.vecs[:, 2],
            color = Colors.black,
            s = 0.3,
        )
        #
        ax31 = fig.add_subplot(133)
        ax31.set_xlabel("Third element")
        ax31.set_ylabel("First element")
        h_ax_range, v_ax_range =  \
            calc_fig_range_2D(self.vecs[:, 2], self.vecs[:, 0], 0.1, False)
        ax31.set_xlim(h_ax_range[0], h_ax_range[1])
        ax31.set_ylim(v_ax_range[0], v_ax_range[1])
        # ax31.set_aspect('equal')
        ax31.scatter(
            self.vecs[:, 2],
            self.vecs[:, 0],
            color = Colors.black,
            s = 0.3,
        )
        #
        fig.align_labels()
        fig.tight_layout()
        save_fig(f"{self.ps.log_dir}fig/", f"scatter_{self.ps.trial_id}")
    
    def kmeans(self):
        self.classifier = KMeans(n_clusters=self.ps.n_code)
        self.clustered = self.classifier.fit(self.vecs)
        _labels = self.clustered.labels_
        self.labels = np.unique(_labels)
        #
        self.clstrd_df = pd.DataFrame({
            "label": _labels,
        })
        for i, element in enumerate(self.ps.TARGET_ELEMENT):
            self.clstrd_df[element] = self.vecs[:, i]
        #
        self.clstrd_df.to_csv(f"{self.ps.log_dir}kmeans_result.csv")
        print("\nK-means result saved.\n")
        #
        if self.ps.dim_vec == 2:
            self.visualize_kmeans_result_2D()
        elif self.ps.dim_vec == 3:
            self.visualize_kmeans_result_3D()

    def visualize_kmeans_result_2D(self):
        set_rcParams()
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1, 1, 1)
        # ax setting
        ax.set_xlabel("First element")
        ax.set_ylabel("Second element")
        h_ax_range, v_ax_range =  \
            calc_fig_range_2D(self.vecs[:, 0], self.vecs[:, 1], 0.1, False)
        ax.set_xlim(h_ax_range[0], h_ax_range[1])
        ax.set_ylim(v_ax_range[0], v_ax_range[1])
        # ax.set_aspect('equal')
        # plot
        for i, label in enumerate(self.labels):
            df = self.clstrd_df[self.clstrd_df["label"]==label]
            ax.scatter(
                df[self.ps.TARGET_ELEMENT[0]].values,
                df[self.ps.TARGET_ELEMENT[1]].values,
                color = cm.viridis((i+1)/len(self.labels)),
                s = 0.3,
            )
        #
        fig.align_labels()
        fig.tight_layout()
        save_fig(f"{self.ps.log_dir}fig/", f"kmeans_result_{self.ps.trial_id}")
    
    def visualize_kmeans_result_3D(self):
        #
        set_rcParams()
        fig = plt.figure(figsize=(15,5))
        #
        ax12 = fig.add_subplot(131)
        ax12.set_xlabel("First element")
        ax12.set_ylabel("Second element")
        h_ax_range, v_ax_range =  \
            calc_fig_range_2D(self.vecs[:, 0], self.vecs[:, 1], 0.1, False)
        ax12.set_xlim(h_ax_range[0], h_ax_range[1])
        ax12.set_ylim(v_ax_range[0], v_ax_range[1])
        for i, label in enumerate(self.labels):
            df = self.clstrd_df[self.clstrd_df["label"]==label]
            ax12.scatter(
                df[self.ps.TARGET_ELEMENT[0]].values,
                df[self.ps.TARGET_ELEMENT[1]].values,
                color = cm.viridis((i+1)/len(self.labels)),
                s = 0.3,
            )
        #
        ax23 = fig.add_subplot(132)
        ax23.set_xlabel("Second element")
        ax23.set_ylabel("Third element")
        h_ax_range, v_ax_range =  \
            calc_fig_range_2D(self.vecs[:, 1], self.vecs[:, 2], 0.1, False)
        ax23.set_xlim(h_ax_range[0], h_ax_range[1])
        ax23.set_ylim(v_ax_range[0], v_ax_range[1])
        for i, label in enumerate(self.labels):
            df = self.clstrd_df[self.clstrd_df["label"]==label]
            ax23.scatter(
                df[self.ps.TARGET_ELEMENT[1]].values,
                df[self.ps.TARGET_ELEMENT[2]].values,
                color = cm.viridis((i+1)/len(self.labels)),
                s = 0.3,
            )
        #
        ax31 = fig.add_subplot(133)
        ax31.set_xlabel("Third element")
        ax31.set_ylabel("First element")
        h_ax_range, v_ax_range =  \
            calc_fig_range_2D(self.vecs[:, 2], self.vecs[:, 0], 0.1, False)
        ax31.set_xlim(h_ax_range[0], h_ax_range[1])
        ax31.set_ylim(v_ax_range[0], v_ax_range[1])
        for i, label in enumerate(self.labels):
            df = self.clstrd_df[self.clstrd_df["label"]==label]
            ax31.scatter(
                df[self.ps.TARGET_ELEMENT[2]].values,
                df[self.ps.TARGET_ELEMENT[0]].values,
                color = cm.viridis((i+1)/len(self.labels)),
                s = 0.3,
            )
        #
        fig.align_labels()
        fig.tight_layout()
        save_fig(f"{self.ps.log_dir}fig/", f"kmeans_result_{self.ps.trial_id}")
    
    def calc_centroid(self):
        centroids = []
        for label in self.labels:
            df = self.clstrd_df[self.clstrd_df["label"]==label]
            vecs = df[self.ps.TARGET_ELEMENT].values
            centroid = np.mean(vecs, axis=0)
            centroids.append(centroid)
        self.centroids = np.array(centroids)
    
    def save_codebook(self):
        # dict
        self.codebook_dict = {}
        for i, label in enumerate(self.labels):
            self.codebook_dict[label] = self.centroids[i]
        # df
        self.codebook_df = pd.DataFrame({
            "code": self.labels,
        })
        for i, element in enumerate(self.ps.TARGET_ELEMENT):
            self.codebook_df[f"centroid ({element})"] = self.centroids[:, i]
        # to_csv
        self.codebook_df.to_csv(f"{self.ps.log_dir}codebook.csv")
        print("\nCodebook saved.\n")
        # visualize
        if self.ps.dim_vec == 2:
            self.visualize_codebook_2D()
    
    def visualize_codebook_2D(self):
        set_rcParams()
        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(1, 1, 1)
        # ax setting
        ax.set_xlabel("First element")
        ax.set_ylabel("Second element")
        h_ax_range, v_ax_range =  \
            calc_fig_range_2D(self.vecs[:, 0], self.vecs[:, 1], 0.1, False)
        ax.set_xlim(h_ax_range[0], h_ax_range[1])
        ax.set_ylim(v_ax_range[0], v_ax_range[1])
        # ax.set_aspect('equal')
        # scatter clusters
        for i, label in enumerate(self.labels):
            df = self.clstrd_df[self.clstrd_df["label"]==label]
            ax.scatter(
                df[self.ps.TARGET_ELEMENT[0]].values,
                df[self.ps.TARGET_ELEMENT[1]].values,
                color = cm.viridis((i+1)/len(self.labels)),
                s = 0.3,
            )
        # plot centroids
        for i, label in enumerate(self.labels):
            h_c = self.codebook_df.iloc[
                i,
                self.codebook_df.columns.get_loc(f"centroid ({self.ps.TARGET_ELEMENT[0]})"),
            ]
            v_c = self.codebook_df.iloc[
                i,
                self.codebook_df.columns.get_loc(f"centroid ({self.ps.TARGET_ELEMENT[1]})"),
            ]
            ax.scatter(
                h_c, v_c,
                s = 10.0,
                color = cm.viridis((i+1)/len(self.labels)),
                edgecolors = Colors.black,
                linewidths = 1.0,
                label = str(label),
            )
        # ax.legend()
        fig.align_labels()
        fig.tight_layout()
        save_fig(f"{self.ps.log_dir}fig/", f"centroids_{self.ps.trial_id}")



if __name__ == "__main__":
    for port in port_name:
        target_port = port
        ORIGINAL_TS_DIR = f"./LDA/src/ts_data/original/csv/{target_port}/"
        print("target port:    " + str(target_port))
        #
        ps = Setting()
        vq = VectorQuantization(ps)
        if PROFILE_FLAG:
            fname = f"profile_log_{PRCSS_TYPE}_{str(N_PRCSS).zfill(4)}"
            dir = f"{ps.log_dir}profile/"
            os.makedirs(dir, exist_ok=True)
            cProfile.run(
                "vq.main()",
                filename = f"{dir}{fname}.bf"
            )
            print("\nDone\n")
        else:
            vq.main()
        print("\nDone\n")
