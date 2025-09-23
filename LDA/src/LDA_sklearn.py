
from gensim.corpora import Dictionary
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import glob
from multiprocessing import Pool
from tqdm import tqdm
import pickle
import logging
import re

from utils.visualization import set_rcParams, Colors, save_fig, calc_ax_range, calc_fig_range_2D
from utils.time_series_figure import TimeSeries, make_traj_fig, make_traj_and_velo_fig
from utils.logger import Logger
from utils.ship_geometry import R


#
N_PRCSS = 2
#
POSTPROCESS_FLAG = True
MAKE_VEC_DIST_FIGS = True
N_SMPL_TS = 5
L_SMPL_TS = 1000
#
port_name = [
    "_Osaka_port1A",
    "_Osaka_port1B",
    "_Tokyo_port2B",
    "_Tokyo_port2C",
    "_Yokkaichi_port1A",
    "_Yokkaichi_port2B"
]
target_port = port_name[0]
ORIGINAL_TS_DIR = f"./LDA/src/ts_data/original/csv/{target_port}"
DT = 1.0  # [s]


class Setting: # セッティング
    def __init__(self):
        # vq
        self.TARGET_ELEMENT = [
            "u [m/s]",
            "vm [m/s]"
        ]
        self.dim_vec = len(self.TARGET_ELEMENT)
        self.K_EACH = 5
        self.n_code = self.K_EACH ** self.dim_vec
        self.vq_trial_id = f"dim_vec_{str(self.dim_vec)}_n_code_{str(self.n_code)}"
        self.vq_log_dir = f"./outputs/{target_port}/{self.vq_trial_id}/"
        # segmentation
        self.PERIOD = 1
        self.L_DOC = 30
        self.DELTA_TS_SHIFT = 15
        self.sgmnt_id = f"PERIOD_{self.PERIOD}_L_DOC_{self.L_DOC}_DELTA_{self.DELTA_TS_SHIFT}"
        self.sgmnt_log_dir = f"{self.vq_log_dir}sgmntd/{self.sgmnt_id}/"
        # LDA
        self.N_TOPIC = 5 # トピック数
        self.KAPPA = 0.7
        self.TAU0 = 10.0
        self.MAX_ITER = 10
        self.BATCH_SIZE = 128
        self.EVALUATE_EVERY = 1
        self.TOTAL_SAMPLES = 1e6
        self.PERP_TOL = 0.1
        self.MEAN_CHANGE_TOL = 1e-3
        self.MAX_DOC_UPDATE_ITER = 100
        self.RANDOM_STATE = 1521
        self.lda_id = f"N_{self.N_TOPIC}_SEED_{self.RANDOM_STATE}"
        #
        self.log_dir = f"{self.sgmnt_log_dir}LDA_sklearn/{self.lda_id}/"
        #
        self.save()
    
    def save(self):
        os.makedirs(f"{self.log_dir}setting/", exist_ok=True)
        with open(f"{self.log_dir}setting/setting.bf", 'wb') as f:
            pickle.dump(self, f)
        with open(f"{self.log_dir}setting/setting.txt", mode="w") as f:
            for index_name in self.__dict__:
                f.write('{} : {}, \n'.format(index_name, self.__dict__[index_name]))


class LDAClustering:
    def __init__(self, ps):
        self.ps = ps

    def main(self): # ここを順に実行していく
        #
        logging.basicConfig(
            filename=f"{self.ps.log_dir}gensim.log",
            format='%(asctime)s : %(levelname)s : %(message)s',
            level=logging.INFO,
        )
        self.make_dict()
        self.make_corpus()
        self.lda()
        if POSTPROCESS_FLAG:
            self.analyze_topic()
    
    def lda(self):
        #
        self.make_input_data()
        #
        self.lda_model_dir = f"{self.ps.log_dir}trained_model/"
        self.lda_fname = f"trained_model_{self.ps.lda_id}"
        #
        if os.path.isfile(f"{self.lda_model_dir}{self.lda_fname}.pkl"):
            self.load_model()
        else:
            # train
            self.lda_model = LatentDirichletAllocation(
                n_components=self.ps.N_TOPIC,
                learning_method="online",
                learning_decay=self.ps.KAPPA,
                learning_offset=self.ps.TAU0,
                max_iter=self.ps.MAX_ITER,
                batch_size=self.ps.BATCH_SIZE,
                evaluate_every=self.ps.EVALUATE_EVERY,
                total_samples=self.ps.TOTAL_SAMPLES,
                perp_tol=self.ps.PERP_TOL,
                mean_change_tol=self.ps.MEAN_CHANGE_TOL,
                max_doc_update_iter=self.ps.MAX_DOC_UPDATE_ITER,
                # n_jobs=N_PRCSS,
                random_state=self.ps.RANDOM_STATE,
            )
            os.environ['JOBLIB_TEMP_FOLDER'] = '/tmp'
            self.lda_model.fit(self.X)
            # save
            self.save_model()
            self.save_train_trans()
            self.make_train_trans_fig()
    
    def analyze_topic(self):
        self.read_codebook()
        self.save_word_dists()
        self.gen_ts(n_ts=N_SMPL_TS)
        self.save_topic_proportions()
        self.arrange_topic_prop_trans()
        self.make_topic_prop_trans_figs()
        if MAKE_VEC_DIST_FIGS:
            if self.ps.dim_vec == 2:
                self.make_vec_dist_figs_2D()
            elif self.ps.dim_vec == 3:
                self.make_vec_dist_figs_3D()
    
    def save_word_dists(self):
        #
        word_dists = self.lda_model.components_
        self.word_dists = []
        n_zfill = len(str(len(word_dists)-1)) + 1
        os.makedirs(f"{self.ps.log_dir}word_dists/", exist_ok=True)
        # for each topic
        for i, word_dist_nrmlzd in enumerate(word_dists):
            word_dist = word_dist_nrmlzd / sum(word_dist_nrmlzd)
            ids_w_sorted = np.argsort(-word_dist)
            words = []
            word_dist_sorted = []
            for id_w_sorted in ids_w_sorted:
                word = self.dictionary[id_w_sorted]
                prob = word_dist[id_w_sorted]
                words.append(word)
                word_dist_sorted.append(prob)
            df = pd.DataFrame({
                "i_word": ids_w_sorted,
                "word": words,
                "prob": word_dist_sorted,
            })
            df.to_csv(f"{self.ps.log_dir}word_dists/topic_{str(i).zfill(n_zfill)}.csv")
            self.word_dists.append(df)
        print("\nWord distributions for each topic saved.\n")
    
    def gen_ts(self, n_ts):
        #
        np.random.seed(self.ps.RANDOM_STATE)
        #
        for i_topic in range(self.ps.N_TOPIC):
            #
            topic_str = f"topic_{str(i_topic).zfill(5)}"
            #
            logger = Logger(
                save_dir=f"{self.ps.log_dir}sample_ts/{topic_str}/csv/",
                header=[
                    "t [s]", "word",
                    "u [m/s]", "vm [m/s]",
                    "p_x [m]", "p_y [m]", "gyro deg [rad]", "gyro deg [deg]",
                ]
            )
            #
            wd = self.word_dists[i_topic]
            theta = wd["prob"].values
            def mnrv2word(mnrv):
                w = wd.iloc[
                    np.where(mnrv == 1)[0][0],
                    wd.columns.get_loc("word"),
                ]
                return w
            #
            fig_ts_l = []
            #
            for i_smpl in range(n_ts):
                #
                smpl_ts_name = f"{topic_str}_smpl_ts_{str(i_smpl).zfill(5)}_L_{L_SMPL_TS}"
                ts_save_name = f"{smpl_ts_name}_{self.ps.lda_id}"
                #
                multinominal_rvs = np.random.multinomial(
                    n=1,
                    pvals=theta,
                    size=L_SMPL_TS+1,
                )
                # init
                t = 0.0
                mnrv = multinominal_rvs[0]
                word = mnrv2word(mnrv)
                zeta = np.zeros(3)  # gyro deg in [rad]
                #
                logger.reset()
                #
                for j in range(L_SMPL_TS):
                    # log
                    info =  \
                        [t]  \
                        + [word]  \
                        + list(zeta)  \
                        + [np.rad2deg(zeta[2])]
                    logger.append(info)
                    # step
                    t += DT
                    mnrv = multinominal_rvs[j+1]
                    word = mnrv2word(mnrv)
                # save as csv
                logger.save(fname=f"{ts_save_name}.csv", fig_flag=False,)
                #
                df = logger.get_df()
                fig_ts = TimeSeries(
                    df=df, dt=DT,
                    label=smpl_ts_name, L=200.0, B=30.0,
                    color=Colors.black, line_width=0.5
                )
                fig_ts_l.append(fig_ts)
            #
            make_traj_and_velo_fig(
                ts_list=fig_ts_l,
                ship_plot_step_period=int(L_SMPL_TS/5),
                alpha_ship_shape=0.0, fig_size=(14, 7)
            )
            save_fig(
                f"{self.ps.log_dir}sample_ts/{topic_str}/fig/",
                f"traj_and_velo_{topic_str}_smpl_ts_{self.ps.lda_id}"
            )
            make_traj_fig(
                ts_list=fig_ts_l,
                ship_plot_step_period=int(L_SMPL_TS/10),
                alpha_ship_shape=0.0, fig_size=(4, 4),
                title=f"$k = {i_topic+1}$",
                legend_flag=False,
            )
            save_fig(
                f"{self.ps.log_dir}sample_ts/{topic_str}/fig/",
                f"traj_{topic_str}_smpl_ts_{self.ps.lda_id}"
            )
        
    def save_topic_proportions(self):
        #
        topic_props = self.lda_model.transform(self.X)
        self.topic_props = {}
        os.makedirs(f"{self.ps.log_dir}topic_props/", exist_ok=True)
        for i_doc, prop in enumerate(topic_props):
            doc_name = list(self.docs_dict.keys())[i_doc]
            #
            topics = []
            probs = []
            for topic, prob in enumerate(prop):
                topics.append(topic)
                probs.append(prob)
            #
            df = pd.DataFrame({
                "topic": topics,
                "prob": probs,
            })
            df.to_csv(f"{self.ps.log_dir}topic_props/tp_{doc_name}_{self.ps.lda_id}.csv")
            #
            self.topic_props = dict(**self.topic_props, **{doc_name: df})
        print("\nTopic proportions for each document saved.\n")
    
    def arrange_topic_prop_trans(self):
        #
        self.original_ts_dict = {}
        for path in glob.glob(f"{ORIGINAL_TS_DIR}*.csv"):
            df = pd.read_csv(path, index_col=0)
            original_ts_name = os.path.splitext(os.path.basename(path))[0]
            self.original_ts_dict = dict(
                **self.original_ts_dict,
                **{original_ts_name: df}
            )
        #
        self.prop_trans_dict = {}
        self.tp_trans_log_dir = f"{self.ps.log_dir}topic_props/topic_prop_trans/"
        os.makedirs(f"{self.tp_trans_log_dir}/csv", exist_ok=True)
        doc_names_all = list(self.docs_dict.keys())
        for ts_name in list(self.original_ts_dict.keys()):
            #
            doc_names = [name for name in doc_names_all if name[:3] == ts_name]
            topic_prop_trans = np.empty((len(doc_names), self.ps.N_TOPIC))
            tstep_l = []
            for i, doc_name in enumerate(doc_names):
                prop_df = self.topic_props[doc_name]
                topic_prop_trans[i, :] = prop_df["prob"].values
                tstep_l.append(i)
            #
            trans_df = pd.DataFrame(
                topic_prop_trans,
                columns=[str(i) for i in list(range(self.ps.N_TOPIC))],
            )
            trans_df["t [s]"] = tstep_l
            self.prop_trans_dict = dict(
                **self.prop_trans_dict,
                **{ts_name: trans_df}
            )
        #
        print("\nTransition of topic prop saved.\n")
    
    def make_topic_prop_trans_figs(self):
        for ts_name in list(self.original_ts_dict.keys()):
            self.make_topic_prop_trans_fig(ts_name)
    
    def make_topic_prop_trans_fig(self, ts_name):
        #
        original_ts_df = self.original_ts_dict[ts_name]
        tp_trans_df = self.prop_trans_dict[ts_name]
        #
        set_rcParams()
        fig = plt.figure(figsize=(7, 6))
        #
        ax_topic = fig.add_subplot(4, 1, 1)
        ax_u = fig.add_subplot(4, 1, 2)
        ax_v = fig.add_subplot(4, 1, 3)
        ax_r = fig.add_subplot(4, 1, 4)
        # time
        time_min = min(original_ts_df["t [s]"])
        time_max = max(original_ts_df["t [s]"])
        # tipic
        # ax_topic.set_xlabel("$t$ [s]")
        ax_topic.set_ylabel(r"${\theta}_{dk}$")
        ax_topic.set_xlim(time_min, time_max)
        ax_topic.set_ylim(calc_ax_range(
            data=np.array([0, 1]),
            margin_rate=0.05,
        ))
        #
        tp_trans_df["t [s]"] = np.arange(
            min(original_ts_df["t [s]"]),
            min(original_ts_df["t [s]"]) + self.ps.DELTA_TS_SHIFT * (len(tp_trans_df) - 1) + 1,
            self.ps.DELTA_TS_SHIFT,
        )
        tp_trans_df.to_csv(f"{self.tp_trans_log_dir}csv/{ts_name}_prop_trans_{self.ps.lda_id}.csv")
        #
        for i in range(self.ps.N_TOPIC):
            ax_topic.plot(
                tp_trans_df["t [s]"].values,
                tp_trans_df[str(i)].values,
                color = cm.viridis(i / (self.ps.N_TOPIC - 1)),
                linestyle = "solid",
                linewidth = 0.5,
                label = str(i),
            )
        # ax_topic.legend()
        # u
        # ax_u.set_xlabel("$t$ [s]")
        ax_u.set_ylabel("$u$ [m/s]")
        ax_u.set_xlim(time_min, time_max)
        ax_u.set_ylim(calc_ax_range(
            data=original_ts_df["u [m/s]"],
            margin_rate=0.05,
        ))
        ax_u.plot(
            original_ts_df["t [s]"],
            original_ts_df["u [m/s]"],
            color = Colors.black,
            linestyle = "solid",
            linewidth = 0.5,
        )
        # ax_u.legend()
        # v
        # ax_v.set_xlabel("$t$ [s]")
        ax_v.set_ylabel("$v$ [m/s]")
        ax_v.set_xlim(time_min, time_max)
        ax_v.set_ylim(calc_ax_range(
            data=original_ts_df["vm [m/s]"],
            margin_rate=0.05,
        ))
        ax_v.plot(
            original_ts_df["t [s]"],
            original_ts_df["vm [m/s]"],
            color = Colors.black,
            linestyle = "solid",
            linewidth = 0.5,
        )
        # ax_v.legend()
        #
        fig.align_labels()
        fig.tight_layout()
        #
        save_fig(f"{self.tp_trans_log_dir}fig/", f"tp_trans_{ts_name}_{self.ps.lda_id}")
    
    def read_codebook(self):
        codebook_float = pd.read_csv(f"{self.ps.vq_log_dir}codebook.csv", index_col=0)
        self.codebook = codebook_float.astype({"code": str})
    
    def make_vec_dist_figs_2D(self):
        set_rcParams()
        for i_topic, word_dist in enumerate(self.word_dists):
            fig = plt.figure(figsize=(5,5))
            ax = fig.add_subplot(1, 1, 1)
            # ax setting
            ax.set_xlabel("First element")
            ax.set_ylabel("Second element")
            df_label_h = f"centroid ({self.ps.TARGET_ELEMENT[0]})"
            df_label_v = f"centroid ({self.ps.TARGET_ELEMENT[1]})"
            h_ax_range, v_ax_range = calc_fig_range_2D(
                data_h=self.codebook[df_label_h].values,
                data_v=self.codebook[df_label_v].values,
                margin=0.1,
                square_flag=False
            )
            ax.set_xlim(h_ax_range[0], h_ax_range[1])
            ax.set_ylim(v_ax_range[0], v_ax_range[1])
            # ax.set_aspect('equal')
            # plot
            for i in range(len(word_dist)-1, -1, -1):
                word = str(word_dist.iloc[
                    i,
                    word_dist.columns.get_loc("word"),
                ])
                prob = word_dist.iloc[
                    i,
                    word_dist.columns.get_loc("prob"),
                ]
                vec = self.decode(code=word)
                prob_normalized = (prob - min(word_dist["prob"])) / (max(word_dist["prob"]) - min(word_dist["prob"]))
                ax.scatter(
                    vec[0],
                    vec[1],
                    color = cm.binary(prob_normalized),
                    s = 0.3,
                )
            #
            fig.align_labels()
            fig.tight_layout()
            save_fig(f"{self.ps.log_dir}word_dists/fig/", f"vec_dist_topic_{i_topic}_{self.ps.lda_id}")
    
    def make_vec_dist_figs_3D(self):
        pool = Pool(processes=N_PRCSS)
        inds = list(range(len(self.word_dists)))
        pool.map(self.make_vec_dist_fig_3D_for_a_topic, inds)
    
    def make_vec_dist_fig_3D_for_a_topic(self, i_topic):
        #
        word_dist = self.word_dists[i_topic]
        #
        set_rcParams()
        fig = plt.figure(figsize=(15,5))
        #
        ax12 = fig.add_subplot(131)
        ax12.set_xlabel("First element")
        ax12.set_ylabel("Second element")
        df_label_h = f"centroid ({self.ps.TARGET_ELEMENT[0]})"
        df_label_v = f"centroid ({self.ps.TARGET_ELEMENT[1]})"
        h_ax_range, v_ax_range = calc_fig_range_2D(
            data_h=self.codebook[df_label_h].values,
            data_v=self.codebook[df_label_v].values,
            margin=0.1,
            square_flag=False
        )
        ax12.set_xlim(h_ax_range[0], h_ax_range[1])
        ax12.set_ylim(v_ax_range[0], v_ax_range[1])
        #
        ax23 = fig.add_subplot(132)
        ax23.set_xlabel("Second element")
        ax23.set_ylabel("Third element")
        df_label_h = f"centroid ({self.ps.TARGET_ELEMENT[1]})"
        df_label_v = f"centroid ({self.ps.TARGET_ELEMENT[2]})"
        h_ax_range, v_ax_range = calc_fig_range_2D(
            data_h=self.codebook[df_label_h].values,
            data_v=self.codebook[df_label_v].values,
            margin=0.1,
            square_flag=False
        )
        ax23.set_xlim(h_ax_range[0], h_ax_range[1])
        ax23.set_ylim(v_ax_range[0], v_ax_range[1])
        #
        ax31 = fig.add_subplot(133)
        ax31.set_xlabel("Third element")
        ax31.set_ylabel("First element")
        df_label_h = f"centroid ({self.ps.TARGET_ELEMENT[2]})"
        df_label_v = f"centroid ({self.ps.TARGET_ELEMENT[0]})"
        h_ax_range, v_ax_range = calc_fig_range_2D(
            data_h=self.codebook[df_label_h].values,
            data_v=self.codebook[df_label_v].values,
            margin=0.1,
            square_flag=False
        )
        ax31.set_xlim(h_ax_range[0], h_ax_range[1])
        ax31.set_ylim(v_ax_range[0], v_ax_range[1])
        #
        for i in range(len(word_dist)-1, -1, -1):
            word = str(word_dist.iloc[
                i,
                word_dist.columns.get_loc("word"),
            ])
            prob = word_dist.iloc[
                i,
                word_dist.columns.get_loc("prob"),
            ]
            vec = self.decode(code=word)
            prob_normalized = (prob - min(word_dist["prob"])) / (max(word_dist["prob"]) - min(word_dist["prob"]))
            ax12.scatter(
                vec[0],
                vec[1],
                color = cm.binary(prob_normalized),
                s = 0.3,
            )
            ax23.scatter(
                vec[1],
                vec[2],
                color = cm.binary(prob_normalized),
                s = 0.3,
            )
            ax31.scatter(
                vec[2],
                vec[0],
                color = cm.binary(prob_normalized),
                s = 0.3,
            )
        #
        fig.align_labels()
        fig.tight_layout()
        save_fig(f"{self.ps.log_dir}word_dists/fig/", f"vec_dist_topic_{i_topic}_{self.ps.lda_id}")
    
    def save_model(self):
        os.makedirs(self.lda_model_dir, exist_ok=True)
        with open(f"{self.lda_model_dir}{self.lda_fname}.pkl",'wb') as f:
            pickle.dump(self.lda_model, f)
        print(f"\nTrained model saved: {self.ps.lda_id}\n")
    
    def load_model(self):
        with open(f"{self.lda_model_dir}{self.lda_fname}.pkl", 'rb') as f:
            self.lda_model = pickle.load(f)
        print(f"\nTrained model loaded: {self.ps.lda_id}\n")
    
    def save_train_trans(self):
        n_iter = getattr(self.lda_model, "n_iter_", None)
        final_bound = float(getattr(self.lda_model, "bound_", float("nan")))

        X_eval = None
        for name in ("X_valid", "X_train", "X"):
            if hasattr(self, name) and getattr(self, name) is not None:
                X_eval = getattr(self, name)
                break

        try:
            perp = float(self.lda_model.perplexity(X_eval)) if X_eval is not None else float("nan")
        except Exception:
            perp = float("nan")

        out_dir = os.path.join(self.ps.log_dir, "train_trans")
        os.makedirs(out_dir, exist_ok=True)

        self.train_trans_df = pd.DataFrame([{
            "iter": n_iter if n_iter is not None else -1,
            "perp": perp,
            "bound": final_bound,
        }])

        out_path = os.path.join(out_dir, f"train_trans_{self.ps.lda_id}.csv")
        self.train_trans_df.to_csv(out_path, index=False)
        print(f"[save_train_trans] wrote: {out_path} (iter={n_iter}, perp={perp:.6g}, bound={final_bound:.6g})")

    
    def make_train_trans_fig(self):
        #
        itera = self.train_trans_df["iter"].values
        perp = self.train_trans_df["perp"].values
        #
        set_rcParams()
        fig = plt.figure(figsize=(5, 5))
        #
        ax_perp = fig.add_subplot(111)
        ax_perp.set_xlabel("Iterations")
        ax_perp.set_ylabel("Perplexity")
        ax_perp.set_xlim(min(itera), max(itera))
        ax_perp.set_ylim(1.0, 1.1 * max(perp))
        ax_perp.plot(
            itera,
            perp,
            color = Colors.black,
            lw = 0.5,
        )
        #
        fig.align_labels()
        fig.tight_layout()
        #
        save_fig(f"{self.ps.log_dir}train_trans/", f"train_trans_{self.ps.lda_id}")
    
    def decode(self, code):
        vec = self.codebook.loc[self.codebook["code"]==code].values[0][1:]
        return np.array(vec)
    
    def make_dict(self):
        # read documents
        self.docs_dict = {}
        # pool = Pool(processes=N_PRCSS)
        paths = sorted(list(glob.glob(f"{self.ps.sgmnt_log_dir}csv/*.csv")))
        print("\nRead encoded time series (documents)\n")
        # for result in tqdm(
        #     pool.imap_unordered(self.read_encoded_ts_from_path, paths),
        #     total=len(paths),
        # ):
        #     self.docs_dict = dict(**self.docs_dict, **result)
        # tqdm
        for path in tqdm(paths):
            doc_d = self.read_encoded_ts_from_path(path)
            self.docs_dict = dict(**self.docs_dict, **doc_d)
        #
        self.dictionary = Dictionary(self.docs_dict.values())
        self.n_words = len(self.dictionary)
        print(f"\nNumber of vocabulary: {self.n_words}\n")
    
    def read_encoded_ts_from_path(self, path):
        doc_name = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path, index_col=0)
        doc_float = df["code"]
        doc_int = doc_float.astype("int64").values.tolist()
        doc = [str(w) for w in doc_int]
        return {doc_name: doc}
    
    def make_corpus(self):
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.docs_dict.values()]
        print(f"\nNumber of documents: {len(self.corpus)}\n")
    
    def make_input_data(self):
        #
        self.indata_dir = f"{self.ps.sgmnt_log_dir}input_data/"
        self.indata_fname = f"lda_indata_{self.ps.sgmnt_id}"
        #
        if os.path.isfile(f"{self.indata_dir}{self.indata_fname}.pkl"):
            # load
            with open(f"{self.indata_dir}{self.indata_fname}.pkl", 'rb') as f:
                self.X = pickle.load(f)
            print(f"\nInput data loaded: {self.ps.sgmnt_id}\n")
        else:
            # make
            self.X = []
            for bow in self.corpus:
                freq = [0 for _ in range(self.n_words)]
                for (i_word, n) in bow:
                    freq[i_word] = n
                self.X.append(freq)
            # save
            os.makedirs(self.indata_dir, exist_ok=True)
            with open(f"{self.indata_dir}{self.indata_fname}.pkl",'wb') as f:
                pickle.dump(self.X, f)
            print(f"\nInput data saved: {self.ps.sgmnt_id}\n")


if __name__ == "__main__":
    #
    ps = Setting()
    lda_clustering = LDAClustering(ps)
    #
    lda_clustering.main()
    #
    print("\nDone\n")
