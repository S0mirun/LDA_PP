
from gensim.corpora import Dictionary
from gensim.models import LdaModel
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

from utils.visualization import set_rcParams, Colors, save_fig, calc_fig_range_2D


#
N_PRCSS = 6


class Setting:
    def __init__(self):
        # vq
        self.TARGET_ELEMENT = [
            "u [m/s]",
            "vm [m/s]",
            "r [deg/s]",
        ]
        self.dim_vec = len(self.TARGET_ELEMENT)
        self.K_EACH = 5
        self.n_code = self.K_EACH ** self.dim_vec
        self.vq_trial_id = f"dim_vec_{str(self.dim_vec)}_n_code_{str(self.n_code)}"
        self.vq_log_dir = f"./outputs/{self.vq_trial_id}/"
        # segmentation
        self.PERIOD = 1
        self.L_DOC = 100
        self.DELTA_TS_SHIFT = 50
        self.sgmnt_id = f"PERIOD_{self.PERIOD}_L_DOC_{self.L_DOC}_DELTA_{self.DELTA_TS_SHIFT}"
        self.sgmnt_log_dir = f"{self.vq_log_dir}sgmntd/{self.sgmnt_id}/"
        # LDA
        self.N_TOPIC = 10
        self.PASSES = 20
        self.ITERATIONS = 1000
        self.EVAL_EVERY = 1
        self.SEED = 1521
        self.lda_id = f"N_TOPIC_{self.N_TOPIC}_PASSES_{self.PASSES}_ITERATIONS_{self.ITERATIONS}_SEED_{self.SEED}"
        #
        self.log_dir = f"{self.sgmnt_log_dir}LDA/{self.lda_id}/"
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

    def main(self):
        #
        logging.basicConfig(
            filename="gensim.log",
            format='%(asctime)s : %(levelname)s : %(message)s',
            level=logging.INFO,
        )
        self.make_dict()
        self.make_corpus()
        self.lda()
        self.save_train_trans()
        self.make_train_trans_fig()
        self.analyze_topic()
    
    def lda(self):
        #
        lda_model_dir = f"{self.ps.log_dir}trained_model/"
        lda_fname = f"trained_model_{self.ps.lda_id}.model"
        if os.path.isfile(f"{lda_model_dir}{lda_fname}"):
            # load saved model
            self.lda_model = LdaModel.load(f"{lda_model_dir}{lda_fname}")
            print(f"\nTrained model loaded: {self.ps.lda_id}\n")
        else:
            # train
            self.lda_model = LdaModel(
                corpus=self.corpus,
                num_topics=self.ps.N_TOPIC,
                alpha="auto",
                eta="auto",
                iterations=self.ps.ITERATIONS,
                passes=self.ps.PASSES,
                eval_every=self.ps.EVAL_EVERY,
                random_state=self.ps.SEED,
            )
            # save trained model
            os.makedirs(lda_model_dir, exist_ok=True)
            self.lda_model.save(f"{lda_model_dir}{lda_fname}")
            print(f"\nTrained model saved: {self.ps.lda_id}\n")
    
    def analyze_topic(self):
        self.save_word_dists()
        self.save_topic_proportions()
        self.read_codebook()
        if self.ps.dim_vec == 2:
            self.make_vec_dist_figs_2D()
        elif self.ps.dim_vec == 3:
            self.make_vec_dist_figs_3D()
        print(f"\nAnalysis of topic model saved.\n")
    
    def save_word_dists(self):
        #
        word_dists = self.lda_model.get_topics()
        self.word_dists = []
        n_zfill = len(str(len(word_dists)-1)) + 1
        os.makedirs(f"{self.ps.log_dir}word_dists/", exist_ok=True)
        # for each topic
        for i, word_dist in enumerate(word_dists):
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
    
    def save_topic_proportions(self):
        #
        topic_props = self.lda_model.get_document_topics(self.corpus, minimum_probability=0.0)
        self.topic_props = {}
        os.makedirs(f"{self.ps.log_dir}topic_props/", exist_ok=True)
        for i_doc, prop in enumerate(topic_props):
            doc_name = list(self.docs_dict.keys())[i_doc]
            #
            topics = []
            probs = []
            for (topic, prob) in prop:
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
        set_rcParams()
        for i_topic, word_dist in enumerate(self.word_dists):
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
            df_label_v = f"centroid ({self.ps.TARGET_ELEMENT[1]})"
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
    
    def save_train_trans(self):
        #
        p = re.compile("(-*\d+\.\d+) per-word .* (\d+\.\d+) perplexity")
        matches = [p.findall(l) for l in open('gensim.log')]
        matches = [m for m in matches if len(m) > 0]
        tuples = [t[0] for t in matches]
        perplexity = [float(t[1]) for t in tuples]
        log2_likelihood = [float(t[0]) for t in tuples]
        iter = list(range(0,len(tuples)*10,10))
        #
        os.makedirs(f"{self.ps.log_dir}train_trans/", exist_ok=True)
        # save as csv
        self.train_trans_df = pd.DataFrame({
            "iter": iter,
            "perp": perplexity,
            "log2_liklihood": log2_likelihood,
        })
        self.train_trans_df.to_csv(f"{self.ps.log_dir}train_trans/train_trans_{self.ps.lda_id}.csv")
    
    def make_train_trans_fig(self):
        #
        itera = self.train_trans_df["iter"].values
        perp = np.exp2(-self.train_trans_df["log2_liklihood"].values)
        log2_l = self.train_trans_df["log2_liklihood"].values
        #
        set_rcParams()
        fig = plt.figure(figsize=(8, 4))
        #
        ax_ll = fig.add_subplot(131)
        ax_ll.set_xlabel("Iterations")
        ax_ll.set_ylabel("$\log_{2}$ likelihood")
        ax_ll.set_xlim(min(itera), max(itera))
        ax_ll.plot(
            itera,
            log2_l,
            color = Colors.black,
            lw = 0.5,
        )
        #
        ax_perp = fig.add_subplot(132)
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
        ax_perp_close = fig.add_subplot(133)
        ax_perp_close.set_xlabel("Iterations")
        ax_perp_close.set_ylabel("Perplexity")
        ax_perp_close.set_xlim(min(itera), max(itera))
        ax_perp_close.set_ylim(1.0, 10.0 * perp[-1])
        ax_perp_close.plot(
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
        print(f"\nNumber of vocabulary: {len(self.dictionary)}\n")
    
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


if __name__ == "__main__":
    #
    ps = Setting()
    lda_clustering = LDAClustering(ps)
    #
    lda_clustering.main()
    #
    print("\nDone\n")
