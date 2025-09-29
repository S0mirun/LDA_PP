
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.visualization import save_fig


class Logger:
    def __init__(self, save_dir, header):
        self.save_dir = save_dir
        self.header = header

    def reset(self):
        self.log = []

    def append(self, l):
        self.log.append(l)

    def save(self, fname, fig_flag):
        # csv
        df = pd.DataFrame(
            self.log,
            columns=self.header
        )
        os.makedirs(self.save_dir, exist_ok = True)
        df.to_csv(self.save_dir + f'{fname}.csv')
        print(f'\ncsv saved: \"{fname}\"\n')
        # fig
        if fig_flag:
            fig = plt.figure(figsize=(6.0, 1.5 * len(df.columns)))
            columns_list = df.columns.values.tolist()
            x_axis_label = columns_list[0]
            columns_list.remove(x_axis_label)
            for i, column in enumerate(columns_list):
                ax = fig.add_subplot(len(columns_list), 1, i+1)
                ax.plot(
                    df[x_axis_label].values,
                    df[column].values,
                    c = "black",
                    linewidth = 1.0,
                )
                ax.set_ylabel(column)
            fig.tight_layout()
            save_fig(f"{self.save_dir}temp_fig/", fname)

    def get_df(self):
        df = pd.DataFrame(self.log, columns=self.header)
        return df
