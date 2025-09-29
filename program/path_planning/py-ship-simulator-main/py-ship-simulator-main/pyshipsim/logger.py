import os
import pandas as pd


class Logger(object):

    def reset(self, header: list[str]):
        self.header = header
        self.log = []

    def append(self, l: list):
        self.log.append(l)

    def get_df(self, steps: int = None) -> pd.DataFrame:
        log = self.log if steps is None else self.log[-steps:]
        df = pd.DataFrame(log, columns=self.header).set_index(self.header[0])
        return df

    def to_csv(self, dir: str = "./", fname: str = "test", df: pd.DataFrame = None):
        if df is None:
            df = self.get_df()
        path = f"{dir}/{fname}.csv"
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        df.to_csv(path)
