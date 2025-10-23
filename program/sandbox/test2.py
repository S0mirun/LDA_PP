import pandas as pd
df = pd.read_pickle("df_rest_all.pickle")  # 圧縮付きでもOK
print(type(df), getattr(df, "shape", None))
