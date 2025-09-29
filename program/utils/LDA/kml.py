
import numpy as np
import pandas as pd


def kml_based_txt_to_csv(txt_path, log_dir, csv_name):
    #
    with open(txt_path, 'r') as f:
        string = f.read()
    string_split = string.split()
    value = []
    for i in range(len(string_split)):
        each_value = []
        each_string = string_split[i].split(',')
        each_value.append(float(each_string[1]))
        each_value.append(float(each_string[0]))
        value.append(each_value)
    #
    value_array = np.array(value)
    value_df = pd.DataFrame(value_array, columns = ['latitude', 'longitude',])
    value_df.to_csv(f"{log_dir}{csv_name}.csv")
