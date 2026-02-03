import pandas as pd
import numpy as np

def generate_strings():
    BASE_STRING = "atp_matches_"
    csv = ".csv"
    num = 2015
    files = []
    for i in range(10):
        new_string = BASE_STRING + str(num) + csv
        print(new_string)
        files.append(new_string)
        num += 1
    return files

files = generate_strings()



list_df = [] 

for i in files:
    df = pd.read_csv(i)
    list_df.append(df)

tourney_list = []
print(list_df[0])






    





















