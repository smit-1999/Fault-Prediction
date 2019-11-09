import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


def get_csv():

    curr_dir = os.getcwd()  # gets the current working directory
    directory = os.path.join(curr_dir, "test_classInfo")  # concatenates
    csv_files = os.listdir(directory)  # list of all files in the directory
    return directory, csv_files


def main():
    directory, csv_files = get_csv()

    for file in csv_files:
        df = pd.read_csv(directory + "\\" + file)

        for i in range(0, len(df.columns)-1):
            df[df.columns[i]] = df[df.columns[i]].astype(int)
            x = df[df.columns[i]].to_numpy().reshape(-1, 1)
            y = df[df.columns[-1]].to_numpy()
            print(type(x[0][0]), type(y[0]))
            model = LinearRegression().fit(x, y)
            print('intercept:', model.intercept_)
            print('slope:', model.coef_)


main()
