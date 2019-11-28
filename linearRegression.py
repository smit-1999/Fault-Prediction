import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def get_csv():
    """
    This method fetches the csv files in the dataset and returns them.
    :return: It returns the directory of the dataset and the list of
     files object.
    """
    curr_dir = os.getcwd()  # gets the current working directory
    directory = os.path.join(curr_dir, "modified")  # concatenates
    csv_files = os.listdir(directory)  # list of all files in the directory
    return directory, csv_files


def main():
    directory, csv_files = get_csv()

    for file in csv_files:
        if os.path.isdir(directory + '\\' + file) is False:
            df = pd.read_csv(directory + "\\" + file)  # create a dataframe
            print(directory + "\\" + file)
            kf = KFold(n_splits=5, shuffle=True)
            boxplot_data = []
            # iterate through each column except the target variable column
            for i in range(0, len(df.columns)-1):
                df[df.columns[i]] = df[df.columns[i]].astype(int)   # convert str to int
                scores = []
                print(df.columns)
                for j in range(5):
                    result = next(kf.split(df[df.columns[i]]), None)
                    x_train = df.iloc[result[0]][df.columns[i]].to_numpy().reshape(-1, 1)
                    x_test = df.iloc[result[1]][df.columns[i]].to_numpy().reshape(-1, 1)
                    y_train = df['bug'].iloc[result[0]]
                    y_test = df['bug'].iloc[result[1]]
                    model = LinearRegression().fit(x_train, y_train)  # create a linear regression model
                    print('Intercept for column:', i, 'is:', model.intercept_)
                    print('Slope for column :', i, 'is:', model.coef_)
                    print('Score of the model:', model.score(x_test, y_test), '\n\n')
                    scores.append(model.score(x_test, y_test))
                boxplot_data.append(scores)
            plt.boxplot(boxplot_data)
            print(directory)
            savepath = directory + '\\boxplot_images\\'
            plt.savefig(savepath + file + '.png')
            plt.clf()
            print('###############')


main()
