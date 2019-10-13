import numpy as np
import pandas as pd
import os
import statistics
import math

def calculate_Confidence(df,confidence):

    Z = dict()  # A dictionary to store z values for a given confidence value
    Z[0.80] = 1.282
    Z[0.85] = 1.440
    Z[0.90] = 1.645
    Z[0.95] = 1.960
    Z[0.99] = 2.576

    CIcols = [] ;  """"a list which stores the pair of min and max values for 
                    each feature in the dataframe"""

    for col in df.columns:
        length_of_columns = len(df[col])
        mean_value = statistics.mean(df[col])
        # Z value is the value required to find the range of confidence interval values
        # Z varies as per interval values.
        stddev = statistics.stdev(df[col])
        sqrt_Count = math.sqrt(length_of_columns)
        min_value = mean_value - Z[confidence]*stddev/sqrt_Count # boundaries of Confidence Interval
        max_value = mean_value + Z[confidence]*stddev/sqrt_Count
        CIcols.append([min_value, max_value])
    return CIcols


def isOverlapping(bugfree, buggy):
    if bugfree[1] < buggy[0]:
        return 0
    elif bugfree[0] > buggy[1]:
        return 0
    else:
        return 1


def save(usefulDf):
    path = workingDirectory + "\modified" # save in the modified subdirectory
    usefulDf.to_csv(os.path.join(path, file), index=False) # row numbers  not stored

workingDirectory = os.getcwd()      # gets the current working directory
directory = os.path.join(workingDirectory, "test_classInfo")     # concatenates
csvFiles = os.listdir(directory)    # list of all files in the directory

for file in csvFiles:
    df = pd.read_csv(directory + "\\" + file)
    df.rename(columns={df.columns[2]: "Alias"}, inplace=True)
    cols = df.columns.tolist()  # list of all columns in dataframe
    print(cols)
    bugs = cols[-1]             # target variable bugs is the last column in the dataframe

    df1 = df.iloc[:, 3:len(cols)]     # the names of projects removed from the original dataframe
    df_bugFree = df1[df1[bugs] == 0]  # new dataframe having target bugs as 0
    df_buggy = df1[df1[bugs] > 0]     # new dataframe having target bugs > 0

    CI_bugFree = calculate_Confidence(df_bugFree, 0.95) # a list which contains the CI for each

    CI_buggy = calculate_Confidence(df_buggy, 0.95)   # column of the dataframe passed(a list of pairs)
    usefulFeatures = []
    for i in range(0, len(CI_bugFree)):
        list1 = CI_bugFree[i]
        list2 = CI_buggy[i]
        if isOverlapping(list1, list2) == 0:
            usefulFeatures.append(cols[3+i])
    print('Useful features for ' + file + ' are: ')
    print(usefulFeatures)
    usefulDf = df.loc[:, usefulFeatures]  # new dataframe which has columns as useful features only
    save(usefulDf)  # save useful dataframe to csv
    print()
