# mean ,
# median , var ,stddev ,
# min , max ,percentile , infogain , entropy ,
# pearson coorelation

import pandas as pd

def findMean(df) :
    ans = df.mean(axis = 0)
    return ans

def findMedian(df):
    ans = df.median()
    return ans
def findMinMax(df) :
    min = df.min()
    max = df.max()
    return min, max

def percentile(col,df,x):
    return df[col].quantile(x)

def findVar(df):
    return df.var()

def corr_p(df):
    return df.corr(method="pearson")

def findstd(df):
    return df.std()

def entropy(col , df):
    """
    Calculate the entropy of a dataset.
    The only parameter of this function is the target_col parameter which specifies the target column
    """
    target_col = df[col]
    elements, counts = np.unique(target_col, return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy


df = pd.read_csv('pmsm_temperature_data.csv')
mean = findMean(df)
median = findMedian(df)
min ,max = findMinMax(df)
var = findVar(df)
stdev = findstd(df)
corr = corr_p(df)

for col in df.columns:
    x = 0.9
    percentile_value  = percentile(col, df, x)
    print(percentile_value)

#for col in df.columns :
entropy('profile_id', df)


