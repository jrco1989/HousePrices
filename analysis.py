import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


df=pd.read_csv('./kc_house_data.csv')

df.dtypes

df.columns

df= df.drop('id',axis=1)

df.describe()

df['floors'].value_counts().to_frame()
sns.boxplot(x='waterfront', y='price', data=df)

sns.regplot(x='sqft_above', y='price', data=df)
