import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


df=pd.read_csv('./data/kc_house_data_NAN.csv')

df.dtypes
df.columns
df.head()
#df= df.drop('id',axis=1)
df.drop(['Unnamed: 0','id'], axis=1, inplace=True)
df.describe()

print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan, mean, inplace=True)
df['bedrooms'].isnull().sum()

mean2=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan, mean2, inplace=True)
df['bathrooms'].isnull().sum()

df['floors'].value_counts().to_frame()

sns.boxplot(x='waterfront', y='price', data=df)

sns.regplot(x='sqft_above', y='price', data=df)

lm=LinearRegression()
y=df['price']
x=df[['sqft_above']]
lm.fit(x,y)
lm.coef_
lm.intercept_
lm.score(x,y)

df.corr()['price'].sort_values()
df[['price','sqft_above']].corr()
yhat=lm.predict(x)
ax1=sns.distplot(df['price'], hist=False, color ='r', label= 'Current Values')
sns.distplot(yhat,hist=False, color='g', label='New values', ax=ax1)

Z=df[['floors','waterfront','lat','bedrooms','sqft_basement','view','bathrooms', 'sqft_living15','sqft_above','grade','sqft_living']]
lm2=LinearRegression()
lm2.fit(Z,y)
lm2.coef_
lm2.intercept_
lm2.score(Z,y)
yhat2=lm2.predict(Z)
ax1=sns.distplot(df['price'], hist=False, color ='r', label= 'Current Values')
sns.distplot(yhat2,hist=False, color='g', label='New values', ax=ax1)

sns.residplot(x=Z, y=df['price'], data=df)
plt.ylim(0,)

Input =[('scale', StandardScaler()),('polynomial',PolynomialFeatures(degree=2)),('model',LinearRegression())]
pipe=Pipeline(Input)
pipe.fit(Z,y)
pipe.score(Z,y)
yhat3=pipe.predict(Z)
ax1=sns.distplot(df['price'], hist=False, color ='r', label= 'Current Values')
sns.distplot(yhat3,hist=False, color='g', label='New values', ax=ax1)


x_train, x_test, y_train, y_test=train_test_split(Z,y, test_size=0.15, random_state=1)
print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

RidgeModel=Ridge(alpha=0.1)
RidgeModel.fit(Z,y)
RidgeModel.score(Z,y)
yhat4=RidgeModel.predict(Z)
ax1=sns.distplot(df['price'], hist=False, color ='r', label= 'Current Values')
sns.distplot(yhat4,hist=False, color='g', label='New values', ax=ax1)

lr=LinearRegression()
pr=PolynomialFeatures(degree=2)
x_train_pr=pr.fit_transform(x_train)
x_test_pr=pr.fit_transform(x_test)
lr.fit(x_train_pr, y_train) 
lr.score(x_test_pr,y_test)
yhat5=lr.predict(x_test_pr)
ax1=sns.distplot(df['price'], hist=False, color ='r', label= 'Current Values')
sns.distplot(yhat5,hist=False, color='g', label='New values', ax=ax1)

RidgeModel2=Ridge(alpha=0.1)
RidgeModel2.fit(x_train_pr,y_train)
RidgeModel2.score(x_test_pr,y_test)
yhat6=RidgeModel.predict(x_train_pr)
