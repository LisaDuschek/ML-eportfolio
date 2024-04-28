import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from sklearn import ensemble, tree, linear_model
import missingno as msno


df = pd.read_csv("AB_NYC_2019.csv")

df.head()

numeric_features = df.select_dtypes(include = [np.number])
#categorical_features = df.select_dtypes(include = [np.object])

price_mean = np.mean(df["price"])

prices = df["price"].to_numpy()
monthly = df["reviews_per_month"].to_numpy()
last = df["last_review"].to_numpy()
total = df["number_of_reviews"].to_numpy()
ids = df["id"].to_numpy()

"""
fig, ax = plt.subplots()
data_line = ax.plot(ids, prices, label = "Prices", marker = 'o')
mean_line = ax.plot(price_mean, label = "Mean", linestyle = '--')
legend = ax.legend(loc = 'upper right')
plt.show()
"""
sdf = 0
for i in prices:
    sdf += (i - price_mean)**2

"""
msno.matrix(df.sample(250))
msno.heatmap(df)
msno.dendrogram(df)
plt.show()
"""

skewness = numeric_features.skew()
kurtosis = numeric_features.kurt()


y = df['availability_365']
"""
plt.figure(1); plt.title('Johnson SU')
sns.distplot(y, kde=False, fit=st.johnsonsu)
plt.figure(2); plt.title('Normal')
sns.distplot(y, kde=False, fit=st.norm)
plt.figure(3); plt.title('Log Normal')
sns.distplot(y, kde=False, fit=st.lognorm)
plt.show()
"""
correlation = numeric_features.corr()
print(correlation['price'].sort_values(ascending = False),'\n')

f, ax = plt.subplots()
plt.title('Correlation of numeric features with price', y = 1, size = 10)
sns.heatmap(correlation, square = True, vmax=0.9)
plt.show()

