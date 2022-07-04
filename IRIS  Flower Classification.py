#!/usr/bin/env python
# coding: utf-8

# In[325]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import accuracy_score


# In[326]:


import csv
df= pd.read_csv(r"C:\Users\Hp\Downloads\IRIS.csv")
df


# In[327]:


df.values


# In[328]:


print(df.columns)


# In[329]:


print(df.describe())


# In[330]:


print(df.info())


# In[331]:


df.isnull().head()


# In[332]:


df.isnull().sum()


# In[333]:


df_copy=df.copy(deep=True)


# In[334]:


df_copy


# In[335]:


df_copy[['sepal_length', 'sepal_width', 'petal_length', 'petal_width',
       'species']] =df_copy[['sepal_length', 'sepal_width', 'petal_length', 'petal_width',
       'species']].replace(0,np.NaN)
df_copy.isnull().sum()


# In[336]:


df.max()


# In[337]:


df.min()


# In[338]:


plt.style.use("seaborn")
p=df.hist(figsize=(20,20))


# In[339]:


df_copy['sepal_length'].fillna(df_copy['sepal_length'].mean(),inplace=True)
df_copy['sepal_width'].fillna(df_copy['sepal_width'].mean(),inplace=True)
df_copy['petal_length'].fillna(df_copy['petal_length'].mean(),inplace=True)
df_copy['petal_width'].fillna(df_copy['petal_width'].mean(),inplace=True)


# In[340]:


p=df_copy.hist(figsize=(20,20))


# In[341]:


y= df.iloc[:,4]
print(y)


# In[342]:


X= df.drop("species",axis=1)
X= df.iloc[:,1:5].values
X


# In[343]:


y=df.iloc[:,4].values

y


# In[344]:


print(X.shape)
print(y.shape)
print(X)
print(y)


# In[345]:


plt.figure(figsize=(10,7))
sns.heatmap(df.corr(),annot=True,cmap="seismic")
plt.show()


# In[346]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[347]:


df['species'] = le.fit_transform(df['species'])
df.head()


# In[348]:



X = df.drop(columns=['species'])
y = df['species']
X[:5]


# In[349]:


y[:5]


# In[350]:


from sklearn.model_selection   import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 1)


# In[351]:


from sklearn.linear_model  import LogisticRegression
from sklearn.neighbors  import KNeighborsClassifier
from sklearn.svm  import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree  import DecisionTreeClassifier
from sklearn.ensemble  import RandomForestClassifier
from sklearn.metrics  import accuracy_score


# In[352]:


lr = LogisticRegression()
knn = KNeighborsClassifier()
svm = SVC()
nv = GaussianNB()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()


# In[353]:


df.iloc[5]
#it will display records only with species "Iris-setosa".
df.loc[df["species"] == "Iris-setosa"]


# In[354]:


g = sns.pairplot(df,hue="species")


# In[355]:


sum_df = df["sepal_length"].sum()
mean_df = df["sepal_length"].mean()
median_df = df["sepal_length"].median()
 
print("Sum:",sum_df, "\nMean:", mean_df, "\nMedian:",median_df)


# In[356]:


sum_df = df["sepal_width"].sum()
mean_df = df["sepal_width"].mean()
median_df = df["sepal_width"].median()
 
print("Sum:",sum_df, "\nMean:", mean_df, "\nMedian:",median_df)


# In[357]:


sum_df = df["petal_length"].sum()
mean_df = df["petal_length"].mean()
median_df = df["petal_length"].median()
 
print("Sum:",sum_df, "\nMean:", mean_df, "\nMedian:",median_df)


# In[358]:


sum_df = df[""petal_width"].sum()
mean_df = df["petal_width"].mean()
median_df = df["petal_width"].median()
 
print("Sum:",sum_df, "\nMean:", mean_df, "\nMedian:",median_df)


# In[ ]:





# In[ ]:




