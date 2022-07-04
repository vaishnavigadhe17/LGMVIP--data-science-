#!/usr/bin/env python
# coding: utf-8

# In[242]:


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
from sklearn import tree
from sklearn.tree import export_graphviz
import matplotlib.image as mpimg


# In[217]:


import csv

df= pd.read_csv(r"C:\Users\Hp\Downloads\bill_authentication.csv")

    


# In[218]:


df.values


# In[219]:


df.shape


# In[220]:


print(df.columns)


# In[221]:


print(df.describe())


# In[222]:


print(df.info())


# In[223]:


df.isnull().head()


# In[224]:


df.isnull().sum()


# In[225]:


df_copy=df.copy(deep=True)


# In[226]:


df_copy


# In[227]:


plt.style.use("seaborn")
p=df.hist(figsize=(20,20))


# In[228]:


df_copy['Variance'].fillna(df_copy['Variance'].mean(),inplace=True)
df_copy['Skewness'].fillna(df_copy['Skewness'].mean(),inplace=True)
df_copy['Curtosis'].fillna(df_copy['Curtosis'].mean(),inplace=True)
df_copy['Entropy'].fillna(df_copy['Entropy'].mean(),inplace=True)
df_copy['Class'].fillna(df_copy['Class'].mean(),inplace=True)


# In[229]:


p=df_copy.hist(figsize=(20,20))


# In[230]:


y = df.iloc[:,-1]
y


# In[231]:


plt.style.use("dark_background")
plt.figure(figsize=(15,10))
sns.scatterplot(data=df,s=100,alpha=0.7)
plt.grid()
plt.show()


# In[232]:


X= df.iloc[:,:-1]

X


# In[233]:


y = df.iloc[:,-1]
y


# In[234]:



    X_train,X_test,y_train,y_test= train_test_split(
    X,
    y,
    test_size=1/3,
)


# In[235]:


classifier= DecisionTreeClassifier(criterion="entropy",max_depth=4)
classifier.fit(X_train,y_train)


# In[236]:


y_pred = classifier.predict(X_test)
y_pred


# In[237]:


print("Accuracy Score : ",format(metrics.accuracy_score(y_test , y_pred)))


# In[238]:


matrix=confusion_matrix(y_test,y_pred)
matrix


# In[239]:


text_rep = tree.export_text(classifier)
print(text_rep)


# In[241]:


dot_data = StringIO()
filename = r"C:\Users\Hp\Desktop\ITR 4\download.png"
features= df.columns.tolist()[:-1] #column names

fig = plt.figure(figsize=(15,15))


img = mpimg.imread(filename)
plt.figure(figsize=(100, 200),dpi=100)
plt.imshow(img,interpolation='nearest')


# In[ ]:





# In[ ]:





# In[ ]:




