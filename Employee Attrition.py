#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('WA_Fn-UseC_-HR-Employee-Attrition.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import os
for dirname, _, filenames in os.walk('WA_Fn-UseC_-HR-Employee-Attrition.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[4]:


df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')
df.head(10)


# In[5]:


df.info()


# In[6]:


df.columns


# In[7]:


df.describe().T


# In[8]:


df.shape


# In[9]:


df.nunique()


# In[10]:


df.drop(['EmployeeCount','StandardHours','Over18','EmployeeNumber'], axis=1, inplace=True)


# In[11]:


columns = list(df.columns)
categorical = [data for data in columns if df[data].dtype=='object']
categorical


# In[12]:


for data in categorical:
    print(pd.crosstab(df[data],df['Attrition'],margins=True))
    print('------------------------------------------------')


# In[13]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[14]:


sns.countplot(x='Gender', hue='Attrition', data=df, palette='prism_r')
plt.show()


# In[15]:


sns.countplot(x='BusinessTravel', hue='Attrition', data=df, palette='prism_r')
plt.show()


# In[16]:


sns.countplot(x='MaritalStatus', hue='Attrition', data=df, palette='prism_r')
plt.show()


# In[17]:


sns.countplot(x='Department', hue='Attrition', data=df, palette='prism_r')
plt.show()


# In[18]:


plt.figure(figsize=(8,5))
sns.countplot(x='JobRole', hue='Attrition', data=df, palette='prism_r')
plt.xticks(rotation=45)
plt.show()


# In[19]:


sns.countplot(x='EducationField', hue='Attrition', data=df, palette='prism_r')
plt.xticks(rotation=45)
plt.show()


# In[20]:


sns.countplot(x='OverTime', hue='Attrition', data=df, palette='prism_r')
plt.show()


# In[21]:


plt.figure(figsize=(15,6))
sns.countplot(x='Age', hue='Attrition', data=df, palette='hot')
plt.show()


# In[22]:


plt.figure(figsize=(15,6))
sns.countplot(x='DistanceFromHome', hue='Attrition', data=df, palette='hot')
plt.show()


# In[23]:


sns.countplot(x='EnvironmentSatisfaction', hue='Attrition', data=df, palette='hot')
plt.show()


# In[24]:


sns.countplot(x='JobInvolvement', hue='Attrition', data=df, palette='hot')
plt.show()


# In[25]:


sns.countplot(x='JobLevel', hue='Attrition', data=df, palette='hot')
plt.show()


# In[26]:


sns.countplot(x='JobSatisfaction', hue='Attrition', data=df, palette='hot')
plt.show()


# In[27]:


sns.stripplot(data=df, x='MonthlyIncome', y='Attrition', palette='prism_r', hue='Attrition')
plt.show()


# In[28]:


sns.countplot(x='NumCompaniesWorked', hue='Attrition', data=df, palette='hot')
plt.show()


# In[29]:


sns.countplot(x='PercentSalaryHike', hue='Attrition', data=df, palette='hot')
plt.show()


# In[30]:


sns.countplot(x='PerformanceRating', hue='Attrition', data=df, palette='hot')
plt.show()


# In[31]:


sns.countplot(x='RelationshipSatisfaction', hue='Attrition', data=df, palette='hot')
plt.show()


# In[32]:


sns.countplot(x='StockOptionLevel', hue='Attrition', data=df, palette='hot')
plt.show()


# In[33]:


plt.figure(figsize=(15,6))
sns.countplot(x='TotalWorkingYears', hue='Attrition', data=df, palette='hot')
plt.show()


# In[34]:


sns.countplot(x='WorkLifeBalance', hue='Attrition', data=df, palette='hot')
plt.show()


# In[35]:


plt.figure(figsize=(15,6))
sns.countplot(x='YearsAtCompany', hue='Attrition', data=df, palette='hot')
plt.show()


# In[36]:


sns.countplot(x='YearsInCurrentRole', hue='Attrition', data=df, palette='hot')
plt.show()


# In[37]:


sns.countplot(x='YearsSinceLastPromotion', hue='Attrition', data=df, palette='hot')
plt.show()


# In[38]:


sns.countplot(x='YearsWithCurrManager', hue='Attrition', data=df, palette='hot')
plt.show()


# In[39]:


plt.figure(figsize=(20,10))
sns.heatmap(df.corr(),annot=True,fmt='.2f',linewidth='0.2')
plt.show()


# In[40]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Attrition"] = le.fit_transform(df.Attrition)


# In[41]:


X = df.drop('Attrition',1)
Y = df.Attrition


# In[42]:


columns_to_encode=[]
for i in X.columns:
    if df[i].nunique() < 20:
        columns_to_encode.append(i)

columns_to_encode


# In[43]:


X=pd.get_dummies(X, columns=columns_to_encode, drop_first=True)
X


# In[44]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8,random_state=42,stratify=Y)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)


# In[45]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_std,Y_train)


# In[46]:


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

lr_Y_test_pred = lr.predict(X_test_std)
lr_Y_train_pred = lr.predict(X_train_std)


# In[47]:


sns.heatmap(confusion_matrix(Y_train, lr_Y_train_pred), annot=True, fmt='d')
print(classification_report(Y_train, lr_Y_train_pred))
print('The accuracy score for the train data :', accuracy_score(Y_train, lr_Y_train_pred))


# In[48]:


sns.heatmap(confusion_matrix(Y_test, lr_Y_test_pred), annot=True, fmt='d')
print(classification_report(Y_test, lr_Y_test_pred))
print('The accuracy score for the test data :', accuracy_score(Y_test, lr_Y_test_pred))


# In[49]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier()
rf.fit(X_train, Y_train)


# In[50]:


rf_pred = rf.predict(X_test)
print(classification_report(Y_test, rf_pred))


# In[51]:


tuned_parameters = [{'max_depth': [10,50,100,200], 'n_estimators': [50,100,200,500], 'max_features': ['sqrt', 'auto']}] 
rf = RandomForestClassifier() 
rf_clf = GridSearchCV(rf, tuned_parameters, cv=5) 
rf_clf.fit(X_train, Y_train)


# In[52]:


rf = RandomForestClassifier(**rf_clf.best_params_) 
rf.fit( X_train, Y_train )


# In[53]:


rf_Y_train_pred = rf.predict(X_train)
rf_Y_test_pred = rf.predict(X_test)


# In[54]:


sns.heatmap(confusion_matrix(Y_train, rf_Y_train_pred), annot=True, fmt='d')
print(classification_report(Y_train, rf_Y_train_pred))
print('The accuracy score for the train data:', accuracy_score(Y_train, rf_Y_train_pred))


# In[55]:


sns.heatmap(confusion_matrix(Y_test, rf_Y_test_pred), annot=True, fmt='d')
print(classification_report(Y_test, rf_Y_test_pred))
print('The accuracy score for the test data:', accuracy_score(Y_test, rf_Y_test_pred))


# In[56]:


imp_features_rf = pd.DataFrame( { 'Features': X_train.columns, 'Importance': rf.feature_importances_ } )[:30]
imp_features_rf = imp_features_rf.sort_values('Importance', ascending = False) 
plt.figure(figsize=(10, 10))
sns.barplot( y = 'Features', x = 'Importance', data = imp_features_rf )
plt.show()


# In[57]:


from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train_std, Y_train)


# In[58]:


svm_pred = rf.predict(X_test_std)
print(classification_report(Y_test, svm_pred))


# In[59]:


parameter_tuning = [{'C': [1, 10, 100], 'kernel': ['linear']},
                    {'C': [1, 10, 100], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf']}]
svm = SVC()
svm_clf = GridSearchCV(svm, parameter_tuning, cv=5)
svm_clf.fit(X_train_std, Y_train)


# In[60]:


svm = SVC(**svm_clf.best_params_)
svm.fit(X_train_std, Y_train)


# In[61]:


svm_Y_train_pred = svm.predict(X_train_std)
svm_Y_test_pred = svm.predict(X_test_std)


# In[62]:


sns.heatmap(confusion_matrix(Y_train, svm_Y_train_pred), annot=True, fmt='d')
print(classification_report(Y_train, svm_Y_train_pred))
print('The accuracy score for the train data:', accuracy_score(Y_train, svm_Y_train_pred))


# In[63]:


sns.heatmap(confusion_matrix(Y_test, svm_Y_test_pred), annot=True, fmt='d')
print(classification_report(Y_test, svm_Y_test_pred))
print('The accuracy score for the test data:', accuracy_score(Y_test, svm_Y_test_pred))


# In[ ]:




