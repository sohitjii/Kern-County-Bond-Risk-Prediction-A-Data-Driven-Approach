#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot,pylab as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder,StandardScaler ,MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LinearRegression,Ridge,Lasso
import seaborn as sns
from scipy.stats import zscore
from sklearn.metrics import r2_score,roc_auc_score,adjusted_rand_score
import statsmodels.formula.api as sm
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


# In[2]:



df=pd.read_csv("Kern_County bond .csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.size


# In[5]:


df.info()


# In[6]:


df.describe ()


# In[7]:


df.nunique()


# In[8]:


df.isnull().sum()


# In[9]:


df["ADTR Reportable Next Fiscal Year"] = df["ADTR Reportable Next Fiscal Year"].fillna(df["ADTR Reportable Next Fiscal Year"].mode().values[0])
df["ADTR Last Reported Year"] = df["ADTR Last Reported Year"].fillna(df["ADTR Last Reported Year"].mode().values[0])
df["Debt Policy"] = df["Debt Policy"].fillna(df["Debt Policy"].mode().values[0])
df["MKR CDIAC Number"] = df["MKR CDIAC Number"].fillna(df["MKR CDIAC Number"].mode().values[0])
df["Project Name"] = df["Project Name"].fillna(df["Project Name"].mode().values[0])
df["Environmental/Social Impact Bonds"] = df["Environmental/Social Impact Bonds"].fillna(df["Environmental/Social Impact Bonds"].mode().values[0])
df['Purpose'] = df['Purpose'].fillna(df['Purpose'].mode().values[0])
df['Source of Repayment'] = df['Source of Repayment'].fillna(df['Source of Repayment'].mode().values[0])
df['Interest Type'] = df['Interest Type'].fillna(df['Interest Type'].mode().values[0])
df['Other Interest Type'] = df['Other Interest Type'].fillna(df['Other Interest Type'].mode().values[0])
df['First Optional Call Date'] = df['First Optional Call Date'].fillna(df['First Optional Call Date'].mode().values[0])
df['Final Maturity Date'] = df['Final Maturity Date'].fillna(df['Final Maturity Date'].mode().values[0])
df['CAB Flag'] = df['CAB Flag'].fillna(df['CAB Flag'].mode().values[0])
df['S and P Rating'] = df['S and P Rating'].fillna(df['S and P Rating'].mode().values[0])
df['Moody Rating'] = df['Moody Rating'].fillna(df['Moody Rating'].mode().values[0])
df['Fitch Rating'] = df['Fitch Rating'].fillna(df['Fitch Rating'].mode().values[0])
df['Other Rating'] = df['Other Rating'].fillna(df['Other Rating'].mode().values[0])
df['Guarantor Flag'] = df['Guarantor Flag'].fillna(df['Guarantor Flag'].mode().values[0])
df['Guarantor'] = df['Guarantor'].fillna(df['Guarantor'].mode().values[0])
df['Underwriter'] = df['Underwriter'].fillna(df['Underwriter'].mode().values[0])
df['Purchaser'] = df['Purchaser'].fillna(df['Purchaser'].mode().values[0])
df['Placement Agent'] = df['Placement Agent'].fillna(df['Placement Agent'].mode().values[0])
df['Financial Advisor'] = df['Financial Advisor'].fillna(df['Financial Advisor'].mode().values[0])
df['Bond Counsel'] = df['Bond Counsel'].fillna(df['Bond Counsel'].mode().values[0])
df['Disclosure Counsel'] = df['Disclosure Counsel'].fillna(df['Disclosure Counsel'].mode().values[0])
df['Co-Financial Advisor'] = df['Co-Financial Advisor'].fillna(df['Co-Financial Advisor'].mode())
df['Trustee'] = df['Trustee'].fillna(df['Trustee'].mode().values[0])





# In[10]:


df['Net Issue Discount/Premium'] = df['Net Issue Discount/Premium'].fillna(df['Net Issue Discount/Premium'].mean())
df['Refunding Amount'] = df['Refunding Amount'].fillna(df['Refunding Amount'].mean())
df['Issue Costs Pct of Principal Amt'] = df['Issue Costs Pct of Principal Amt'].fillna(df['Issue Costs Pct of Principal Amt'].mean())
df['Total Issuance Costs'] = df['Total Issuance Costs'].fillna(df['Total Issuance Costs'].mean())
df['Lender'] = df['Lender'].fillna(df['Lender'].mean())
df['NIC Interest Rate'] = df['NIC Interest Rate'].fillna(df['NIC Interest Rate'].mean())
df['TIC Interest Rate'] = df['TIC Interest Rate'].fillna(df['TIC Interest Rate'].mean())


# In[11]:


df.isnull().sum()


# In[12]:


df.drop(columns=["CDIAC Number","Issuer","MKR CDIAC Number","Fitch Rating","Other Rating","Lender","Co-Financial Advisor","Co-Bond Counsel","Borrower Counsel","Issuer County","TIC Interest Rate","NIC Interest Rate","Net Issue Discount/Premium","Sale Date","Project Name","Trustee","ADTR Reportable Next Fiscal Year"],inplace=True)


# In[13]:


df


# In[14]:


df[df.duplicated()]


# In[15]:


df.duplicated().sum()


# In[16]:


df.drop_duplicates(inplace=True)


# In[17]:


df.duplicated().sum()


# Detecting the outliers

# In[18]:


plt.figure(figsize=(30,30))
plt.subplot(3,2,1)
sns.boxplot(df["Refunding Amount"])
plt.subplot(3,2,2)
sns.boxplot(df["New Money"])
plt.subplot(3,2,3)
sns.boxplot(df["Principal Amount"])


# In[19]:


plt.figure(figsize=(30,30))
plt.subplot(2,2,1)
sns.boxplot(df["Total Issuance Costs"])
plt.subplot(2,2,2)
sns.boxplot(df["Issue Costs Pct of Principal Amt"])


# Treating the outliers

# In[20]:


mean = df['Refunding Amount'].mean()
median = df['Refunding Amount'].median()
df['Refunding Amount'] = np.where(df['Refunding Amount'] > median,mean, df['Refunding Amount'])


# In[21]:


mean = df['New Money'].mean()
median = df['New Money'].median()
df['New Money'] = np.where(df['New Money'] > median,mean, df['New Money'])


# In[22]:


mean = df['Principal Amount'].mean()
median = df['Principal Amount'].median()
df['Principal Amount'] = np.where(df['Principal Amount'] > median,mean, df['Principal Amount'])


# In[23]:


mean = df['Total Issuance Costs'].mean()
median = df['Total Issuance Costs'].median()
df['Total Issuance Costs'] = np.where(df['Total Issuance Costs'] > median,mean, df['Total Issuance Costs'])


# In[24]:


mean = df['Issue Costs Pct of Principal Amt'].mean()
median = df['Issue Costs Pct of Principal Amt'].median()
df['Issue Costs Pct of Principal Amt'] = np.where(df['Issue Costs Pct of Principal Amt'] > median,mean, df['Issue Costs Pct of Principal Amt'])


# In[25]:


plt.figure(figsize=(30,30))
plt.subplot(3,2,1)
sns.boxplot(df["Refunding Amount"])
plt.subplot(3,2,2)
sns.boxplot(df["New Money"])
plt.subplot(3,2,3)
sns.boxplot(df["Principal Amount"])


# In[26]:


plt.figure(figsize=(30,30))
plt.subplot(2,2,1)
sns.boxplot(df["Total Issuance Costs"])
plt.subplot(2,2,2)
sns.boxplot(df["Issue Costs Pct of Principal Amt"])


# data visualization
# 

# In[27]:


sns.pairplot(df)


# In[28]:


plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.countplot("Issuance Documents",hue="ADTR Report",data=df)
plt.subplot(2,2,2)
sns.countplot("Issuance Documents",hue="ADTR Reportable",data=df)


# In[29]:


plt.figure(figsize=(10,10))
plt.subplot(2,2,1)
sns.countplot(df["MKR Authority"])
plt.subplot(2,2,2)
sns.countplot(df["Local Obligation"])


# In[30]:


plt.figure(figsize=(10,5))
sns.countplot(df["Issuer Group"])


# In[31]:


plt.figure(figsize=(60,20))
sns.countplot(df["S and P Rating"])


# In[32]:


plt.figure(figsize=(50,10))
sns.barplot("S and P Rating","Principal Amount",data=df,ci=True)


# In[33]:


plt.figure(figsize=(10,5))
sns.countplot(df["CAB Flag"])


# In[34]:


import matplotlib.pyplot as plt
import seaborn as sns

columns = ["Principal Amount", "Total Issuance Costs", "S and P Rating"]

plt.figure(figsize=(70, 30))

for i, col in enumerate(columns, 1):
    plt.subplot(2, 2, i)
    if col == "S and P Rating":
        sns.histplot(df[col], kde=True)
    else:
        sns.distplot(df[col])

plt.tight_layout()
plt.show()


# In[35]:


columns_to_encode = [
    "Issuance Documents", "Sold Status", "ADTR Report", "ADTR Filing Status", 
    "ADTR Reportable", "ADTR Last Reported Year", "Debt Policy", "MKR Authority", 
    "Local Obligation", "Issuer Group", "Issuer Type", "Environmental/Social Impact Bonds", 
    "Debt Type", "Purpose", "Source of Repayment", "Interest Type", "Other Interest Type", 
    "Federally Taxable", "CAB Flag", "S and P Rating", "Moody Rating", "Guarantor Flag", 
    "Guarantor", "Sale Type (Comp/Neg)", "Private Placement Flag"]

le = LabelEncoder()

for col in columns_to_encode:
    df[col] = le.fit_transform(df[col])


# In[36]:


ohe=pd.get_dummies(df,columns=["Underwriter","Total Issuance Costs","Disclosure Counsel","Bond Counsel","Financial Advisor","Placement Agent","Purchaser","Final Maturity Date","First Optional Call Date"])
df=ohe
df


# In[37]:


df.nunique()


# In[39]:


df.corr()


# In[ ]:





# In[40]:


x=df.drop("S and P Rating",axis=1)
y=df["S and P Rating"]


# In[41]:


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.750,random_state=100)


# In[42]:


mean = df.mean()
std = df.std()
df = (df - mean) / std
df


# 1. linear regression model:

# In[43]:


lr= LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)


# In[44]:


r2=r2_score(y_test,y_pred)
r2


# In[45]:


lr.get_params()


# In[46]:


params={"n_jobs":[5,20,30],"positive":[True],"fit_intercept":[False],"copy_X":[False,True]}
lrt=GridSearchCV(lr,params,cv=7,verbose=1,scoring='r2')
lrt.fit(x_train,y_train)


# In[47]:


lrt.best_score_


# In[48]:


lrt.best_params_


# 2. ridge regression:

# In[49]:


ridge=Ridge()
ridge.get_params()
ridge.fit(x_train,y_train)
y_pred_ridge=ridge.predict(x_test)


# In[50]:


r_2=r2_score(y_test,y_pred_ridge)
r_2


# In[51]:


ridge.get_params()


# In[52]:


params={"max_iter":[3,5,20],"positive":[True,False],"fit_intercept":[True,False],"copy_X":[True,False],"random_state":[10,20,30,45],"tol":[0.0003,0.0005,0.0002,0.0001],"alpha":[1.0,2.0,4.0,5.0]}
ridgetu=GridSearchCV(ridge,params,cv=4,verbose=1,n_jobs=20,scoring='r2')
ridgetu.fit(x_train,y_train)


# In[53]:


ridgetu.best_score_


# In[54]:


ridgetu.best_params_


# 3. lasso regression:

# In[55]:


lasso=Lasso()
lasso.fit(x_train,y_train)
y_pred_lasso = lasso.predict(x_test)


# In[56]:


r_2=r2_score(y_test,y_pred_lasso)
r_2


# In[57]:


lasso.get_params()


# In[58]:


params={"alpha":[1.0,2.0,3.0,4.0],"max_iter":[100,500,1000],"random_state":[30,45],"fit_intercept":[True,False],"copy_X":[True,False],"positive":[True,False],"precompute":[True,False],"tol":[0.0003,0.0005,0.0002,0.0001]}
lass=GridSearchCV(lasso,params,cv=4,n_jobs=20,scoring='r2')
lass.fit(x_train,y_train)


# In[59]:


lass.best_score_


# In[60]:


lass.best_params_


# 4. adaboost regressor:

# In[61]:


ada=AdaBoostRegressor()
ada.fit(x_train,y_train)
y_pred_adb = ada.predict(x_test)


# In[62]:


r_2=r2_score(y_test,y_pred_adb)
r_2


# In[63]:


ada.get_params()


# In[64]:


params={"learning_rate":[1.0,2.0,3.0,4.0],"random_state":[30,45,60,64],'n_estimators':[3,5,10,20,30,40,50,70]}
adat=GridSearchCV(ada,params,cv=3,n_jobs=5,verbose=1,scoring='r2')
adat.fit(x_train,y_train)


# In[65]:


adat.best_score_


# In[66]:


adat.best_params_


# 5. decision tree regressor:

# In[67]:


dtcr=DecisionTreeRegressor()
dtcr.fit(x_train,y_train)
y_pred_dtcr=dtcr.predict(x_test)


# In[68]:


r2=r2_score(y_test,y_pred_dtcr)
r2


# In[69]:


dtcr.get_params()


# In[70]:


params={"max_depth":[10,20,30,40],"random_state":[20,40],'min_impurity_decrease':[1,2,3,4],"min_samples_leaf":[1,2,3,4,5],"min_samples_split":[2,3,4,5,6],"ccp_alpha":[0.5,0.6,0.7]}
dtcrt=GridSearchCV(dtcr,params,cv=5,n_jobs=5,verbose=1,scoring='r2')
dtcrt.fit(x_train,y_train)


# In[71]:


dtcrt.best_params_


# In[72]:


dtcrt.best_score_


# 6. random forest regressor

# In[73]:


rfr= RandomForestRegressor()
rfr.fit(x_train,y_train)
y_pred_rf=rfr.predict(x_test)


# In[74]:


r2=r2_score(y_test,y_pred_rf)
r2


# In[75]:


rfr.get_params()


# In[76]:


params={"max_depth":[100,200],"min_samples_split":[3,5],"min_impurity_decrease":[0.5,1.0],
        "n_estimators":[200,500],"random_state":[10,20,40],"bootstrap":[True],"oob_score":[True]}
rfrt=GridSearchCV(rfr,params,cv=5,n_jobs=20,verbose=1,scoring="r2")
rfrt.fit(x_train,y_train)


# In[77]:


rfrt.best_params_


# In[78]:


rfrt.best_score_


# Accuracy of all the models:
# 
# 
# linear regression = 0.21
# Ridge regression = 0.61
# lasso regression = 0.48
# Adaboost regression = 0.66
# decision tree regressor = 0.68
# Random forest regressor = 0.72 

# In[ ]:




