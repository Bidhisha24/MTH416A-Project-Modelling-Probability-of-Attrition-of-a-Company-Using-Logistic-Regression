#!/usr/bin/env python
# coding: utf-8

# # HR Analytics Project

# In[1]:


pip install imbalanced-learn==0.6.0


# In[2]:


pip install scikit-learn


# In[3]:


pip install scipy


# ### Import packages

# In[4]:


import numpy as np
import pandas as pd


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[6]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_curve
from imblearn.over_sampling import SMOTE 


# In[7]:


import warnings
warnings.filterwarnings('ignore')


# ### Import Data files

# In[8]:


in_time=pd.read_csv("C:/Users/USER/Desktop/MTH416A Project/HR Dataset/in_time.csv")
manager_survey_data=pd.read_csv("C:/Users/USER/Desktop/MTH416A Project/HR Dataset/manager_survey_data.csv")
employee_survey_data = pd.read_csv('C:/Users/USER/Desktop/MTH416A Project/HR Dataset/employee_survey_data.csv')
data_dictionary= pd.read_excel('C:/Users/USER/Desktop/MTH416A Project/HR Dataset/data_dictionary.xlsx')
out_time = pd.read_csv('C:/Users/USER/Desktop/MTH416A Project/HR Dataset/out_time.csv')
general_data = pd.read_csv('C:/Users/USER/Desktop/MTH416A Project/HR Dataset/general_data.csv')


# ### Clean in_time

# In[9]:


pd.set_option("display.max_columns", None)
in_time.head()


# In[10]:


in_time.shape


# In[11]:


in_time=in_time.replace(np.nan,0)
in_time


# In[12]:


in_time.iloc[:,1:]=in_time.iloc[:,1:].apply(pd.to_datetime,errors='coerce')
in_time


# ### Clean out_time

# In[13]:


out_time=out_time.replace(np.nan,0)
out_time.iloc[:,1:]=out_time.iloc[:,1:].apply(pd.to_datetime,errors='coerce')
out_time


# ### Find average in and out time difference

# In[14]:


time=in_time.append(out_time)
time.shape


# In[15]:


time=time.diff(periods=4410)
time=time.iloc[4410:]
time.reset_index(inplace=True)
time


# In[16]:


time.drop(['index','Unnamed: 0'],axis=1,inplace=True)
time.drop(['2015-01-01', '2015-01-14','2015-01-26','2015-03-05',
          '2015-05-01','2015-07-17','2015-09-17','2015-10-02',
            '2015-11-09','2015-11-10','2015-11-11','2015-12-25'], axis = 1,inplace=True)
time


# In[17]:


time['Actual Time']=time.mean(axis=1)
time['hrs']=time['Actual Time']/np.timedelta64(1, 'h')
time.reset_index(inplace=True)
time.head()


# In[18]:


time.drop(time.columns.difference(['index','hrs']), axis=1,inplace=True)
time.rename(columns={'index':'EmployeeID'},inplace=True)
time['EmployeeID']=time['EmployeeID']+1
time


# In[19]:


hr_data = pd.merge(time, manager_survey_data, how='inner', on='EmployeeID')
hr_data = pd.merge(hr_data, employee_survey_data, how='inner', on='EmployeeID')
hr_data = pd.merge(hr_data, general_data, how='inner', on='EmployeeID')
hr_data


# ## drop variables with constant value

# In[20]:


counts = []
for col in hr_data.columns:
    tmp = hr_data[col].value_counts()
    tmp.name = col
    counts.append(tmp)
hr_data_value_counts = pd.concat(counts, axis=1)
hr_data_value_counts


# In[21]:


cols=hr_data_value_counts.columns
search=4410
throw=['EmployeeID']

for i in cols:
    if (hr_data_value_counts[i] == search).any():
        throw.append(i)

throw


# In[22]:


hr_data.drop(throw, axis=1, inplace=True)
hr_data


# In[23]:


hr_data["BusinessTravel"].value_counts()


# In[24]:


hr_data["Gender"].value_counts()


# In[25]:


hr_data["Education"].value_counts()


# In[26]:


hr_data["Department"].value_counts()


# In[27]:


hr_data["MaritalStatus"].value_counts()


# ### Inspecting the Dataframe and computing Missing Values

# In[28]:


hr_data.describe()


# In[29]:


hr_data.info()


# In[30]:


missing=[]
percent_missing=[]
for x in hr_data.columns:
    if hr_data[x].isnull().sum()!=0:
        missing.append(x)
        percent_missing.append(hr_data[x].isnull().sum()*100/4410)
d={"variable":missing, "Percentage of missing value":percent_missing}
missing_data=pd.DataFrame(d)
missing_data


# In[31]:


hr_data['EnvironmentSatisfaction'].value_counts(ascending=False)


# In[32]:


sns.countplot(x='EnvironmentSatisfaction',data=hr_data);


# In[33]:


sns.boxplot(x='EnvironmentSatisfaction',data=hr_data);


# In[34]:


hr_data['EnvironmentSatisfaction'].mean()


# In[35]:


hr_data['EnvironmentSatisfaction'].median()


# In[36]:


hr_data['EnvironmentSatisfaction'] = hr_data['EnvironmentSatisfaction'].fillna(3)
hr_data['EnvironmentSatisfaction'].isnull().sum()


# In[37]:


hr_data['JobSatisfaction'].value_counts(ascending=False)


# In[38]:


sns.countplot(x='JobSatisfaction',data=hr_data);


# In[39]:


sns.boxplot(x='JobSatisfaction',data=hr_data);


# In[40]:


hr_data['JobSatisfaction'].mean()


# In[41]:


hr_data['JobSatisfaction'].median()


# In[42]:


hr_data['JobSatisfaction'] = hr_data['JobSatisfaction'].fillna(3)
hr_data['JobSatisfaction'].isnull().sum()


# In[43]:


hr_data['WorkLifeBalance'].value_counts(ascending=False)


# In[44]:


sns.countplot(x='WorkLifeBalance',data=hr_data);


# In[45]:


sns.boxplot(x='WorkLifeBalance',data=hr_data);


# In[46]:


hr_data['WorkLifeBalance'].mean()


# In[47]:


hr_data['WorkLifeBalance'].median()


# In[48]:


hr_data['WorkLifeBalance'] = hr_data['WorkLifeBalance'].fillna(3)
hr_data['WorkLifeBalance'].isnull().sum()


# In[49]:


hr_data['NumCompaniesWorked'].value_counts(ascending=False)


# In[50]:


sns.countplot(x='NumCompaniesWorked',data=hr_data)


# In[51]:


sns.boxplot(x='NumCompaniesWorked',data=hr_data)


# In[52]:


hr_data['NumCompaniesWorked'].mean()


# In[53]:


hr_data['NumCompaniesWorked'].median()


# In[54]:


hr_data['NumCompaniesWorked'] = hr_data['NumCompaniesWorked'].fillna(2)
hr_data['NumCompaniesWorked'].isnull().sum()


# In[55]:


hr_data['TotalWorkingYears'].value_counts(ascending=False)


# In[56]:


plt.figure(figsize=(8,8))
fig1 = sns.distplot(hr_data['TotalWorkingYears'], hist=True, hist_kws={'edgecolor':'black'},kde_kws={'color':'darkblue'})
fig1.set_xlabel('Total Working Years')
fig1.set_ylabel('Number of Employees')


# In[57]:


sns.boxplot(x='TotalWorkingYears',data=hr_data);


# In[58]:


hr_data['TotalWorkingYears'].mean()


# In[59]:


hr_data['TotalWorkingYears'].median()


# In[60]:


hr_data['TotalWorkingYears'] = hr_data['TotalWorkingYears'].fillna(10)
hr_data['TotalWorkingYears'].isnull().sum()


# In[61]:


hr_data.info()


# ### EDA

# In[62]:


plt.figure(figsize=(8,8))
fig = sns.countplot(x='JobSatisfaction', data=hr_data, hue="Attrition")
fig.set_ylabel('Number of Employee')
bars = fig.patches
half = int(len(bars)/2)
left_bars = bars[:half]
right_bars = bars[half:]

for left, right in zip(left_bars, right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    total = height_l + height_r

    fig.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")
    fig.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")


# In[63]:


plt.figure(figsize=(8,8))
fig2 = sns.countplot(x='EnvironmentSatisfaction',data=hr_data,hue="Attrition")
bars = fig2.patches
half = int(len(bars)/2)
left_bars = bars[:half]
right_bars = bars[half:]
for left, right in zip(left_bars, right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    total = height_l + height_r
    fig2.text(left.get_x() + left.get_width()/2, height_l + 30, '{0:.0%}'.format(height_l/total), ha="center")
    fig2.text(right.get_x() + right.get_width()/2, height_r + 30, '{0:.0%}'.format(height_r/total), ha="center")


# In[64]:


plt.figure(figsize=(8,8))
fig2 = sns.countplot(x='WorkLifeBalance',data=hr_data,hue="Attrition")
bars = fig2.patches
half = int(len(bars)/2)
left_bars = bars[:half]
right_bars = bars[half:]
for left, right in zip(left_bars, right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    total = height_l + height_r
    fig2.text(left.get_x() + left.get_width()/2, height_l + 30, '{0:.0%}'.format(height_l/total), ha="center")
    fig2.text(right.get_x() + right.get_width()/2, height_r + 30, '{0:.0%}'.format(height_r/total), ha="center")


# In[65]:


plt.figure(figsize=(8,8))
fig3 = sns.countplot(x='PerformanceRating', data=hr_data, hue="Attrition")
fig3.set_ylabel('# of Employee')
bars = fig2.patches
half = int(len(bars)/2)
left_bars = bars[:half]
right_bars = bars[half:]

for left, right in zip(left_bars, right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    total = height_l + height_r

    fig2.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")
    fig2.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")


# In[66]:


plt.figure(figsize=(8,8))
sns.violinplot(y='Age',x='Attrition',data=hr_data)
plt.show()


# In[67]:


plt.figure(figsize=(8,8))
fig = sns.countplot(x='BusinessTravel', data=hr_data, hue="Attrition")
fig.set_ylabel('# of Employee')
bars = fig.patches
half = int(len(bars)/2)
left_bars = bars[:half]
right_bars = bars[half:]

for left, right in zip(left_bars, right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    total = height_l + height_r

    fig.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")
    fig.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")


# In[68]:


plt.figure(figsize=(8,8))
fig = sns.countplot(x='Department', data=hr_data, hue="Attrition")
fig.set_ylabel('# of Employee')
bars = fig.patches
half = int(len(bars)/2)
left_bars = bars[:half]
right_bars = bars[half:]

for left, right in zip(left_bars, right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    total = height_l + height_r

    fig.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")
    fig.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")


# In[69]:


plt.figure(figsize=(8,8))
sns.violinplot(y='DistanceFromHome',x='Attrition',data=hr_data)

plt.show()


# In[70]:


plt.figure(figsize=(8,8))
fig = sns.countplot(x='Education', data=hr_data, hue="Attrition")
fig.set_ylabel('# of Employee')
bars = fig.patches
half = int(len(bars)/2)
left_bars = bars[:half]
right_bars = bars[half:]

for left, right in zip(left_bars, right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    total = height_l + height_r

    fig.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")
    fig.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")


# In[71]:


plt.figure(figsize=(15,8))
fig = sns.countplot(x='EducationField', data=hr_data, hue="Attrition")
fig.set_ylabel('# of Employee')
bars = fig.patches
half = int(len(bars)/2)
left_bars = bars[:half]
right_bars = bars[half:]

for left, right in zip(left_bars, right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    total = height_l + height_r

    fig.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")
    fig.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")


# In[72]:


plt.figure(figsize=(8,8))
fig = sns.countplot(x='Gender', data=hr_data, hue="Attrition")
fig.set_ylabel('# of Employee')
bars = fig.patches
half = int(len(bars)/2)
left_bars = bars[:half]
right_bars = bars[half:]

for left, right in zip(left_bars, right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    total = height_l + height_r

    fig.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")
    fig.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")


# In[73]:


plt.figure(figsize=(8,8))
fig = sns.countplot(x='JobLevel', data=hr_data, hue="Attrition")
fig.set_ylabel('# of Employee')
bars = fig.patches
half = int(len(bars)/2)
left_bars = bars[:half]
right_bars = bars[half:]

for left, right in zip(left_bars, right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    total = height_l + height_r

    fig.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")
    fig.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")


# In[74]:


plt.figure(figsize=(20,8))
fig = sns.countplot(x='JobRole', data=hr_data, hue="Attrition")
fig.set_ylabel('# of Employee')
bars = fig.patches
half = int(len(bars)/2)
left_bars = bars[:half]
right_bars = bars[half:]

for left, right in zip(left_bars, right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    total = height_l + height_r

    fig.text(left.get_x() + left.get_width()/2., height_l + 20, '{0:.0%}'.format(height_l/total), ha="center")
    fig.text(right.get_x() + right.get_width()/2., height_r + 20, '{0:.0%}'.format(height_r/total), ha="center")


# In[75]:


plt.figure(figsize=(20,8))
fig = sns.countplot(x='MaritalStatus', data=hr_data, hue="Attrition")
fig.set_ylabel('# of Employee')
bars = fig.patches
half = int(len(bars)/2)
left_bars = bars[:half]
right_bars = bars[half:]

for left, right in zip(left_bars, right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    total = height_l + height_r

    fig.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")
    fig.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")


# In[76]:


plt.figure(figsize=(8,8))
sns.violinplot(y='MonthlyIncome',x='Attrition',data=hr_data)

plt.show()


# In[77]:


plt.figure(figsize=(8,8))
sns.violinplot(y='PercentSalaryHike',x='Attrition',data=hr_data)

plt.show()


# In[78]:


plt.figure(figsize=(20,8))
fig = sns.countplot(x='StockOptionLevel', data=hr_data, hue="Attrition")
fig.set_ylabel('# of Employee')
bars = fig.patches
half = int(len(bars)/2)
left_bars = bars[:half]
right_bars = bars[half:]

for left, right in zip(left_bars, right_bars):
    height_l = left.get_height()
    height_r = right.get_height()
    total = height_l + height_r

    fig.text(left.get_x() + left.get_width()/2., height_l + 40, '{0:.0%}'.format(height_l/total), ha="center")
    fig.text(right.get_x() + right.get_width()/2., height_r + 40, '{0:.0%}'.format(height_r/total), ha="center")


# In[79]:


plt.figure(figsize=(8,8))
sns.violinplot(y='TotalWorkingYears',x='Attrition',data=hr_data)

plt.show()


# In[80]:


plt.figure(figsize=(8,8))
sns.violinplot(y='TrainingTimesLastYear',x='Attrition',data=hr_data)

plt.show()


# In[81]:


plt.figure(figsize=(8,8))
sns.violinplot(y='YearsAtCompany',x='Attrition',data=hr_data)

plt.show()


# In[82]:


plt.figure(figsize=(8,8))
sns.violinplot(y='YearsSinceLastPromotion',x='Attrition',data=hr_data)

plt.show()


# In[83]:


plt.figure(figsize=(8,8))
sns.violinplot(y='YearsWithCurrManager',x='Attrition',data=hr_data)

plt.show()


# In[84]:


plt.figure(figsize=(8,8))
sns.violinplot(y='hrs',x='Attrition',data=hr_data)

plt.show()


# In[85]:


plt.figure(figsize=(20,18))
sns.heatmap(hr_data.corr(), annot = True, cmap="Oranges");


# In[86]:


hr_num=hr_data[[ 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike',
       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
           'DistanceFromHome','Age','hrs']]

sns.pairplot(hr_num, diag_kind='kde')
plt.show()


# In[87]:


hr_data.info()


# ## Separate the Continuous variables

# In[88]:


cont_var=hr_data.columns.difference(['JobInvolvement', 'PerformanceRating', 'EnvironmentSatisfaction',
                                 'JobSatisfaction', 'WorkLifeBalance','BusinessTravel', 'Department',
                                 'Education','EducationField', 'Gender', 'JobLevel', 'JobRole',
                                 'MaritalStatus'])
cont_var=cont_var.drop('Attrition')
cont_var


# In[89]:


hr_data


# ## Create Dummy Variable

# In[90]:


hr_data['JobLevel']=hr_data['JobLevel'].astype('object') 


# In[91]:


hr_data['Education'] = hr_data['Education'].replace({ 1 : 'Below College', 2: 'College',3: 'Bachelor',4: 'Master',5 : 'Doctor'})
hr_data['EnvironmentSatisfaction'] = hr_data['EnvironmentSatisfaction'].replace({ 1 : 'Low', 2: 'Medium',3: 'High',4: 'Very High'})
hr_data['JobInvolvement'] = hr_data['JobInvolvement'].replace({ 1 : 'Low', 2: 'Medium',3: 'High',4: 'Very High'})
hr_data['JobSatisfaction'] = hr_data['JobSatisfaction'].replace({ 1 : 'Low', 2: 'Medium',3: 'High',4: 'Very High'})
hr_data['PerformanceRating'] = hr_data['PerformanceRating'].replace({ 1 : 'Low', 2: 'Good',3: 'Excellent',4: 'Outstanding'})
hr_data['WorkLifeBalance'] = hr_data['WorkLifeBalance'].replace({ 1 : 'Bad', 2: 'Good',3: 'Better',4: 'Best'})


# In[92]:


hr_data


# In[93]:


dummy = pd.get_dummies(hr_data[['JobInvolvement', 'PerformanceRating', 'EnvironmentSatisfaction',
                                 'JobSatisfaction', 'WorkLifeBalance','BusinessTravel', 'Department',
                                 'Education','EducationField', 'Gender', 'JobLevel', 'JobRole',
                                 'MaritalStatus']], drop_first=True)
hr_data = pd.concat([hr_data, dummy], axis=1)


# In[94]:


hr_data


# ## Drop the variables of which we created dummies

# In[95]:


hr_data = hr_data.drop(['JobInvolvement', 'PerformanceRating', 'EnvironmentSatisfaction',
                                 'JobSatisfaction', 'WorkLifeBalance','BusinessTravel', 'Department',
                                 'Education','EducationField', 'Gender', 'JobLevel', 'JobRole',
                                 'MaritalStatus'], 1)


# In[96]:


hr_data


# ## Mapping attrition as 1 for yes and 0 for no

# In[97]:


hr_data['Attrition'] = hr_data['Attrition'].replace({'Yes': 1, "No": 0})
hr_data


# ## Calculating rate of attrition

# In[98]:


Attrition = (sum(hr_data['Attrition'])/len(hr_data['Attrition'].index))*100
Attrition


# In[99]:


def balanced_data(df): 
    y=df["Attrition"] 
    X=df.drop(["Attrition"], axis=1) 
    sm = SMOTE(random_state = 2) 
    df_train_res, y_train_res = sm.fit_sample(X, y) 
    return df_train_res,y_train_res 

X_bal, y_bal=balanced_data(hr_data)


# ## Splitting the data into training and testing sets

# In[100]:


X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(X_bal, y_bal, train_size=0.8, test_size=0.2, random_state=0)
#y_test=X_test["Attrition"]
#X_test1=X_test.drop(["Attrition"],axis=1)
X_test_bal


# In[101]:


X_train_bal


# In[102]:


y_test_bal


# In[103]:


y_train_bal


# In[104]:


#y_train_bal


# In[105]:


#Attrition1 = (sum(X_train['Attrition'])/len(X_train['Attrition'].index))*100
#Attrition1


# In[106]:


#X_train_bal,y_bal=balanced_data(X_train)
print("After oversampling, the shape of train data is:{}".format(X_bal.shape))
print("After oversampling, shape of y_train:{}\n ".format(y_bal.shape))
print("After OverSampling, counts of label '1 in Target' : {}".format(sum(y_bal == 1)))
print("After OverSampling, counts of label '0'in Target : {}".format(sum(y_bal == 0)))


# ## Standardising the variables

# In[107]:


scaler = StandardScaler()

X_train_bal[[ 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike',
       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
           'DistanceFromHome','Age','hrs']] = scaler.fit_transform(X_train_bal[[ 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike',
       'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
       'YearsAtCompany', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
           'DistanceFromHome','Age','hrs']])

X_train_bal


# In[108]:


X_train_bal[cont_var]


# ## Calculate correlation matrix

# In[109]:


corrmat = X_train_bal[cont_var].corr() 
corrdf = corrmat.where(np.triu(np.ones(corrmat.shape), k=1).astype(np.bool))
corrdf = corrdf.unstack().reset_index()
corrdf.columns = ['Var1', 'Var2', 'Correlation']
corrdf.dropna(subset = ['Correlation'], inplace = True)
corrdf['Correlation'] = round(corrdf['Correlation'], 2)
corrdf['Correlation'] = abs(corrdf['Correlation'])
matrix= corrdf.sort_values(by = 'Correlation', ascending = False).head(50)
matrix


# In[110]:


unique=list(set(matrix.Var2))
len(unique)


# In[111]:


unique


# In[112]:


#X_test = X_test.drop(unique,1)
#X_train = X_train.drop(unique,1)


# In[113]:


#X_train


# In[114]:


#X_test


# In[115]:


plt.figure(figsize = (50,25))
sns.heatmap(X_train_bal[cont_var].corr(),annot = True,cmap="Reds")
plt.show()


# In[116]:


model=sm.GLM(y_train_bal, sm.add_constant(X_train_bal), family=sm.families.Binomial())
result=model.fit()
result.summary()


# In[117]:


lr=LogisticRegression()
lr.fit(X_train_bal,y_train_bal)


# In[118]:


y_train_predicted = lr.predict(X_train_bal)
y_train_predicted


# In[119]:


#y_train_predicted = y_train_predicted.values.reshape(-1)
#y_train_predicted


# In[120]:


y_train_predicted_final = pd.DataFrame({'Attrition':y_train_bal.values, 'Predicted_Attrition':y_train_predicted})
y_train_predicted_final['EmployeeID'] = y_train_bal.index
y_train_predicted_final


# In[121]:


y_train_predicted_final['predicted'] = y_train_predicted_final.Predicted_Attrition.map(lambda x: 1 if x > 0.5 else 0)
y_train_predicted_final


# In[122]:


confusion = metrics.confusion_matrix(y_train_predicted_final.Attrition, y_train_predicted_final.predicted )
print(confusion)


# In[123]:


print(metrics.accuracy_score(y_train_predicted_final.Attrition, y_train_predicted_final.predicted))


# ## Saving the pre processed data

# In[124]:


X_train_bal.to_csv(r"C:\Users\USER\Desktop\file1.csv")


# ## List the selected variables after variable selection using lasso regression

# In[125]:


col=["hrs", "Age", "NumCompaniesWorked", "StockOptionLevel", "TotalWorkingYears","TrainingTimesLastYear", 
     "YearsSinceLastPromotion", "YearsWithCurrManager", "JobInvolvement_Low", "JobInvolvement_Medium",
     "JobInvolvement_Very High", "EnvironmentSatisfaction_Low", "EnvironmentSatisfaction_Very High", "JobSatisfaction_Low",
    "JobSatisfaction_Very High", "WorkLifeBalance_Best", "WorkLifeBalance_Better", "WorkLifeBalance_Good"]
col


# In[126]:


X_train_bal.drop(X_train_bal.columns.difference(col), axis=1,inplace=True)


# In[127]:


X_train_bal_sel=X_train_bal
X_train_bal_sel


# In[128]:


model1=sm.GLM(y_train_bal, sm.add_constant(X_train_bal_sel), family=sm.families.Binomial())
result1=model1.fit()
result1.summary()


# In[129]:


lr1=LogisticRegression()
lr1.fit(X_train_bal_sel,y_train_bal)


# In[130]:


y_train_pred = lr1.predict(X_train_bal_sel)
y_train_pred


# In[131]:


#y_train_pred = y_train_pred.values.reshape(-1)
#y_train_pred


# In[132]:


y_train_pred_final = pd.DataFrame({'Attrition':y_train_bal.values, 'Predicted_Attrition':y_train_pred})
y_train_pred_final['EmployeeID'] = y_train_bal.index
y_train_pred_final


# In[133]:


#y_train_pred_final['predicted'] = y_train_pred_final.Attrition_Prob.map(lambda x: 1 if x > 0.5 else 0)
#y_train_pred_final


# In[134]:


confusion = metrics.confusion_matrix(y_train_pred_final.Attrition, y_train_pred_final.Predicted_Attrition)
print(confusion)


# In[135]:


print(metrics.accuracy_score(y_train_pred_final.Attrition, y_train_pred_final.Predicted_Attrition))


# ## VIF

# In[136]:


vif = pd.DataFrame()
vif['Features'] = X_train_bal_sel.columns
vif['VIF'] = [variance_inflation_factor(X_train_bal_sel.values, i) for i in range(X_train_bal_sel.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# ## Metrics beyond simply accuracy

# In[137]:


TruePos = confusion[1,1] 
TrueNeg = confusion[0,0] 
FalsePos = confusion[0,1]
FalseNeg = confusion[1,0]


# In[138]:


# Check the sensitivity of our logistic regression model
TruePos / float(TruePos+FalseNeg)


# In[139]:


# calculate specificity
TrueNeg / float(FalsePos+TrueNeg)


# In[140]:


# Calculate false postive rate - 
print(FalsePos/ float(TrueNeg+FalsePos))


# In[141]:


# positive predictive value 
print (TruePos / float(TruePos+FalsePos))


# In[142]:


# negative predictive value 
print (TrueNeg / float(TrueNeg+FalseNeg))


# In[143]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    return None

fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Attrition, y_train_pred_final.Predicted_Attrition, 
                                         drop_intermediate = False )

draw_roc(y_train_pred_final.Attrition, y_train_pred_final.Predicted_Attrition)


# ## Precision and Recall

# In[144]:


precision_score(y_train_pred_final.Attrition, y_train_pred_final.Predicted_Attrition)


# In[145]:


recall_score(y_train_pred_final.Attrition, y_train_pred_final.Predicted_Attrition)


# ## Precision and Recall Tradeoff

# In[146]:


p, r, thresholds = precision_recall_curve(y_train_pred_final.Attrition, y_train_pred_final.Predicted_Attrition)


# In[147]:


plt.plot(thresholds, p[:-1], "g-")
plt.plot(thresholds, r[:-1], "r-")
plt.xlabel("Thresholds")
plt.ylabel("Score")
plt.title("Precision and recall curve")
plt.show()


# ## Prediction using Testing Data

# In[175]:


X_test_bal.drop(X_test_bal.columns.difference(col), axis=1,inplace=True)
X_test_bal_sel=X_test_bal
X_test_bal_sel


# In[176]:


X_test_bal_sel


# In[177]:


#X_test_sm = sm.add_constant(X_test1)
y_test_pred = lr1.predict(X_test_bal_sel)
y_test_pred


# In[178]:


y_pred_1 = pd.DataFrame(y_test_pred)
y_pred_1


# In[179]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test_bal)
# Putting EmployeeID to index
y_test_df['EmployeeID'] = y_test_df.index
# Removing index for both dataframes to append them side by side 
y_pred_1.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)
# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_1],axis=1)
# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Predicted_Attrition'})
# Rearranging the columns
y_pred_final = y_pred_final.reindex(['EmployeeID','Attrition','Predicted_Attrition'], axis=1)
# Let's see the y_pred_final
y_pred_final


# In[180]:


#y_pred_final['final_predicted'] = y_pred_final.Attrition_Prob.map(lambda x: 1 if x > 0.18 else 0)
#y_pred_final


# In[181]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Attrition, y_pred_final.Predicted_Attrition)


# In[182]:


confusion2 = metrics.confusion_matrix(y_pred_final.Attrition, y_pred_final.Predicted_Attrition)
confusion2


# In[183]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[184]:


# Let's see the sensitivity of our logistic regression model
TP / float(TP+FN)


# In[185]:


# Let us calculate specificity
TN / float(TN+FP)


# In[186]:


fpr, tpr, thresholds = metrics.roc_curve( y_pred_final.Attrition, y_pred_final.Predicted_Attrition, 
                                         drop_intermediate = False )
draw_roc(y_pred_final.Attrition, y_pred_final.Predicted_Attrition)


# In[187]:


precision_score(y_pred_final.Attrition, y_pred_final.Predicted_Attrition)


# In[188]:


recall_score(y_pred_final.Attrition, y_pred_final.Predicted_Attrition)


# In[ ]:





# In[ ]:




