#!/usr/bin/env python
# coding: utf-8

# In[18]:


# Question 1


# In[46]:


import pandas as pd 
import numpy as np 
import seaborn as sns
import statsmodels.formula.api as smf


# In[19]:


delivery_time = pd.read_csv("F:/Dataset/delivery_time.csv")


# In[20]:


delivery_time


# In[21]:


sns.distplot(delivery_time['Delivery Time'])


# In[22]:


sns.distplot(delivery_time['Sorting Time'])


# In[27]:


delivery_time=delivery_time.rename({'Delivery Time':'d_t', 'Sorting Time':'s_t'},axis=1)


# In[29]:


delivery_time.corr()


# In[30]:


model = smf.ols('d_t~s_t',data=delivery_time).fit()


# In[32]:


sns.regplot(x='s_t',y='d_t',data=delivery_time)


# In[33]:


model.params


# In[34]:


(model.tvalues,'\n',model.pvalues)


# In[35]:


model.rsquared , model.rsquared_adj


# In[37]:


Delivery_Time = (6.582734) + (1.649020)*(5)


# In[38]:


Delivery_Time


# In[41]:


new_data=pd.Series([5,8])


# In[42]:


new_data


# In[43]:


data_pred=pd.DataFrame(new_data,columns=['s_t'])


# In[44]:


data_pred


# In[45]:


model.predict(data_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[1]:


# Question 2


# In[2]:


import pandas as pd 
import numpy as np 


# In[3]:


Salary= pd.read_csv('F:/Dataset/Salary_Data.csv')


# In[4]:


Salary


# In[6]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[7]:


sns.distplot(Salary['YearsExperience'])
sns.distplot(Salary['Salary'])
plt.legend('YearsExperience','Salary')


# In[8]:


Salary.corr()


# In[9]:


import statsmodels.formula.api as smf


# In[10]:


model=smf.ols('Salary~YearsExperience',data=Salary).fit()


# In[11]:


sns.regplot(x='YearsExperience',y='Salary',data=Salary)


# In[12]:


model.params


# In[22]:


(model.tvalues,'\n',model.pvalues)


# In[23]:


model.rsquared,model.rsquared_adj


# In[24]:


# 3 years salary hike calculation :


# In[25]:


Salary_hike=(25792.200199)+(9449.962321)*(3)


# In[26]:


Salary_hike


# In[30]:


data=pd.Series([3,5])


# In[31]:


data


# In[32]:


data_pred=pd.DataFrame(data,columns=['YearsExperience'])


# In[33]:


data_pred


# In[34]:


model.predict(data_pred)


# In[ ]:




