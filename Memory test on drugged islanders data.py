#!/usr/bin/env python
# coding: utf-8

# In[73]:


from IPython.display import HTML
HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')


# # Memory Test on Drugged islanders data
# ## Description of the data set
# An experiment on the effects of anti-anxiety medicine on memory recall when being primed with happy or sad memories. The participants were done on novel Islanders whom mimic real-life humans in response to external factors.
# 
# Drugs of interest (known-as) [Dosage 1, 2, 3]:
# 
# A - Alprazolam (Xanax, Long-term) [1mg/3mg/5mg]
# 
# T - Triazolam (Halcion, Short-term) [0.25mg/0.5mg/0.75mg]
# 
# S- Sugar Tablet (Placebo) [1 tab/2tabs/3tabs]
# 
# *Dosages follow a 1:1 ratio to ensure validity
# *Happy or Sad memories were primed 10 minutes prior to testing
# *Participants tested every day for 1 week to mimic addiction
# 
# Building the Case:
# Obstructive effects of Benzodiazepines (Anti-Anxiety Medicine):
# 
# Long term adverse effects on Long Term Potentiation of synapses, metacognition and memory recall ability
# http://www.jstor.org/stable/43854146
# Happy Memories:
# 
# research shown positive memories to have a deeper and greater volume of striatum representation under an fMRI
# https://www.sciencedirect.com/science/article/pii/S0896627314008484
# Sad Memories:
# 
# research shown sad memories invokes better memory recall for evolutionary purpose whereas, happy memories are more susceptible to false memories
# http://www.jstor.org/stable/40064315
# Participants - all genders above 25+ years old to ensure a fully developed pre-frontal cortex, a region responsible for higher level cognition and memory recall.
# 
# Content
# File contains information on participants drug treatment information along with their test scores.

# In[74]:


import pandas as pd


# In[75]:


import numpy as np
import plotly as plty
import matplotlib.pyplot as plt
import seaborn as sns


# In[76]:


df = pd.read_csv('C:/Users/gotti/Downloads/islander_data.csv')


# In[77]:


df.head()


# ## Attributes of dataset
# 
#  1)**first_name** : First name of Islander  
#  2)**last_name** : Last name of Islander  
#  3)**age** : Age of Islander  
#  4)**Happy_Sad_group** : Happy or Sad Memory priming block  
#  5)**Dosage** : 1-3 to indicate the level of dosage (low - medium - over recommended daily intake)  
#  6)**Drug** : Type of Drug administered to Islander  
#  7)**Mem_Score_Before** : Seconds - how long it took to finish a memory test before drug exposure  
#  8)**Mem_Score_After** : Seconds - how long it took to finish a memory test after addiction achieved  
#  9)**Diff** : Seconds - difference between memory score before and after  

# ## Shape of dataset:

# In[78]:


df.shape


#  Our datset consists of 198 records with 9 columns present in it

# ## Information of dataset: 
#     These gives the information of dataset by giving the type of each attribute

# In[79]:


df.info()


# In[80]:


df.isnull().sum()


#  From the above obtained results. we observe there are no missing values present in the data

# ## Statistics of each variable present in datset:

# In[81]:


df.describe(include='all')


# From the above statistics we observe that it consists of minimum age of 24 maximum age is 83 the average age is 37. The minimum **Dosage** is 1 where as maximum is 3 the average is 2. The **Mem_score_Before** minimum seconds are 57 where as maximum seconds are 110 its average seconds are 54. The **Mem_score_After** its minimum seconds are 27 and maximum seconds are 120 where as 56 are its average seconds. BY the obtained values in seconds we observe the difference between them are the minimum difference is -40 seconds from these we observe that the drug is making us to loose the memory very early may be these has been obtained like these because it also consists of the data of the persons with higher dosage of the drug where as its maximum difference is 49 these may be happened by having little dosage and the average is 1.70 seconds these occured due to the minimum dosage. 

# ## Univariate analysis

# ## Age:-
#  Now we observe the number of the persons present the particular age

# In[82]:


df['age'].value_counts()


# The above shown data is used to mention the number of persons in the particular age group

# ## Mean value for each Drug:-
#    Now we group the same drug usage persons into one group and we find the mean value for it.

# In[83]:


df1=df.groupby('Drug')['Diff'].agg(['count','mean']).sort_values(by = 'mean',ascending = False)


# In[84]:


df1


# We create a separate data frame for mentioning average age over difference. which is used to group the drugs with the obtained difference of time between drugs before and after.

# In[85]:


df['avg_ovr_diff'] = df.groupby('Drug')['Diff'].transform('mean')
df['avg_ovr_diff']


# Now we would group the age group which consists of 5 members in each group

# In[86]:


df['age_group'] = pd.cut(
    df['age'],
    np.arange(start=df['age'].min(), step=5, stop=df['age'].max())
)
df[['age', 'age_group']].head()


#  Now  we are adding a new column for giving average age group

# In[87]:


df['avg_age_diff'] = df.groupby('age_group')['Diff'].transform('mean')
df[['age_group', 'avg_age_diff']].head()


# In the above step  we are adding a new column for giving average age group

# ## Grouping the persons by Drugs:-
#   Now we would group the persons by the type of drug used.

# In[88]:


alprazolam_df = df.loc[df['Drug'] == 'A']
alprazolam_df


# In[89]:


triazolam_df = df.loc[df['Drug'] == 'T']
triazolam_df


# In[90]:


sugar_df = df.loc[df['Drug'] == 'S']
sugar_df


# ## Average age difference for each drug:-
#    We give the average age difference for each group.

# In[91]:


alprazolam_df['avg_age_diff'] = alprazolam_df.groupby('age_group')['Diff'].transform('mean')
alprazolam_df[['age_group', 'avg_age_diff']].head()


# In[92]:


triazolam_df['avg_age_diff'] = triazolam_df.groupby('age_group')['Diff'].transform('mean')
triazolam_df[['age_group', 'avg_age_diff']].head()


# In[93]:


sugar_df['avg_age_diff'] = sugar_df.groupby('age_group')['Diff'].transform('mean')
sugar_df[['age_group', 'avg_age_diff']].head()


# ## Usage of the drug:- 
#     From the data we observe the number of records available for each drug

# In[94]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.countplot(x='Drug', data=df, ax=ax)
plt.tight_layout()
plt.title('Number of records for each record')
plt.show()


# From the above obtained plot we observe that the data available for each drug is almost same

# ## Number of seconds for Mem_score_Before:-
#     Now we take the number of seconds for each member before taking drug is as mentioned below.

# In[95]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.boxplot(x='Mem_Score_Before', data=df, ax=ax)
plt.tight_layout()
plt.title('Number of seconds for memory loss')
plt.show()


# From the above obtained boxplot we observe there is one outlayer present in it from the plot we observe the average age is approximately 55.

# ## Mem_Score_after:-
#      The number of seconds for the memory score after the drug is given is obtained below

# In[96]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.boxplot(x='Mem_Score_After', data=df, ax=ax)
plt.tight_layout()
plt.title('Number of seconds for memory loss')
plt.show()


# From the above obtained box plot we observe the average value is approximately 56. It consists of two outliers present in it

# ## Difference between the memory before and after:-
#     Now we observe the difference between the memory before and after the drug is given are mentioned.

# In[97]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.boxplot(x='Diff', data=df, ax=ax)
plt.tight_layout()
plt.title('Difference in seconds before and after the drug')
plt.show()


# From the above obtained boxplot we observe more number of outliers present in it the average difference between them is less it would be across 0-5 seconds

# # Bivariate Analysis:-

# ## Difference Memory Before and After:-
#      Now we compare the memory before and after the drug is injected

# In[98]:


plt.figure(figsize=(14,8))
plt.scatter(x=df['Mem_Score_Before'],y=df['Mem_Score_After'])
plt.title('The difference between before and after the drug is injected')
plt.xlabel('Mem_Score_Before')
plt.ylabel('Mem_Score_After')
plt.show()


# From the above obtained scatter plot we observe the memory before and after the drug is injected. It shows that the memory score is almost equal between them.

# ## Difference in memory score for each drug:- 
#    Now we take the difference in memory score by each drug for all ages

# In[99]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.barplot(x=df['Drug'], y=df['avg_ovr_diff'], ax=ax)
plt.tight_layout()
plt.title('Overall Avg Difference in Memory Score by Drug \n (For all Ages)')
plt.show()


# # Multivariate analysis:- 

# ## Number of observations for each age group:- 
#      We take the number of observations for each group.

# In[100]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.countplot(x='age_group', data=df, ax=ax)
plt.title('Number of Observations by Age Group')
plt.tight_layout()
plt.show()


# From the above obtained graph we observe the number of observations for each group.

# In[101]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.barplot(x=alprazolam_df['age_group'], y=alprazolam_df['avg_age_diff'], ax=ax)
plt.title('Average Memory Score Difference \n (Alprazolam)')
plt.tight_layout()
plt.show()


# In[102]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.barplot(x=triazolam_df['age_group'], y=triazolam_df['avg_age_diff'], ax=ax)
plt.plot('Average Memory Score Difference \n (Triazolam)')
plt.tight_layout()
plt.show()


# In[103]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.barplot(x=sugar_df['age_group'], y=sugar_df['avg_age_diff'], ax=ax)
plt.title('Average Memory Score Difference \n (Placebo)')
plt.tight_layout()
plt.show()


# In[104]:


fig = plt.figure(figsize=(12, 5))
ax = fig.add_axes([0,0,1,1])
sns.barplot(x=df['age_group'], y=df['avg_age_diff'], ax=ax)
plt.title('Avg Change in Memory Score by Age Group \n (Includes Both Drugs and Placebo)')
plt.tight_layout()
plt.show()


# # Conclusion:- 
#     from the above obtained graphs and the results we observe the age group above 64 are not getting affected to the high dosage of drugs where as they are impacting more for using less dosage of drugs. 

# In[ ]:




