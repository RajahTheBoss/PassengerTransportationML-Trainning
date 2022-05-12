#!/usr/bin/env python
# coding: utf-8

# Lets start with importing libraries

# ## 1.1 Preparing data

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# Loading data

# In[2]:


first_part = pd.read_excel("train_first_part.xlsx")
second_part = pd.read_json("train_second_part.json")


# In[3]:


weather = pd.read_csv("weather.csv")


# In[4]:


first_part.head()


# In[5]:


second_part.head()


# In[6]:


df = pd.concat([first_part, second_part]).drop_duplicates("id").drop("id", axis = 1)


# In[7]:


df


# In[8]:


df.info()


# Для удобства работы с послед. данными переводим колонки с типом данных `datetime` в нужный формат

# In[9]:


df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])


# In[10]:


df.info()


# ## 1.2 Форматирование данных

# ### 1.2.1 Осмотр данных на пропуски

# In[11]:


plt.figure(figsize = (10, 10))
sns.heatmap(df.isna())


# In[12]:


df.info()


# In[13]:


df.describe()


# In[14]:


df["trip_duration"] = df["trip_duration"].apply(lambda x: x / 60)


# In[15]:


df["trip_duration"]


# In[16]:


df.describe()


# Lets display all data on the plane

# In[17]:


import warnings
warnings.filterwarnings("ignore")
sns.boxplot(df["trip_duration"])


# Cleaning up emissions

# In[18]:


q = df["trip_duration"].quantile(0.92)
df = df[df["trip_duration"] < q]


# ### 1.2.2 Factorization of data

# In[19]:


df["store_and_fwd_flag"].value_counts()


# In[20]:


d_tm = {"N" : 0, "Y" : 1}
df["store_and_fwd_flag"] = df["store_and_fwd_flag"].map(d_tm)


# Result of factorization

# In[21]:


df["store_and_fwd_flag"].value_counts()


# ### Formatting weather conditions

# In[22]:


weather.isna().sum()


# In[23]:


weather.info()


# In[24]:


for col in ["precipitation", "snow fall", "snow depth"]:
    print(weather[col].unique())
    print("-" * 10)


# In[25]:


for col in ["precipitation", "snow fall", "snow depth"]:
    weather.loc[weather[col] == "T", col] = 0
    weather[col] = weather[col].astype(float)


# In[26]:


weather.info()


# In[27]:


weather.describe()


# In[28]:


for temp in ["maximum temperature", "minimum temperature", "average temperature"]:
    weather[temp] = weather[temp].apply(lambda x : (x - 32) / 1.8)


# Let's check the data conversion from Fahrenheit to Celsius

# In[29]:


weather.describe()


# ## 1.3 Combining a set of travel and weather data

# In[30]:


df["date"] = df["pickup_datetime"].apply(lambda x : str(x.day) + '-' + str(x.month) + '-' + str(x.year))
df = df.merge(weather, on = "date").drop(["date", "dropoff_datetime"], axis = 1)


# In[31]:


df.head()


# # 1.4 Saving data and conclusion

# In[32]:


df.to_csv("c1_result.csv", index = False)


# 1. Gaps in all data have been processed
# 2. All outliers in all data have been processed
# 3. The data is brought to the desired format
# 4. All samples were combined into 1 final
