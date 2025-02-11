#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np


# In[2]:


if not os.path.exists("plots/") : os.mkdir("plots/")


# In[3]:


df = []
result_list = sorted(os.listdir('./results/'))
for file in result_list:
    df.append(pd.read_csv('./results/' + file, names = ['cor', 'cov', 'gc', 'sic', 'throughput']))


# In[4]:


cor = []
cov = []
gc = []
sic = []
throughput = []
for x in df:
    cor.append(np.mean(x['cor']))
    cov.append(np.mean(x['cov']))
    gc.append(np.mean(x['gc']))
    sic.append(np.mean(x['sic']))
    throughput.append(np.mean(x['throughput']))


# In[5]:


q = np.arange(0, 1.1, 0.1)


# plot correlation

# In[6]:


fig = plt.figure(figsize=(8,6), dpi=200)
plt.xticks(q)
plt.plot(q, cor, '-*', color = 'red')
plt.title("Pearson Degree Correlation", fontsize = 14)
plt.xlabel("q")
plt.ylabel("cor")
plt.savefig('./plots/correlation.png')
plt.show()


# In[7]:


fig = plt.figure(figsize=(8,6), dpi=200)
plt.xticks(q)
plt.plot(q, cov, '-*', color = 'blue')
plt.title("Covariance", fontsize = 14)
plt.xlabel("q")
plt.ylabel("cov")
plt.savefig('./plots/covariance.png')
plt.show()


# In[8]:


fig = plt.figure(figsize=(8,6), dpi=200)
plt.xticks(q)
plt.plot(q, gc, '-*', color = 'green')
plt.title("Giant Component Size", fontsize = 14)
plt.xlabel("q")
plt.ylabel("gc")
plt.savefig('./plots/giant_component_size.png')
plt.show()


# In[9]:


fig = plt.figure(figsize=(8,6), dpi=200)
plt.xticks(q)
plt.plot(q, sic, '-*', color = 'm')
plt.title("SIC success rate", fontsize = 14)
plt.xlabel("q")
plt.ylabel("sic")
plt.savefig('./plots/sic.png')
plt.show()


# In[ ]:


fig = plt.figure(figsize=(8,6), dpi=200)
plt.xticks(q)
plt.plot(q, throughput, '-*', color = 'c')
plt.title("Throughput", fontsize = 14)
plt.xlabel("q")
plt.ylabel("throughput")
plt.savefig('./plots/throughput.png')
plt.show()

