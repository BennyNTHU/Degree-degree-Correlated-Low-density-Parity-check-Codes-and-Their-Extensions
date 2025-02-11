#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import ast
import csv
import sys
import string
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


parseStr = lambda x: x.isalpha() and x or x.isdigit() and int(x) or x.isalnum() and x or len(set(string.punctuation).intersection(x)) == 1 and x.count('.') == 1 and float(x) or x


# ## 一. 畫圖參數設定

# In[3]:


d = int(sys.argv[1])
G = int(sys.argv[2])
# q_range = ast.literal_eval(sys.argv[3])


# In[4]:


# d = 3
# G = 3
q_range = np.arange(0, 1.0, 0.2)


# In[5]:


q_str = ['{:.1f}'.format(q) for q in q_range] # q會跑哪些值
deltas_tick = np.arange(0, 1.1, 0.1)
plot_dir = "./plots/d" + str(d) + 'G' + str(G) + '/'
result_dir = "./results/d" + str(d) + 'G' + str(G) + '/'


# 放置圖片的資料夾

# In[6]:


if not os.path.exists(plot_dir) : os.mkdir(plot_dir)


# ## 二. 讀取前2個metric並繪圖
# 前2個metric是放在result_dir內的獨立檔案，後面幾個metric是放在以$q$值不同的資料夾中

# In[7]:


cor = []
cov = []


# In[8]:


df = []
result_list = sorted([f for f in os.listdir(result_dir) if (os.path.isdir(result_dir + f)==False)]) # 不要讀到資料夾
for file in result_list:
    df.append(pd.read_csv(result_dir + file, names = ['cor', 'cov']))


# In[9]:


for x in df:
    cor.append(np.mean(x['cor']))
    cov.append(np.mean(x['cov']))


# In[10]:


fig = plt.figure(figsize=(8,6), dpi=200)
plt.xticks(q_range)
plt.plot(q_range, cor, '-*', color = 'red')
plt.title("Pearson Degree Correlation", fontsize = 14)
plt.xlabel("q")
plt.ylabel("cor")
plt.annotate(str(cor[0])[0:6], (q_range[0], cor[0]))
plt.annotate(str(cor[-1])[0:6], (q_range[-1], cor[-1]))
plt.savefig(plot_dir + 'correlation.png')
# plt.show()


# In[11]:


fig = plt.figure(figsize=(8,6), dpi=200)
plt.xticks(q_range)
plt.plot(q_range, cov, '-*', color = 'blue')
plt.title("Covariance", fontsize = 14)
plt.xlabel("q")
plt.ylabel("cov")
plt.annotate(str(cov[0])[0:6], (q_range[0], cov[0]))
plt.annotate(str(cov[-1])[0:6], (q_range[-1], cov[-1]))
plt.savefig(plot_dir + 'covariance.png')
# plt.show()


# In[12]:


del df, result_list


# ## 三. 讀取與$\delta$有關的metric並繪圖
# ### 1. 讀取九個metric並計算平均

# In[13]:


ad_limit = []
a2d_limit = []
alpha = []
bd_limit = []
b2d_limit = []
beta = []
yd_limit = []
y2d_limit = []
gamma = []


# In[14]:


for q in q_str: # 對每個q的資料夾
    df_list = []
    result_list = sorted(os.listdir(result_dir + q)) # q=0到1的每個檔案
    
    ad_limit_delta = []
    a2d_limit_delta = []
    alpha_delta = []
    bd_limit_delta = []
    b2d_limit_delta = []
    beta_delta = []
    yd_limit_delta = []
    y2d_limit_delta = []
    gamma_delta = []
    
    for file in result_list: # 把所有dataframe存進df_list裡面
        df_list.append(pd.read_csv(result_dir + q + '/' + file, 
                                   names = ['ad_limit_delta', 'a2d_limit_delta', 'alpha_delta', 'bd_limit_delta', 
                                            'b2d_limit_delta', 'beta_delta', 'yd_limit_delta', 'y2d_limit_delta', 
                                            'gamma_delta'], 
                                   engine='python'))
        
    for x in df_list: # 對每個dataframe
        ad_limit_delta.append(np.mean(x['ad_limit_delta']))
        a2d_limit_delta.append(np.mean(x['a2d_limit_delta']))
        alpha_delta.append(np.mean(x['alpha_delta']))
        bd_limit_delta.append(np.mean(x['bd_limit_delta']))
        b2d_limit_delta.append(np.mean(x['b2d_limit_delta']))
        beta_delta.append(np.mean(x['beta_delta']))
        yd_limit_delta.append(np.mean(x['yd_limit_delta']))
        y2d_limit_delta.append(np.mean(x['y2d_limit_delta']))
        gamma_delta.append(np.mean(x['gamma_delta']))
        
    ad_limit.append(ad_limit_delta)
    a2d_limit.append(a2d_limit_delta)
    alpha.append(alpha_delta)
    bd_limit.append(bd_limit_delta)
    b2d_limit.append(b2d_limit_delta)
    beta.append(beta_delta)
    yd_limit.append(yd_limit_delta)
    y2d_limit.append(y2d_limit_delta)
    gamma.append(gamma_delta)


# In[15]:


metrics = [ad_limit, a2d_limit, alpha, bd_limit, b2d_limit, beta, yd_limit, y2d_limit, gamma]
metrics_str = ['ad_limit', 'a2d_limit', 'alpha', 'bd_limit', 'b2d_limit', 
               'beta', 'yd_limit', 'y2d_limit', 'gamma']
metrics_LaTeX = [r'$\lim_{i\to\infty}\alpha_d^{(i)}$',
                 r'$\lim_{i\to\infty}\alpha_{2d}^{(i)}$', 
                 r'$\lim_{i\to\infty}\alpha^{(i)}$', 
                 r'$\lim_{i\to\infty}\beta_d^{(i)}$', 
                 r'$\lim_{i\to\infty}\beta_{2d}^{(i)}$', 
                 r'$\lim_{i\to\infty}\beta^{(i)}$',
                 r'$\lim_{i\to\infty}\gamma_d^{(i)}$', 
                 r'$\lim_{i\to\infty}\gamma_{2d}^{(i)}$', 
                 r'$\lim_{i\to\infty}\gamma^{(i)}$']


# ### 2. 一個找出某一$q$值跑了那些$\delta$的函式

# In[16]:


def find_deltas(q): # 找出每個q值資料夾中所有.csv檔案名稱中的delta值,input q必須是字串
    csv_list = sorted(os.listdir(result_dir + q))
    d = [parseStr((file_name[0:][:-4])[6:]) for file_name in csv_list]
    d = [0.0] + d
    d.remove('0.0')
    return d # 去掉尾端的.csv以及開頭的delta=


# ### 3. 繪製log plot

# In[17]:


for m, m_str, m_latex in zip(metrics, metrics_str, metrics_LaTeX):
    fig = plt.figure(figsize=(8,6), dpi=200)

    plt.xticks(deltas_tick) # 設定刻度
    plt.yscale("log", base=10) # log plot
    plt.grid(color = 'blue', linestyle = '--', linewidth = 1) # 設定網格
    plt.title(m_latex + ' (simulation, log plot)', fontsize = 14) # 標題
    plt.xlabel(r'$\delta$') # x軸標題
    plt.ylabel("probability.") # y軸標題

    colormap = plt.cm.gist_ncar # 顏色輪流出現
    
    lines = []
    for data, qs in zip(m, q_str):
        deltas = find_deltas(qs)
        line = plt.plot(deltas, data, '-*')
        lines.append(r'$q=$' + qs)
    plt.legend(lines)
    
    plt.savefig(plot_dir + m_str + '_log.png')
#     plt.show()


# ### 4. Normal plot

# In[18]:


for m, m_str, m_latex in zip(metrics, metrics_str, metrics_LaTeX):
    fig = plt.figure(figsize=(8,6), dpi=200)

    plt.xticks(deltas_tick) # 設定刻度
    plt.grid(color = 'blue', linestyle = '--', linewidth = 1) # 設定網格
    plt.title(m_latex + ' (simulation)', fontsize = 14) # 標題
    plt.xlabel(r'$\delta$') # x軸標題
    plt.ylabel("probability.") # y軸標題

    colormap = plt.cm.gist_ncar # 顏色輪流出現
    
    lines = []
    for data, qs in zip(m, q_str):
        deltas = find_deltas(qs)
        line = plt.plot(deltas, data, '-*')
        lines.append(r'$q=$' + qs)
    plt.legend(lines)
    
    plt.savefig(plot_dir + m_str + '.png')
#     plt.show()


# ## 四. Percolation Threshold分析

# In[19]:


for data_a, data_y, qs in zip(metrics[0], metrics[8], q_str):
    
    deltas = find_deltas(qs)
    deltas_r = list(reversed(deltas))
    data_ar = list(reversed(data_a))
    data_yr = list(reversed(data_y))
    
#     # 不允許任何誤差下測量Percolation Threshold
#     i = next(x[0] for x in enumerate(data_ar) if x[1] == 0)
#     j = next(x[0] for x in enumerate(data_yr) if x[1] == 1)
    
#     # 允許0.01的誤差下測量Percolation Threshold
#     i = next(x[0] for x in enumerate(data_ar) if x[1] > 1/100)
#     j = next(x[0] for x in enumerate(data_yr) if x[1] < 99/100)
    
    # 藉由significant gap測量Percolation Threshold
    data_a_diff = [abs(data_a[i]-data_a[i-1]) for i in range(1,len(data_a))]
    max_gap_a = 0
    for i in range(1, len(data_a_diff)):
        if (abs(deltas[i]-deltas[i-1])<0.002):
            if (max_gap_a < data_a_diff[i]):
                max_gap_a = data_a_diff[i]
                
    data_y_diff = [abs(data_y[i]-data_y[i-1]) for i in range(1,len(data_y))]
    max_gap_y = 0
    for j in range(1, len(data_y_diff)):
        if (abs(deltas[j]-deltas[j-1])<0.002):
            if (max_gap_y < data_y_diff[j]):
                max_gap_y = data_y_diff[j]
    
    i = data_a_diff.index(max_gap_a)
    j = data_y_diff.index(max_gap_y)
    
    delta_a = deltas[i]
    delta_y = deltas_r[j]

    with open('./threshold' + '.csv', 'a', newline = '') as csvFile:
        csvWriter = csv.writer(csvFile, delimiter = ',')
        csvWriter.writerow([qs, str(delta_a), str(delta_y)])


# 請筱雯幫忙畫一下圖

# In[20]:


for (alphas, gammas, qs) in zip(ad_limit, gamma, q_str):
    deltas = find_deltas(qs)
    with open(plot_dir + 'alpha_and_gamma_q=' + qs + '.csv', 'a', newline = '') as csvFile:
        csvWriter = csv.writer(csvFile, delimiter = ',')
        for (delta, alpha, gamma) in zip(deltas, alphas, gammas):
            csvWriter.writerow([delta, alpha, gamma])

