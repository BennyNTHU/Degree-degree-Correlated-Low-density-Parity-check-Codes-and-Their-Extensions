#!/usr/bin/env python
# coding: utf-8

# # LDPC example 2

# In[1]:


import os
import ast
import csv
import sys
import time
import math
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from networkx.algorithms import bipartite
from numpy.random import poisson


# ## 一. 設定實驗參數

# In[2]:


d1 = int(sys.argv[1])
d2 = int(sys.argv[2])
p1 = float(sys.argv[3])
p2 = float(sys.argv[4])
G = int(sys.argv[5])
N = int(sys.argv[6])
comb_set = int(sys.argv[7])
# deltas = ast.literal_eval(sys.argv[8])


# In[3]:


# d1 = 4
# d2 = 3 # please make sure d1>d2
# p1 = 0.1
# p2 = 0.2
# G = 3
# N = 18000
# comb_set = 1
deltas = np.arange(0.25, 0.35, 0.001)# delta 會跑哪些值
result_dir = "./results/comb" + str(comb_set) + "_decode/"
cov_dir = "./results/comb" + str(comb_set) + "_cor/"
model_dir = "./model/comb" + str(comb_set) + "/"


# In[4]:


d1 = max(d1,d2) # make sure d1>d2
d2 = min(d1,d2)
n = N # Number of transmitter, i.e., n
M = int(N / G) # Number of reciever, i.e., n-k
k = N - M


# In[5]:


random.seed(time.time()) # random seeds
if not os.path.exists(result_dir) : os.mkdir(result_dir)
if not os.path.exists(cov_dir) : os.mkdir(cov_dir)
if not os.path.exists(model_dir) : os.mkdir(model_dir)


# ## 二. 建立有Negative Degree Correlation的Bipartite Graph for example 2
# 
# 實作步驟：
# 1. 設有一個bipartite graph，左邊的頂點為transmitter $T$，右邊為reciever $R$，數量分別為$n$和$n-k$
# 2. Degree sequence中$d_1,d_2,Gd_1,Gd_2$的比例由(40)給出
# 
#     (1) 所連之node之degree為$d_1$的stub為transmitter block 1, 比例為$$\frac{\frac{p_1+p_2}{d_1}}{\frac{p_1+p_2}{d_1}+\frac{1-p_1-p_2}{d2}}$$
#     
#     (2) 所連之node之degree為$d_2$的stub為transmitter block 2, 比例為$$\frac{\frac{1-p_1-p_2}{d_2}}{\frac{p_1+p_2}{d_1}+\frac{1-p_1-p_2}{d2}}$$
#     
#     (3) 所連之node之degree為$Gd_1$的stub為reciever block 1, 比例為$$\frac{\frac{p_1+p_2}{d_1}}{\frac{p_1+p_2}{d_1}+\frac{1-p_1-p_2}{d2}}$$
#     
#     (4) 所連之node之degree為$Gd_2$的stub為recieverblock 2, 比例為$$\frac{\frac{1-p_1-p_2}{d_2}}{\frac{p_1+p_2}{d_1}+\frac{1-p_1-p_2}{d2}}$$
#     
# 3. 將此bipartite graph建立起來，但先不連結邊
# 4. 建立Stubs
# 
#     (1) transmitter block 1中,type 1所佔之比例為$\frac{p_2}{p_1+p_2}$, type 2所佔之比例為$\frac{p_1}{p_1+p_2}$
#     
#     (2) transmitter block 2中,type 1所佔之比例為$\frac{p_2}{1-p_1-p_2}$, type 2所佔之比例為$\frac{p_1}{1-p_1-2p_2}$
#     
#     (3) reciever block 1中,type 1所佔之比例為$\frac{p_2}{p_1+p_2}$, type 2所佔之比例為$\frac{p_1}{p_1+p_2}$
#     
#     (4) reciever block 2中,type 1所佔之比例為$\frac{p_2}{1-p_1-p_2}$, type 2所佔之比例為$\frac{p_1}{1-p_1-2p_2}$
#     
# 5. 將邊連起來：自$T$中任意選一stub，
# 
#     (1) 若是block 1中的type 1 stub，就連到$R$中block 2的type 1 stub。若是type 2 stub就連到$R$中block 1的type 2 stub。
#     
#     (2) 若是block 2中的type 1 stub，就連到$R$中block 1的type 1 stub。若是type 2 stub就連到$R$中block 2任意的type 2 stub。

# ### Step 2

# 製造兩個degree sequence

# In[6]:


b1_ratio = ((p1+p2)/d1) / ((p1+p2)/d1 + (1-p1-p2)/d2)


# In[7]:


tk = [d1] * int(N*b1_ratio) + [d2] * (N - int(N*b1_ratio))
rk = [G*d1] * int(M*b1_ratio) + [G*d2] * (M - int(M*b1_ratio))


# 若有degree為0的點，則補一個stub給它(in this case必無此點,無檢查必要)

# In[8]:


# tk = [1 if i==0 else i for i in tk]
# rk = [1 if i==0 else i for i in rk]


# 讓所有degree的和相同

# In[9]:


n = sum(tk) - sum(rk)
if (n < 0): # transmitter端stub比較少
    for i in range(0,-n):
        index = random.choice(list(range(0,N)))
        tk[index] = tk[index] + 1
elif (n > 0): # reiever端stub比較少
    for i in range(0,n):
        index = random.choice(list(range(0,M)))
        rk[index] = rk[index] + 1
else:
    pass


# 調整degree總數為偶數

# In[10]:


if (sum(tk)%2 != 0): # 
    index = random.choice(list(range(0,N)))
    tk[index] = tk[index] + 1
    index = random.choice(list(range(0,M)))
    rk[index] = rk[index] + 1


# 經過這些處理之後，transmitter端與reciever端具有相同的stub數(degree)以外，也能確保整個bipartite graph的degree是偶數

# In[11]:


rk = sorted(rk, reverse=True) # sort 2m stubs in descending order
tk = sorted(tk, reverse=True)
degree_transmitter = sum(tk)
degree_reciever = sum(rk)


# In[12]:


# print(np.unique(tk))
# print(np.unique(rk))
# print(degree_transmitter) # 理論上要有幾條邊
# print(degree_reciever)


# ### Step 3
# 現在將此bipartite graph建立起來，且已經保證所有頂點的degree不為0

# In[13]:


transmitter_node_degree_dict = {}
for nodes in range(0,N): # The nodes are named after 0~N-1
    transmitter_node_degree_dict[nodes] = tk[nodes]


# In[14]:


# transmitter_node_degree_dict # degree sequence


# In[15]:


reciever_node_degree_dict = {}
for nodes in range(0,M): # The nodes are named after '0'~'N-1'
    reciever_node_degree_dict[str(nodes)] = rk[nodes]


# In[16]:


model = nx.Graph()
model.add_nodes_from(transmitter_node_degree_dict.keys(), bipartite=0)
model.add_nodes_from(reciever_node_degree_dict.keys(), bipartite=1)


# In[17]:


del tk, rk


# ### Step 4

# 建立transmitter和reciever的stub

# In[18]:


class Stubs():
    def __init__(self, node, stubtype, block):
        self.node = node
        self.stubtype = stubtype
        self.block = block
    
    def get_node(self):
        return self.node
    
    def get_stubtype(self):
        return self.stubtype
    
    def get_block(self):
        return self.block


# In[19]:


transmitter_stubs = []
for nodes in transmitter_node_degree_dict.keys():
    for degree in range(0, transmitter_node_degree_dict[nodes]):
        transmitter_stubs.append(Stubs(nodes, 2, 1))


# In[20]:


reciever_stubs = []
for nodes in reciever_node_degree_dict.keys():
    for degree in range(0, reciever_node_degree_dict[nodes]):  
        reciever_stubs.append(Stubs(nodes, 2, 1))


# 分block, degree為$d_1$ (resp. $Gd_1$)的為block 1, degree為$d_2$ (resp. $Gd_2$)的為block 2

# In[21]:


transmitter_block1 = [stub for stub in transmitter_stubs if (transmitter_node_degree_dict[stub.get_node()]>=d1)] 
transmitter_block2 = [stub for stub in transmitter_stubs if (transmitter_node_degree_dict[stub.get_node()]<d1)]


# In[22]:


reciever_block1 = [stub for stub in reciever_stubs if (reciever_node_degree_dict[stub.get_node()]>=G*d1)] 
reciever_block2 = [stub for stub in reciever_stubs if (reciever_node_degree_dict[stub.get_node()]<G*d1)]


# In[23]:


degree_transmitter = sum(transmitter_node_degree_dict.values())
degree_reciever = sum(reciever_node_degree_dict.values())


# In[24]:


# print(degree_transmitter)
# print(degree_reciever)
# print(len(transmitter_block1))
# print(len(transmitter_block2))
# print(len(reciever_block1))
# print(len(reciever_block2))


# 指定type

# In[25]:


b1_type1_count = int(len(transmitter_block1) * (p2/(p1+p2))) # type 1數量
b2_type1_count = int(len(transmitter_block2) * (p2/(1-p1-p2))) # type 1數量


# In[26]:


transmitter_block1_type1_stubs = random.sample(transmitter_block1, b1_type1_count)
transmitter_block2_type1_stubs = random.sample(transmitter_block2, b2_type1_count)
for stubs in transmitter_block1_type1_stubs:
    stubs.stubtype = 1
for stubs in transmitter_block2_type1_stubs:
    stubs.stubtype = 1
transmitter_block1_type2_stubs = [x for x in transmitter_block1 if (x.get_stubtype() == 2)]
transmitter_block2_type2_stubs = [x for x in transmitter_block2 if (x.get_stubtype() == 2)]


# In[27]:


reciever_block1_type1_stubs = random.sample(reciever_block1, b1_type1_count)
reciever_block2_type1_stubs = random.sample(reciever_block2, b2_type1_count)
for stubs in reciever_block1_type1_stubs:
    stubs.stubtype = 1
for stubs in reciever_block2_type1_stubs:
    stubs.stubtype = 1
reciever_block1_type2_stubs = [x for x in reciever_block1 if (x.get_stubtype() == 2)]
reciever_block2_type2_stubs = [x for x in reciever_block2 if (x.get_stubtype() == 2)]


# In[28]:


def remove_stubs(stub_list1, stub_list2):
    x = len(stub_list1) - len(stub_list2)
    if (x>0):
        for i in range (0,x):
            stub_list1.remove(random.choice(stub_list1))
    elif (x<0):
        for i in range (0,abs(x)):
            stub_list2.remove(random.choice(stub_list2))


# In[29]:


# print(len(transmitter_block1_type1_stubs))
# print(len(transmitter_block1_type2_stubs))
# print(len(transmitter_block2_type1_stubs))
# print(len(transmitter_block2_type2_stubs))
# print(len(reciever_block1_type1_stubs))
# print(len(reciever_block1_type2_stubs))
# print(len(reciever_block2_type1_stubs))
# print(len(reciever_block2_type2_stubs))


# In[30]:


remove_stubs(transmitter_block1_type1_stubs, reciever_block2_type1_stubs)
remove_stubs(transmitter_block1_type2_stubs, reciever_block1_type2_stubs)
remove_stubs(transmitter_block2_type1_stubs, reciever_block1_type1_stubs)
remove_stubs(transmitter_block2_type2_stubs, reciever_block2_type2_stubs)


# In[31]:


# print(len(transmitter_block1_type1_stubs))
# print(len(transmitter_block1_type2_stubs))
# print(len(transmitter_block2_type1_stubs))
# print(len(transmitter_block2_type2_stubs))
# print(len(reciever_block1_type1_stubs))
# print(len(reciever_block1_type2_stubs))
# print(len(reciever_block2_type1_stubs))
# print(len(reciever_block2_type2_stubs))


# In[32]:


for stubs in transmitter_block1_type1_stubs:
    stubs.stubtype = 1
    stubs.block = 1
for stubs in transmitter_block2_type1_stubs:
    stubs.stubtype = 1
    stubs.block = 2
    
for stubs in transmitter_block1_type2_stubs:
    stubs.stubtype = 2
    stubs.block = 1
    
for stubs in transmitter_block2_type2_stubs:
    stubs.stubtype = 2
    stubs.block = 2
    
for stubs in reciever_block1_type1_stubs:
    stubs.stubtype = 1
    stubs.block = 1
    
for stubs in reciever_block2_type1_stubs:
    stubs.stubtype = 1
    stubs.block = 2

for stubs in reciever_block1_type2_stubs:
    stubs.stubtype = 2
    stubs.block = 1
    
for stubs in reciever_block2_type2_stubs:
    stubs.stubtype = 2
    stubs.block = 2


# In[33]:


transmitter_stubs = transmitter_block1_type1_stubs + transmitter_block1_type2_stubs +transmitter_block2_type1_stubs + transmitter_block2_type2_stubs


# ### Step 5

# In[34]:


for i in tqdm(range(0,len(transmitter_stubs))): # while there are unconnected stubs
    stub_t = random.choice(transmitter_stubs) # Randomly select a stub from transmitter
    v = stub_t.get_node()
    transmitter_stubs.remove(stub_t) # to prevent seclecting the same stub
    
    if (stub_t.get_block() == 1):
        if (stub_t.get_stubtype() == 1): # block 1 type 1
            stub_r = random.choice(reciever_block2_type1_stubs)
            w = stub_r.get_node()
            reciever_block2_type1_stubs.remove(stub_r) # to prevent seclecting the same stub
        elif (stub_t.get_stubtype() == 2): # block 1 type 2
            stub_r = random.choice(reciever_block1_type2_stubs)
            w = stub_r.get_node()
            reciever_block1_type2_stubs.remove(stub_r)
            transmitter_block1_type2_stubs.remove(stub_t)
    elif (stub_t.get_block() == 2):
        if (stub_t.get_stubtype() == 1): # block 2 type 1
            stub_r = random.choice(reciever_block1_type1_stubs)
            w = stub_r.get_node()
            reciever_block1_type1_stubs.remove(stub_r)
        elif (stub_t.get_stubtype() == 2): # block 2 type 2
            stub_r = random.choice(reciever_block2_type2_stubs)
            w = stub_r.get_node()
            reciever_block2_type2_stubs.remove(stub_r)
        
    # connect the edges
    model.add_edge(v, w)


# In[35]:


nx.write_edgelist(model, model_dir + "model.txt", data=False)


# In[36]:


del transmitter_node_degree_dict, reciever_node_degree_dict
del transmitter_stubs, reciever_stubs, transmitter_block1, transmitter_block2, reciever_block1, reciever_block2
del transmitter_block1_type1_stubs, transmitter_block1_type2_stubs, transmitter_block2_type1_stubs
del transmitter_block2_type2_stubs, reciever_block1_type1_stubs, reciever_block1_type2_stubs
del reciever_block2_type1_stubs, reciever_block2_type2_stubs
del b1_ratio, index, degree_transmitter, degree_reciever, nodes, stubs, degree, stub_t, stub_r, v, w


# ### Step 6
# 1. Degree Correlation and covariance

# In[37]:


xl=[]
yl=[]
xl_square = []
yl_square = []
deg_sum = 0
for (v, w) in model.edges():
    deg_sum = deg_sum + model.degree[v] * model.degree[w]
    xl.append(model.degree[v])
    yl.append(model.degree[w])
    xl_square.append(model.degree[v]*model.degree[v])
    yl_square.append(model.degree[w]*model.degree[w])

EX = np.mean(xl)
EY = np.mean(yl)
EX_square = np.mean(xl_square)
EY_square = np.mean(yl_square)
EXY = deg_sum / len(model.edges)
sigma_X = math.sqrt(EX_square - EX*EX)
sigma_Y = math.sqrt(EY_square - EY*EY)

cov = EXY - EX * EY
cor = cov / (sigma_X * sigma_Y)


# In[38]:


del xl, yl, xl_square, yl_square


# plot the bipartite graph

# In[39]:


# top = nx.bipartite.sets(model)[0]
# pos = nx.bipartite_layout(model, top)
# plt.figure(figsize=(12,12)) 
# nx.draw(model, pos=pos)
# plt.show()


# ### 結果寫入檔案

# In[40]:


with open(cov_dir + 'cov.csv',  'a', newline = '') as csvFile:
    csvWriter = csv.writer(csvFile, delimiter = ',')
    csvWriter.writerow([cor, cov])


# ### 這些是等一下要用到的東西！！！！

# In[41]:


mt1 = len([e for e in list(model.edges()) if (model.degree(e[0])==d1)]) # E中原本transmitter端degree為d的邊
mt2 = len([e for e in list(model.edges()) if (model.degree(e[0])==d2)]) # E中原本trans端degree為2d的邊
mr1 = len([e for e in list(model.edges()) if (model.degree(e[1])==G*d1)]) # E中原本receiver端degree為d的邊
mr2 = len([e for e in list(model.edges()) if (model.degree(e[1])==G*d2)]) # E中原本receiver端degree為2d的邊


# ## 三. BEC channel
# BEC channel會刪除$G$中transmitter端比例為$\delta$的node $V_\delta$，以及與其相連之edge $E_\delta=\{(v,w)|v\in V_\delta, w\in R\}$
# 
# LDPC decode這些被刪除的點是錯誤更正碼欲更正的對象(換句話說，$\delta=0$的時候代表沒有掉包,decode成功率為100\%)
# 故decode的對象是$G_\delta=(V_\delta, E_\delta)$，方法是對$G_\delta$進行SIC。為了加快實驗進度，每產生一張圖，就每個delta值去跑100次實驗計算平均後再寫入檔案。三與四兩部份的code是只取一個$\delta$值作為示範和debug用。遍歷所有$\delta$值的程式在最後面。

# In[42]:


# delta = 0.4


# In[43]:


# transmitter_nodes_model = [x for x in list(model.nodes()) if isinstance(x, int)]
# reciever_nodes_model = [x for x in list(model.nodes()) if isinstance(x, str)]

# V_delta = [] # 丟銅板決定要不要加入某個node
# for v in transmitter_nodes_model:
#     p=random.random()
#     if p<=delta:
#         V_delta.append(v)
# E_delta = model.edges(V_delta)

# G_delta = nx.Graph()
# G_delta.add_nodes_from(V_delta, bipartite=0)
# G_delta.add_nodes_from(reciever_nodes_model, bipartite=1)
# G_delta.add_edges_from(E_delta)


# ## 四. LDPC Decode
# ### Step 1. 蒐集一些等等算metric時要用的串列

# In[44]:


# transmitter_nodes = [x for x in list(G_delta.nodes()) if isinstance(x, int)]
# reciever_nodes = [x for x in list(G_delta.nodes()) if isinstance(x, str)]


# In[45]:


# m = len(model.edges()) # 邊的總數
# trans_d1_edge = [e for e in list(E_delta) if (model.degree(e[0])==d1)] # E_\delta中原本transmitter端degree為d的邊
# trans_d2_edge = [e for e in list(E_delta) if (model.degree(e[0])==d2)] # E_\delta中原本trans端degree為2d的邊
# rec_Gd1_edge = [e for e in list(E_delta) if (model.degree(e[1])==G*d1)] # E_\delta中原本receiver端degree為d的邊
# rec_Gd2_edge = [e for e in list(E_delta) if (model.degree(e[1])==G*d2)] # E_\delta中原本receiver端degree為2d的邊
# trans_d1_node = [x for x in transmitter_nodes if (model.degree(x)==d1)] # V_\delta中transmitter端degree為d的點
# trans_d2_node = [x for x in transmitter_nodes if (model.degree(x)==d2)] # V_\delta中transmitter端degree為2d的點


# ### Step 2. SIC演算法
# SIC演算法的步驟：
# 1. 找出reciever中所有degree為1的點
# 2. 若$w$是這樣的點,假設他唯一的那條邊叫做$(v,w)$,$v$為transmitter上的點,那麼就把所有過$v$之邊刪除
# 3. 重複1,2直到無法再刪除,也就是reciever端不再有degree為1的點
# 
# 如同Poisson Reciever一樣去執行SIC(是對$G_\delta$進行)

# In[46]:


# deg1_r = len([x for x in reciever_nodes if (G_delta.degree(x)==1)]) # 當下reciver中degree為1的點的總數
# deg1_r


# In[47]:


# deg1_r = len([x for x in reciever_nodes if (G_delta.degree(x)==1)]) # 當下reciver中degree為1的點的總數
# while(deg1_r != 0):
#     for node in reciever_nodes:
#         if (G_delta.degree(node) == 1): # 找出reciever中degree為1的點
#             w, v = list(G_delta.edges([node]))[0] # 令這條唯一的邊叫做(w,v),v是transmitter上的點,w是reciever上的點
#             remove = list(G_delta.edges(v))
#             G_delta.remove_edges_from(remove) # 刪除這些邊
            
#     deg1_r = len([x for x in reciever_nodes if (G_delta.degree(x)==1)]) # update


# ### Step 3. 計算各個Metric
# 在此實驗中，metric一共有9個
# 1. $\lim_{i\to\infty}\alpha_d^{(i)}$，物理意義是當執行完SIC演算法後，任選一邊$e$出來，設其transmitter端為$v$。若未執行SIC前$\deg(v)=d$，則執行後$e$的 variable (transmitter)端未被decode之機率。故計算公式為$$\frac{\text{$E$中原本transmitter端degree為$d$的邊在執行完SIC後，transmitter端的degree沒有變成0，這樣的邊的個數}}{\text{$E$中原本transmitter端degree為$d$的邊}}$$
# 2. $\lim_{i\to\infty}\alpha_{2d}^{(i)}$，物理意義是當執行完SIC演算法後，任選一邊$e$出來，設其transmitter端為$v$。若未執行SIC前$\deg(v)=2d$，則執行後$e$的 variable (transmitter)端未被decode之機率。故計算公式為$$\frac{\text{$E_\delta$中原本transmitter端degree為$2d$的邊在執行完SIC後，transmitter端的degree沒有變成0，這樣的邊的個數}}{\text{$E$中原本transmitter端degree為$2d$的邊}}$$
# 3. $\lim_{i\to\infty}\alpha^{(i)}$，物理意義是當執行完SIC演算法後，任選一邊$e$的variable (transmitter)端未被decode之機率。故計算公式為$$\frac{\text{$E_\delta$在執行完SIC後，transmitter端的degree沒有變成0，這樣的邊的個數}}{\text{未執行SIC與BEC前，原本的圖$G$上的邊總數}}$$
# 4. $\lim_{i\to\infty}\beta_d^{(i)}$，物理意義是當執行完SIC演算法後，任選一邊$e$出來，設其receiver端為$w$。若未執行SIC前$\deg(w)=d$，則執行後$e$的 check (receiver)端未被decode之機率。故計算公式為$$\frac{\text{$E_\delta$中原本receiver端degree為$d$的邊在執行完SIC後，receiver端的degree沒有變成0，這樣的邊的個數}}{\text{$E$中原本transmitter端degree為$d$的邊}}$$
# 5. $\lim_{i\to\infty}\beta_{2d}^{(i)}$，物理意義是當執行完SIC演算法後，任選一邊$e$出來，設其receiver端為$w$。若未執行SIC前$\deg(w)=2d$，則執行後$e$的 check (receiver)端未被decode之機率。故計算公式為$$\frac{\text{$E_\delta$中原本receiver端degree為$2d$的邊在執行完SIC後，receiver端的degree沒有變成0，這樣的邊的個數}}{\text{$E$中原本transmitter端degree為$2d$的邊}}$$
# 6. $\lim_{i\to\infty}\beta^{(i)}$，物理意義是當執行完SIC演算法後，任選一邊$e$的check (receiver)端未被decode之機率。故計算公式為$$\frac{\text{$E_\delta$在執行完SIC後，receiver端的degree沒有變成0，這樣的邊的個數}}{\text{未執行SIC與BEC前，原本的圖$G$上的邊總數}}$$
# 7. $\lim_{i\to\infty}\gamma_{d}^{(i)}$，物理意義是當執行完SIC演算法後，任選一node $v$出來，若未執行SIC前$\deg(v)=d$，則執行後$v$成功被decode之機率。故計算公式為$$1-\frac{\text{$V_\delta$中原本transmitter端degree為$d$的點在執行完SIC後，degree沒有變成0，這樣的點的個數}}{\text{原本的圖$G$上transmitter端degree為$d$的點總數}}$$
# 8. $\lim_{i\to\infty}\gamma_{2d}^{(i)}$，物理意義是當執行完SIC演算法後，任選一node $v$出來，若未執行SIC前$\deg(v)=2d$，則執行後$v$成功被decode之機率。故計算公式為$$1-\frac{\text{$V_\delta$中原本transmitter端degree為$2d$的點在執行完SIC後，degree沒有變成0，這樣的點的個數}}{\text{原本的圖$G$上transmitter端degree為$2d$的點總數}}$$
# 9. $\lim_{i\to\infty}\gamma^{(i)}$，物理意義是當執行完SIC演算法後，任選一node $v$成功被decode之機率。故計算公式為$$1-\frac{\text{$V_\delta$中原本transmitter端的點在執行完SIC後，degree沒有變成0，這樣的點的個數}}{\text{原本的圖$G$上transmitter端的點總數}}$$

# In[48]:


# ad1_limit = len([x for x in trans_d1_edge if (G_delta.degree(x[0])!=0)]) / mt1
# ad2_limit = len([x for x in trans_d2_edge if (G_delta.degree(x[0])!=0)]) / mt2
# alpha = len([x for x in E_delta if (G_delta.degree(x[0])!=0)]) / m
# bGd1_limit = len([x for x in rec_Gd1_edge if (G_delta.degree(x[1])!=0)]) / mr1
# bGd2_limit = len([x for x in rec_Gd2_edge if (G_delta.degree(x[1])!=0)]) / mr2
# beta = len([x for x in E_delta if (G_delta.degree(x[1])!=0)]) / m
# yd1_limit = 1 - len([x for x in trans_d1_node if (G_delta.degree(x)!=0)]) / len(trans_d1_node)
# yd2_limit = 1 - len([x for x in trans_d2_node if (G_delta.degree(x)!=0)]) / len(trans_d2_node)
# gamma = 1 - len([x for x in transmitter_nodes if (G_delta.degree(x)!=0)]) / N


# ### Step 4. 將結果寫入檔案

# In[49]:


# with open(result_dir  + str(f'{delta:.4f}') + '.csv', 'a', newline = '') as csvFile:
#     csvWriter = csv.writer(csvFile, delimiter = ',')
#     csvWriter.writerow([ad1_limit, ad2_limit, alpha, bGd1_limit, bGd2_limit, beta, yd1_limit, yd2_limit, gamma])


# ## 五. 遍歷不同的$\delta$值
# 接下來的實驗要讓此圖通過BEC channel，並就不同$\delta$值為橫軸繪圖，且一張圖上會以q值不同繪製出不同的線。故我們需要將每次的實驗結果存到不同q值的資料夾內。因為每次($q$值不同時)實驗都會重複100遍，我們就這100張圖，每張圖都把所有$\delta值$做1次實驗。這樣一來對每個$\delta$值而言，每個$q$值都有100個來自不同圖的實驗結果，不會有統計上bias之問題。

# In[50]:


for i in tqdm(range(0,100)):
    for delta in deltas:
        # BEC channel
        transmitter_nodes_model = [x for x in list(model.nodes()) if isinstance(x, int)]
        reciever_nodes_model = [x for x in list(model.nodes()) if isinstance(x, str)]

        V_delta = [] # 丟銅板決定要不要加入某個node
        for v in transmitter_nodes_model:
            p=random.random()
            if p<=delta:
                V_delta.append(v)
        E_delta = model.edges(V_delta)

        G_delta = nx.Graph()
        G_delta.add_nodes_from(V_delta, bipartite=0)
        G_delta.add_nodes_from(reciever_nodes_model, bipartite=1)
        G_delta.add_edges_from(E_delta)

        # LDPC Decode
        # Step 1. 蒐集一些等等算metric時要用的串列
        transmitter_nodes = [x for x in list(G_delta.nodes()) if isinstance(x, int)]
        reciever_nodes = [x for x in list(G_delta.nodes()) if isinstance(x, str)]

        m = len(model.edges()) # 邊的總數
        trans_d1_edge = [e for e in list(E_delta) if (model.degree(e[0])==d1)] # E_\delta中原本transmitter端degree為d的邊
        trans_d2_edge = [e for e in list(E_delta) if (model.degree(e[0])==d2)] # E_\delta中原本trans端degree為2d的邊
        rec_Gd1_edge = [e for e in list(E_delta) if (model.degree(e[1])==G*d1)] # E_\delta中原本receiver端degree為d的邊
        rec_Gd2_edge = [e for e in list(E_delta) if (model.degree(e[1])==G*d2)] # E_\delta中原本receiver端degree為2d的邊
        trans_d1_node = [x for x in transmitter_nodes if (model.degree(x)==d1)] # V_\delta中transmitter端degree為d的點
        trans_d2_node = [x for x in transmitter_nodes if (model.degree(x)==d2)] # V_\delta中transmitter端degree為2d的點
        
        # Step 2. SIC演算法
        deg1_r = len([x for x in reciever_nodes if (G_delta.degree(x)==1)]) # 當下reciver中degree為1的點的總數
        while(deg1_r != 0):
            for node in reciever_nodes:
                if (G_delta.degree(node) == 1): # 找出reciever中degree為1的點
                    w, v = list(G_delta.edges([node]))[0] # 令這條唯一的邊叫做(w,v),v是transmitter上的點,w是reciever上的點
                    remove = list(G_delta.edges(v))
                    G_delta.remove_edges_from(remove) # 刪除這些邊
            
            deg1_r = len([x for x in reciever_nodes if (G_delta.degree(x)==1)]) # update

        # Step 3. 計算各個Metric
        ad1_limit = len([x for x in trans_d1_edge if (G_delta.degree(x[0])!=0)]) / mt1
        ad2_limit = len([x for x in trans_d2_edge if (G_delta.degree(x[0])!=0)]) / mt2
        alpha = len([x for x in E_delta if (G_delta.degree(x[0])!=0)]) / m
        bGd1_limit = len([x for x in rec_Gd1_edge if (G_delta.degree(x[1])!=0)]) / mr1
        bGd2_limit = len([x for x in rec_Gd2_edge if (G_delta.degree(x[1])!=0)]) / mr2
        beta = len([x for x in E_delta if (G_delta.degree(x[1])!=0)]) / m
        yd1_limit = 1 - len([x for x in trans_d1_node if (G_delta.degree(x)!=0)]) / len(trans_d1_node)
        yd2_limit = 1 - len([x for x in trans_d2_node if (G_delta.degree(x)!=0)]) / len(trans_d2_node)
        gamma = 1 - len([x for x in transmitter_nodes if (G_delta.degree(x)!=0)]) / N

        # Step 4. 將結果寫入檔案
        with open(result_dir  + str(f'{delta:.4f}') + '.csv', 'a', newline = '') as csvFile:
            csvWriter = csv.writer(csvFile, delimiter = ',')
            csvWriter.writerow([ad1_limit, ad2_limit, alpha, bGd1_limit, bGd2_limit, beta, yd1_limit, yd2_limit, gamma])

