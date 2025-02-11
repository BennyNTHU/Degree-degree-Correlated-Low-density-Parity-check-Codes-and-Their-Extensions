#!/usr/bin/env python
# coding: utf-8

# # LDPC Decode with 2 degree

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


# q = float(sys.argv[1])
# d = int(sys.argv[2])
# G = int(sys.argv[3])
# N = int(sys.argv[4])
# deltas = ast.literal_eval(sys.argv[5])


# In[3]:


q = 0.2 # type 1 stub ratio
d = 3
G = 3
N = 18000
deltas = [0.3] # delta 會跑哪些值
result_dir = './'


# In[4]:


n = N # Number of transmitter, i.e., n
M = int(N / G) # Number of reciever, i.e., n-k
k = N - M


# In[5]:


random.seed(time.time()) # random seeds
# if not os.path.exists(result_dir) : os.mkdir(result_dir)
# if not os.path.exists(result_dir + str(q) + '/') : os.mkdir(result_dir + str(q) + '/')


# ## 二. 建立有Negative Degree Correlation的Bipartite Graph
# 
# 實作步驟：
# 1. 設有一個bipartite graph，左邊的頂點為transmitter $T$，右邊為reciever $R$，數量分別為$n$和$n-k$
# 2. 建立degree sequence $t_k$給tansmitter，作為stub的數量。degree為$2d$的共$n/3$個,degree為$d$的共$2n/3$個,其後,建立degree sequence $r_k$給reciever，作為stub的數量。degree為$2Gd$的共$(n-k)/3$個,degree為$Gd$的共$2(n-k)/3$個,其中$G=n/(n-k)$
# 
#     (1) 計算$t_k$的和$m_T$與平均值$\lambda_t$
#     
#     (2) 以$\lambda_t$作為seed生成另一個Poisson degree sequence $r_k$給reciever，作為stub的數量
#     
#     (3) 計算$r_k$之和$m_R$，$m_T$與$m_R$不相等時，隨機從$r_k$挑出部分stub進行修改
# 
# 3. 將此bipartite graph建立起來，但先不連結邊
# 4. 建立Stubs
# 
#     (1) 隨機指定$T$中比例為$q$的stub為type 1 stub，其餘為type2 stub。
#     
#     (2) 將$T$中的stub隨機等分成兩個block
#     
#     (3) 對$R$亦如是
#     
# 5. 將邊連起來：自$T$中任意選一stub，
# 
#     (1) 若是block 1中的type 1 stub，就連到$R$中block 2的type 1 stub。若是type 2 stub就連到$R$中任意的type 2 stub。
#     
#     (2) 若是block 2中的type 1 stub，就連到$R$中block 1的type 1 stub。若是type 2 stub就連到$R$中任意的type 2 stub。
#     
# 6. 重複此實驗100次，畫2張圖：x軸是$q$從0取到1間隔0.1，這11個值。
# 
#     (1) 第一張圖的y軸是相對應的cov的值
#     
#     (2) 第二張圖的y軸是cor的值

# ### Step 2

# 製造兩個degree sequence

# In[6]:


tk = [2*d] * int(N/3) + [d] * int(2*N/3)
rk = [2*G*d] * int(M/3) + [G*d] * int(2*M/3)


# 若有degree為0的點，則補一個stub給它(in this case必無此點,無檢查必要)

# In[7]:


# tk = [1 if i==0 else i for i in tk]
# rk = [1 if i==0 else i for i in rk]


# 讓所有degree的和相同(in this case必相同,無檢查必要)

# In[8]:


# n = sum(tk) - sum(rk)
# if (n < 0): # transmitter端stub比較少
#     for i in range(0,-n):
#         index = random.choice(list(range(0,N)))
#         tk[index] = tk[index] + 1
# elif (n > 0): # reiever端stub比較少
#     for i in range(0,n):
#         index = random.choice(list(range(0,M)))
#         rk[index] = rk[index] + 1
# else:
#     pass


# 調整degree總數為偶數

# In[9]:


if (sum(tk)%2 != 0): # 
    index = random.choice(list(range(0,N)))
    tk[index] = tk[index] + 1
    index = random.choice(list(range(0,M)))
    rk[index] = rk[index] + 1


# 經過這些處理之後，transmitter端與reciever端具有相同的stub數(degree)以外，也能確保整個bipartite graph的degree是偶數

# In[10]:


rk = sorted(rk, reverse=True) # sort 2m stubs in descending order
tk = sorted(tk, reverse=True)
degree_transmitter = sum(tk)
degree_reciever = sum(rk)


# In[11]:


# print(np.unique(tk))
# print(np.unique(rk))
# print(degree_transmitter) # 理論上要有幾條邊
# print(degree_reciever)


# ### Step 3
# 現在將此bipartite graph建立起來，且已經保證所有頂點的degree不為0

# In[12]:


transmitter_node_degree_dict = {}
for nodes in range(0,N): # The nodes are named after 0~N-1
    transmitter_node_degree_dict[nodes] = tk[nodes]


# In[13]:


# transmitter_node_degree_dict # degree sequence


# In[14]:


reciever_node_degree_dict = {}
for nodes in range(0,M): # The nodes are named after '0'~'N-1'
    reciever_node_degree_dict[str(nodes)] = rk[nodes]


# In[15]:


model = nx.Graph()
model.add_nodes_from(transmitter_node_degree_dict.keys(), bipartite=0)
model.add_nodes_from(reciever_node_degree_dict.keys(), bipartite=1)


# In[16]:


del tk, rk


# ### Step 4

# 建立transmitter和reciever的stub

# In[17]:


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


# In[18]:


transmitter_stubs = []
for nodes in transmitter_node_degree_dict.keys():
    for degree in range(0, transmitter_node_degree_dict[nodes]):
        transmitter_stubs.append(Stubs(nodes, 2, 1))


# In[19]:


reciever_stubs = []
for nodes in reciever_node_degree_dict.keys():
    for degree in range(0, reciever_node_degree_dict[nodes]):  
        reciever_stubs.append(Stubs(nodes, 2, 1))


# 指定block。這個過程中必須遵守三個原則：
# 1. 同一個頂點的所有stub必須位於同一stub
# 2. degree相同的所有頂點的所有stub必須位於同一個block之中
# 3. 每個block的stub總數仍要相同
# 
# 若有不符合之處可再加stub，但必須滿足stub的總數必須是偶數的原則。
# 
# 首先我們先滿足條件1與2，再來執行條件3。滿足條件1的方式是先選出pivot這個stub，pivot是第m/2個stub。接著，把所有跟pivot在同一node上的stub加進block 1中，也把degree與該頂點相同的所有頂點上的所有stub加進block 1中，剩下的加入block 2。因此pivot會位於block 1之中。

# In[20]:


pivot_index = int(degree_transmitter / 2)
pivot = transmitter_stubs[pivot_index-1] # 第m/2個stub
transmitter_block1 = transmitter_stubs[0:pivot_index] # 屬於degree較大的點
transmitter_block2 = transmitter_stubs[pivot_index:] # 屬於degree較小的點


# 執行原則1

# In[21]:


transmitter_block1.extend(list(filter(lambda x: (x.get_node() == pivot.get_node()), transmitter_block2)))
transmitter_block2 = list(filter(lambda x: (x.get_node() != pivot.get_node()), transmitter_block2))


# 執行原則2與3

# In[22]:


stubs_degree_toolarge = list(filter(lambda x: (transmitter_node_degree_dict[x.get_node()] 
                                               == transmitter_node_degree_dict[pivot.get_node()]), 
                                    transmitter_block2)) 
nodes_too_large = np.unique([stubs.get_node() for stubs in stubs_degree_toolarge])
# 等等需要增加的stub數目x=要刪掉的stub數+執行原則1後所需補的數目
x = len(nodes_too_large) + len(transmitter_block1) - len(transmitter_block2)


# In[23]:


# 此stub所連的node的degree跟pivot一樣大
for stubs in transmitter_block2: # 自所有degree跟pivot一樣大的點刪掉一個stub
    this = stubs.get_node()
    if ((this in nodes_too_large) and 
        (transmitter_node_degree_dict[this] == transmitter_node_degree_dict[pivot.get_node()])):
        transmitter_node_degree_dict[this] -= 1 # update degree sequence
        transmitter_block2.remove(stubs) # 刪除的動作


# In[24]:


b = [] # 候選點
block_2_node = np.unique([stubs.get_node() for stubs in transmitter_block2])
for nodes in block_2_node:
    if ((transmitter_node_degree_dict[pivot.get_node()] - transmitter_node_degree_dict[nodes] > 1) and 
        (transmitter_node_degree_dict[pivot.get_node()] - transmitter_node_degree_dict[nodes] < 4)):
        # degree小2的可以補一個stub,degree小3的可以補兩個stub
        for i in range(0, transmitter_node_degree_dict[pivot.get_node()] - 
                       transmitter_node_degree_dict[nodes] - 1):
            b.append(Stubs(nodes, 2, 1))


# In[25]:


#assert (x < len(b)), 因為只有兩種degree,絕不會失敗


# In[26]:


while (x > 0 and len(b) > 0):
    i = random.choice(list(range(0, len(b))))
    # 自block 2隨機抽出一stub，且保證此stub所屬之點的degree之值為(pivot所屬之點的degree-2 or degree-3)
    this = b[i].get_node() 
    transmitter_node_degree_dict[this] += 1 # update degree sequence
    transmitter_block2.append(Stubs(this, 2, 1)) # 增加一個stub
    x = x - 1
    b.pop(i)


# In[27]:


for stubs in transmitter_block2:
    stubs.block = 2


# 對Reciever進行一樣的操作

# In[28]:


pivot_index = int(degree_reciever / 2)
pivot = reciever_stubs[pivot_index-1] # 第m/2個stub
reciever_block1 = reciever_stubs[0:pivot_index] # 屬於degree較大的點
reciever_block2 = reciever_stubs[pivot_index:] # 屬於degree較小的點

reciever_block1.extend(list(filter(lambda x: (x.get_node() == pivot.get_node()), reciever_block2)))
reciever_block2 = list(filter(lambda x: (x.get_node() != pivot.get_node()), reciever_block2))

stubs_degree_toolarge = list(filter(lambda x: (reciever_node_degree_dict[x.get_node()] 
                                               == reciever_node_degree_dict[pivot.get_node()]), 
                                    reciever_block2)) 
nodes_too_large = np.unique([stubs.get_node() for stubs in stubs_degree_toolarge])
# 等等需要增加的stub數目x=要刪掉的stub數+執行原則1後所需補的數目
x = len(nodes_too_large) + len(reciever_block1) - len(reciever_block2)

# 此stub所連的node的degree跟pivot一樣大
for stubs in reciever_block2: # 自所有degree跟pivot一樣大的點刪掉一個stub
    this = stubs.get_node()
    if ((this in nodes_too_large) and 
        (reciever_node_degree_dict[this] == reciever_node_degree_dict[pivot.get_node()])):
        reciever_node_degree_dict[this] -= 1 # update degree sequence
        reciever_block2.remove(stubs) # 刪除的動作
        
b = [] # 候選點
block_2_node = np.unique([stubs.get_node() for stubs in reciever_block2])
for nodes in block_2_node:
    if ((reciever_node_degree_dict[pivot.get_node()] - reciever_node_degree_dict[nodes] > 1) and 
        (reciever_node_degree_dict[pivot.get_node()] - reciever_node_degree_dict[nodes] < 4)):
        # degree小2的可以補一個stub,degree小3的可以補兩個stub
        for i in range(0, reciever_node_degree_dict[pivot.get_node()] - 
                       reciever_node_degree_dict[nodes] - 1):
            b.append(Stubs(nodes, 2, 1))
            
# assert (x < len(b)), 因為只有兩種degree,絕不會失敗

while (x > 0 and len(b) > 0):
    i = random.choice(list(range(0, len(b))))
    # 自block 2隨機抽出一stub，且保證此stub所屬之點的degree之值為(pivot所屬之點的degree-2 or degree-3)
    this = b[i].get_node() 
    reciever_node_degree_dict[this] += 1 # update degree sequence
    reciever_block2.append(Stubs(this, 2, 1)) # 增加一個stub
    x = x - 1
    b.pop(i)

for stubs in reciever_block2:
    stubs.block = 2


# In[29]:


del b, stubs_degree_toolarge, nodes_too_large


# 這時候transmitter的degree和reciever的degree不一定相同。調整的方法是：誰比較少就把誰補滿。假設今天是transmitter比較少：
# 1. block 1因為是大degree，所以亂補也沒關係
# 2. 設block 2的頂點中最大的degree是$d$，則只有補degree比$d$小的點可以補，可以補的數量是$d-$該點degree

# In[30]:


x = len(transmitter_block2) - len(reciever_block2) # 每個block要補多少


# In[31]:


if (x > 0): # reciever比較少
    block_1_node = np.unique([stubs.get_node() for stubs in reciever_block1]) 
    block_2_node = np.unique([stubs.get_node() for stubs in reciever_block2])
    
    # block 2不能亂補，必須要小於其中degree最大值的才可以補
    max_degree_block2 = max([reciever_node_degree_dict[nodes] for nodes in block_2_node]) # 找出block 2中最大的degree
    b2 = [] # 可以用來補的stub候選清單
    for nodes in block_2_node:
        if (reciever_node_degree_dict[nodes] < max_degree_block2): # 小於其中degree最大值的node才可以補
            for i in range(0, max_degree_block2 - reciever_node_degree_dict[nodes]): # 可以補的數量
                b2.append(Stubs(nodes, 2, 2)) # block 2

    assert (x < len(b2)), '實驗失敗 請重新生成圖片' # 若block 2無法補齊，則實驗失敗
    
    for i in range(0, x):
        i = random.choice(list(range(0, len(b2))))
        node = b2[i].get_node() # 自候選清單中隨機抽出一stub
        reciever_node_degree_dict[node] += 1 # update degree sequence
        reciever_block2.append(Stubs(node, 2, 2)) # 增加一個stub
        b2.pop(i)
    
    # block 1可以亂補
    for i in range(0, x):
        j = random.choice(list(range(0, len(block_1_node))))
        node = block_1_node[j] # 自block 1隨機抽出一node
        reciever_node_degree_dict[node] += 1 # update degree sequence
        reciever_block1.append(Stubs(node, 2, 1)) # 增加一個stub
        
    del b2, block_1_node, block_2_node


# In[32]:


if (x < 0): # transmitter比較少
    x = abs(x)
    block_1_node = np.unique([stubs.get_node() for stubs in transmitter_block1]) 
    block_2_node = np.unique([stubs.get_node() for stubs in transmitter_block2])
    
    # block 2不能亂補，必須要小於其中degree最大值的才可以補
    max_degree_block2 = max([transmitter_node_degree_dict[nodes] for nodes in block_2_node]) # 找出block 2中最大的degree
    b2 = [] # 可以用來補的stub候選清單
    for nodes in block_2_node:
        if (transmitter_node_degree_dict[nodes] < max_degree_block2): # 小於其中degree最大值的node才可以補
            for i in range(0, max_degree_block2 - transmitter_node_degree_dict[nodes]): # 可以補的數量
                b2.append(Stubs(nodes, 2, 2)) # block 2

    assert (x < len(b2)), '實驗失敗 請重新生成圖片' # 若block 2無法補齊，則實驗失敗
    
    for i in range(0, x):
        i = random.choice(list(range(0, len(b2))))
        node = b2[i].get_node() # 自候選清單中隨機抽出一stub
        transmitter_node_degree_dict[node] += 1 # update degree sequence
        transmitter_block2.append(Stubs(node, 2, 2)) # 增加一個stub
        b2.pop(i)
    
    # block 1可以亂補
    for i in range(0, x):
        j = random.choice(list(range(0, len(block_1_node))))
        node = block_1_node[j] # 自block 1隨機抽出一node
        transmitter_node_degree_dict[node] += 1 # update degree sequence
        transmitter_block1.append(Stubs(node, 2, 1)) # 增加一個stub
        
    del b2, block_1_node, block_2_node


# 更新degree

# In[33]:


degree_transmitter = sum(transmitter_node_degree_dict.values())
degree_reciever = sum(reciever_node_degree_dict.values())


# In[34]:


# print(degree_transmitter)
# print(degree_reciever)
# print(len(transmitter_block1))
# print(len(transmitter_block2))
# print(len(reciever_block1))
# print(len(reciever_block2))


# 指定type

# In[35]:


type1_count = int(len(transmitter_block1)*q) # type 1數量


# In[36]:


transmitter_block1_type1_stubs = random.sample(transmitter_block1, type1_count)
transmitter_block2_type1_stubs = random.sample(transmitter_block2, type1_count)
for stubs in transmitter_block1_type1_stubs:
    stubs.stubtype = 1
for stubs in transmitter_block2_type1_stubs:
    stubs.stubtype = 1
transmitter_type2_stubs = list(filter(lambda x: (x.get_stubtype() == 2), transmitter_block1+transmitter_block2))


# In[37]:


reciever_block1_type1_stubs = random.sample(reciever_block1, type1_count)
reciever_block2_type1_stubs = random.sample(reciever_block2, type1_count)
for stubs in reciever_block1_type1_stubs:
    stubs.stubtype = 1
for stubs in reciever_block2_type1_stubs:
    stubs.stubtype = 1
reciever_type2_stubs = list(filter(lambda x: (x.get_stubtype() == 2), reciever_block1+reciever_block2))


# In[38]:


# print(len(transmitter_block1_type1_stubs))
# print(len(transmitter_block2_type1_stubs))
# print(len(transmitter_type2_stubs))
# print(len(reciever_block1_type1_stubs))
# print(len(reciever_block2_type1_stubs))
# print(len(reciever_type2_stubs))


# In[39]:


transmitter_stubs = transmitter_block1_type1_stubs + transmitter_block2_type1_stubs + transmitter_type2_stubs


# ### Step 5

# In[40]:


for i in tqdm(range(0, degree_transmitter)): # while there are unconnected stubs
    stub_t = random.choice(transmitter_stubs) # Randomly select a stub from transmitter
    v = stub_t.get_node()
    transmitter_stubs.remove(stub_t) # to prevent seclecting the same stub
    
    if (stub_t.get_stubtype() == 1):
        if (stub_t.get_block() == 1):
            stub_r = random.choice(reciever_block2_type1_stubs)
            w = stub_r.get_node()
            reciever_block2_type1_stubs.remove(stub_r) # to prevent seclecting the same stub
        else:
            stub_r = random.choice(reciever_block1_type1_stubs)
            w = stub_r.get_node()
            reciever_block1_type1_stubs.remove(stub_r)
    else:
        stub_r = random.choice(reciever_type2_stubs)
        w = stub_r.get_node()
        reciever_type2_stubs.remove(stub_r)
        
    # connect the edges
    model.add_edge(v, w)


# In[41]:


del transmitter_stubs, reciever_stubs, transmitter_block1, transmitter_block2, reciever_block1, reciever_block2
del transmitter_block1_type1_stubs, transmitter_block2_type1_stubs, transmitter_type2_stubs, 
del transmitter_node_degree_dict, reciever_node_degree_dict
del reciever_block1_type1_stubs, reciever_block2_type1_stubs, reciever_type2_stubs


# ### Step 6
# 1. Degree Correlation and covariance

# In[42]:


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


# In[43]:


cov = EXY - EX * EY


# In[44]:


cor = cov / (sigma_X * sigma_Y)


# In[45]:


del xl, yl, xl_square, yl_square


# 2. $P(X_e=x|Y_e=y)$

# In[46]:


# trans_d = [e for e in list(model.edges()) if (model.degree(e[0])==d)] # V_\delta中transmitter端degree為d的點
# trans_2d = [e for e in list(model.edges()) if (model.degree(e[0])==2*d)] # V_\delta中transmitter端degree為2d的點
# rec_Gd = [e for e in list(model.edges()) if (model.degree(e[1])==G*d)] # E_\delta中原本receiver端degree為d的邊
# rec_2Gd = [e for e in list(model.edges()) if (model.degree(e[1])==2*G*d)] # E_\delta中原本receiver端degree為2d的邊


# In[47]:


# p1 = len([e for e in trans_2d if (model.degree(e[1])==2*G*d)]) / len(trans_2d)
# p2 = len([e for e in trans_d if (model.degree(e[1])==2*G*d)]) / len(trans_d)
# p3 = len([e for e in trans_2d if (model.degree(e[1])==G*d)]) / len(trans_2d)
# p4 = len([e for e in trans_d if (model.degree(e[1])==G*d)]) / len(trans_d)


# In[48]:


# print(p1,p2,p3,p4)


# plot the bipartite graph

# In[49]:


# top = nx.bipartite.sets(model)[0]
# pos = nx.bipartite_layout(model, top)
# plt.figure(figsize=(12,12)) 
# nx.draw(model, pos=pos)
# plt.show()


# ### 結果寫入檔案

# In[50]:


# with open(result_dir + str(q) + '.csv',  'a', newline = '') as csvFile:
#     csvWriter = csv.writer(csvFile, delimiter = ',')
#     csvWriter.writerow([cor, cov])


# ## 三. BEC channel
# BEC channel會刪除$G$中transmitter端比例為$\delta$的node $V_\delta$，以及與其相連之edge $E_\delta=\{(v,w)|v\in V_\delta, w\in R\}$
# 
# LDPC decode這些被刪除的點是錯誤更正碼欲更正的對象(換句話說，$\delta=0$的時候代表沒有掉包,decode成功率為100\%)
# 故decode的對象是$G_\delta=(V_\delta, E_\delta)$，方法是對$G_\delta$進行SIC。為了加快實驗進度，每產生一張圖，就每個delta值去跑100次實驗計算平均後再寫入檔案。三與四兩部份的code是只取一個$\delta$值作為示範和debug用。遍歷所有$\delta$值的程式在最後面。

# In[51]:


# delta = 0.4


# In[52]:


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

# In[53]:


# transmitter_nodes = [x for x in list(G_delta.nodes()) if isinstance(x, int)]
# reciever_nodes = [x for x in list(G_delta.nodes()) if isinstance(x, str)]


# In[54]:


# m = len(model.edges()) # 邊的總數
# trans_d_edge = [e for e in list(E_delta) if (model.degree(e[0])==d)] # E_\delta中原本transmitter端degree為d的邊
# trans_2d_edge = [e for e in list(E_delta) if (model.degree(e[0])==2*d)] # E_\delta中原本trans端degree為2d的邊
# rec_d_edge = [e for e in list(E_delta) if (model.degree(e[1])==G*d)] # E_\delta中原本receiver端degree為d的邊
# rec_2d_edge = [e for e in list(E_delta) if (model.degree(e[1])==2*G*d)] # E_\delta中原本receiver端degree為2d的邊
# trans_d_node = [x for x in transmitter_nodes if (model.degree(x)==d)] # V_\delta中transmitter端degree為d的點
# trans_2d_node = [x for x in transmitter_nodes if (model.degree(x)==2*d)] # V_\delta中transmitter端degree為2d的點


# ### Step 2. SIC演算法
# SIC演算法的步驟：
# 1. 找出reciever中所有degree為1的點
# 2. 若$w$是這樣的點,假設他唯一的那條邊叫做$(v,w)$,$v$為transmitter上的點,那麼就把所有過$v$之邊刪除
# 3. 重複1,2直到無法再刪除,也就是reciever端不再有degree為1的點
# 
# 如同Poisson Reciever一樣去執行SIC(是對$G_\delta$進行)

# In[55]:


# deg1_r = len([x for x in reciever_nodes if (G_delta.degree(x)==1)]) # 當下reciver中degree為1的點的總數
# deg1_r


# In[56]:


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
# 1. $\lim_{i\to\infty}\alpha_d^{(i)}$，物理意義是當執行完SIC演算法後，任選一邊$e$出來，設其transmitter端為$v$。若未執行SIC前$\deg(v)=d$，則執行後$e$的 variable (transmitter)端未被decode之機率。故計算公式為$$\frac{\text{$E_\delta$中原本transmitter端degree為$d$的邊在執行完SIC後，transmitter端的degree沒有變成0，這樣的邊的個數}}{\text{未執行SIC與BEC前，原本的圖$G$上的邊總數}}$$
# 2. $\lim_{i\to\infty}\alpha_{2d}^{(i)}$，物理意義是當執行完SIC演算法後，任選一邊$e$出來，設其transmitter端為$v$。若未執行SIC前$\deg(v)=2d$，則執行後$e$的 variable (transmitter)端未被decode之機率。故計算公式為$$\frac{\text{$E_\delta$中原本transmitter端degree為$2d$的邊在執行完SIC後，transmitter端的degree沒有變成0，這樣的邊的個數}}{\text{未執行SIC與BEC前，原本的圖$G$上的邊總數}}$$
# 3. $\lim_{i\to\infty}\alpha^{(i)}$，物理意義是當執行完SIC演算法後，任選一邊$e$的variable (transmitter)端未被decode之機率。故計算公式為$$\frac{\text{$E_\delta$在執行完SIC後，transmitter端的degree沒有變成0，這樣的邊的個數}}{\text{未執行SIC與BEC前，原本的圖$G$上的邊總數}}$$
# 4. $\lim_{i\to\infty}\beta_d^{(i)}$，物理意義是當執行完SIC演算法後，任選一邊$e$出來，設其receiver端為$w$。若未執行SIC前$\deg(w)=d$，則執行後$e$的 check (receiver)端未被decode之機率。故計算公式為$$\frac{\text{$E_\delta$中原本receiver端degree為$d$的邊在執行完SIC後，receiver端的degree沒有變成0，這樣的邊的個數}}{\text{未執行SIC與BEC前，原本的圖$G$上的邊總數}}$$
# 5. $\lim_{i\to\infty}\beta_{2d}^{(i)}$，物理意義是當執行完SIC演算法後，任選一邊$e$出來，設其receiver端為$w$。若未執行SIC前$\deg(w)=2d$，則執行後$e$的 check (receiver)端未被decode之機率。故計算公式為$$\frac{\text{$E_\delta$中原本receiver端degree為$2d$的邊在執行完SIC後，receiver端的degree沒有變成0，這樣的邊的個數}}{\text{未執行SIC與BEC前，原本的圖$G$上的邊總數}}$$
# 6. $\lim_{i\to\infty}\beta^{(i)}$，物理意義是當執行完SIC演算法後，任選一邊$e$的check (receiver)端未被decode之機率。故計算公式為$$\frac{\text{$E_\delta$在執行完SIC後，receiver端的degree沒有變成0，這樣的邊的個數}}{\text{未執行SIC與BEC前，原本的圖$G$上的邊總數}}$$
# 7. $\lim_{i\to\infty}\gamma_{d}^{(i)}$，物理意義是當執行完SIC演算法後，任選一node $v$出來，若未執行SIC前$\deg(v)=d$，則執行後$v$成功被decode之機率。故計算公式為$$1-\frac{\text{$V_\delta$中原本transmitter端degree為$d$的點在執行完SIC後，degree沒有變成0，這樣的點的個數}}{\text{原本的圖$G$上transmitter端的點總數}}$$
# 8. $\lim_{i\to\infty}\gamma_{2d}^{(i)}$，物理意義是當執行完SIC演算法後，任選一node $v$出來，若未執行SIC前$\deg(v)=2d$，則執行後$v$成功被decode之機率。故計算公式為$$1-\frac{\text{$V_\delta$中原本transmitter端degree為$2d$的點在執行完SIC後，degree沒有變成0，這樣的點的個數}}{\text{原本的圖$G$上transmitter端的點總數}}$$
# 9. $\lim_{i\to\infty}\gamma^{(i)}$，物理意義是當執行完SIC演算法後，任選一node $v$成功被decode之機率。故計算公式為$$1-\frac{\text{$V_\delta$中原本transmitter端的點在執行完SIC後，degree沒有變成0，這樣的點的個數}}{\text{原本的圖$G$上transmitter端的點總數}}$$

# In[57]:


# ad_limit = len([x for x in trans_d_edge if (G_delta.degree(x[0])!=0)]) / m
# a2d_limit = len([x for x in trans_2d_edge if (G_delta.degree(x[0])!=0)]) / m
# alpha = len([x for x in E_delta if (G_delta.degree(x[0])!=0)]) / m
# bd_limit = len([x for x in rec_d_edge if (G_delta.degree(x[1])!=0)]) / m
# b2d_limit = len([x for x in rec_2d_edge if (G_delta.degree(x[1])!=0)]) / m
# beta = len([x for x in E_delta if (G_delta.degree(x[1])!=0)]) / m
# yd_limit = 1 - len([x for x in trans_d_node if (G_delta.degree(x)!=0)]) / N
# y2d_limit = 1 - len([x for x in trans_2d_node if (G_delta.degree(x)!=0)]) / N
# gamma = 1 - len([x for x in transmitter_nodes if (G_delta.degree(x)!=0)]) / N


# ### Step 4. 將結果寫入檔案

# In[58]:


# with open(result_dir + str(q) + '/' + 'delta=' + str(delta) + '.csv', 'a', newline = '') as csvFile:
#     csvWriter = csv.writer(csvFile, delimiter = ',')
#     csvWriter.writerow([ad_limit, a2d_limit, alpha, bd_limit, b2d_limit, beta, yd_limit, y2d_limit, gamma])


# ## 五. 遍歷不同的$\delta$值
# 接下來的實驗要讓此圖通過BEC channel，並就不同$\delta$值為橫軸繪圖，且一張圖上會以q值不同繪製出不同的線。故我們需要將每次的實驗結果存到不同q值的資料夾內。因為每次($q$值不同時)實驗都會重複100遍，我們就這100張圖，每張圖都把所有$\delta值$做1次實驗。這樣一來對每個$\delta$值而言，每個$q$值都有100個來自不同圖的實驗結果，不會有統計上bias之問題。

# In[59]:



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
    trans_d_edge = [e for e in list(E_delta) if (model.degree(e[0])==d)] # E_\delta中原本trans端degree為d的邊
    trans_2d_edge = [e for e in list(E_delta) if (model.degree(e[0])==2*d)] # E_\delta中原本trans端degree為2d的邊
    rec_d_edge = [e for e in list(E_delta) if (model.degree(e[1])==G*d)] # E_\delta中原本receiver端degree為d的邊
    rec_2d_edge = [e for e in list(E_delta) if (model.degree(e[1])==2*G*d)] # E_\delta中原本rec端degree為2d的邊
    trans_d_node = [x for x in transmitter_nodes if (model.degree(x)==d)] # V_\delta中transmitter端degree為d的點
    trans_2d_node = [x for x in transmitter_nodes if (model.degree(x)==2*d)] # V_\delta中transdegree為2d的點

    # Step 2. SIC演算法
    deg1_r = len([x for x in reciever_nodes if (G_delta.degree(x)==1)]) # 當下reciver中degree為1的點的總數
    deg1_r_init = deg1_r
    iter_times = 0
    while(deg1_r != 0):
        for node in reciever_nodes:
            if (G_delta.degree(node) == 1): # 找出reciever中degree為1的點
                w, v = list(G_delta.edges([node]))[0] # 令這條唯一的邊叫做(w,v),v是trans上的點,w是reciever上的點
                remove = list(G_delta.edges(v))
                G_delta.remove_edges_from(remove) # 刪除這些邊

        deg1_r = len([x for x in reciever_nodes if (G_delta.degree(x)==1)]) # update
        iter_times += 1

    # Step 3. 計算各個Metric
    ad_limit = len([x for x in trans_d_edge if (G_delta.degree(x[0])!=0)]) / m
    a2d_limit = len([x for x in trans_2d_edge if (G_delta.degree(x[0])!=0)]) / m
    alpha = len([x for x in E_delta if (G_delta.degree(x[0])!=0)]) / m
    bd_limit = len([x for x in rec_d_edge if (G_delta.degree(x[1])!=0)]) / m
    b2d_limit = len([x for x in rec_2d_edge if (G_delta.degree(x[1])!=0)]) / m
    beta = len([x for x in E_delta if (G_delta.degree(x[1])!=0)]) / m
    yd_limit = 1 - len([x for x in trans_d_node if (G_delta.degree(x)!=0)]) / N
    y2d_limit = 1 - len([x for x in trans_2d_node if (G_delta.degree(x)!=0)]) / N
    gamma = 1 - len([x for x in transmitter_nodes if (G_delta.degree(x)!=0)]) / N

    # Step 4. 將結果寫入檔案
    with open(result_dir + 'q=' + str(q) + 'delta=' + str(delta) + '.csv', 'a', newline = '') as csvFile:
        csvWriter = csv.writer(csvFile, delimiter = ',')
        csvWriter.writerow([ad_limit, gamma, iter_times, deg1_r_init])

