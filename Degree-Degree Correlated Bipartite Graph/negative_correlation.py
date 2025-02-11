#!/usr/bin/env python
# coding: utf-8

# In[1]:


import networkx as nx
import random
import numpy as np
import time
import csv
import os
import sys
from tqdm import tqdm
from networkx.algorithms import bipartite
from numpy.random import poisson
import matplotlib.pyplot as plt


# In[2]:


random.seed(time.time()) # random seeds
if not os.path.exists("results/") : os.mkdir("results/")


# ## 建立有Negative Degree Correlation的Bipartite Graph
# 
# 實作步驟：
# 1. 設有一個bipartite graph，左邊的頂點為transmitter $T$，右邊為reciever $R$，數量均為10000
# 2. 建立一個Poisson degree sequence $t_k$給tansmitter，作為stub的數量
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
# 6. 重複此實驗100次，畫四張圖：x軸是$q$從0取到1間隔0.1，這11個值。
# 
#     (1) 第一張圖的y軸是相對應的cov的值
#     
#     (2) 第二張圖的y軸是cor的值
#     
#     (3) 第四張圖$\text{the ratio of giant component size}/(N+M)$
#     
#     (4) 第三張圖是SIC解出的指標，一個指標可以是：$$\frac{\text{成功解出的edge總數}}{\text{graph中edge的總數}}$$

# ## Issues:
# 1. 讓$N$可以不等於$M$(已解決)
# 2. 讓$q$可以取值在0或1(已解決)
# 3. degree是0的點要刪掉(已解決)
# 4. degree相同的點必須位於同一個block之中，但每個block的stub總數仍要相同(已解決)

# ## Step 1
# 設定實驗參數

# In[3]:


LAM = 5 # Poisson degree distribution
N = 10000 # Number of transmitter
M = 10000 # Number of reciever
q = float(sys.argv[1])
# q = 1 # type 1 stub ratio


# ### Step 2

# 製造兩個degree sequence

# In[4]:


tk = poisson(lam = LAM, size = N) # Generate integer sequence
rk = poisson(lam = LAM, size = M)


# 若有degree為0的點，則補一個stub給它

# In[5]:


tk = [1 if i==0 else i for i in tk]
rk = [1 if i==0 else i for i in rk]


# 讓所有degree的和相同

# In[6]:


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

# In[7]:


if (sum(tk)%2 != 0): # 
    index = random.choice(list(range(0,N)))
    tk[index] = tk[index] + 1
    index = random.choice(list(range(0,M)))
    rk[index] = rk[index] + 1


# 經過這些處理之後，transmitter端與reciever端具有相同的stub數(degree)以外，也能確保整個bipartite graph的degree是偶數

# In[8]:


rk = sorted(rk, reverse=True) # sort 2m stubs in descending order
tk = sorted(tk, reverse=True)
degree_transmitter = sum(tk)
degree_reciever = sum(rk)


# In[9]:


# print(rk)
# print(tk)
# print(degree_transmitter)
# print(degree_reciever)


# ### Step 3
# 現在將此bipartite graph建立起來，且已經保證所有頂點的degree不為0

# In[10]:


transmitter_node_degree_dict = {}
for nodes in range(0,N): # The nodes are named after 0~N-1
    transmitter_node_degree_dict[nodes] = tk[nodes]


# In[11]:


# transmitter_node_degree_dict # degree sequence


# In[12]:


reciever_node_degree_dict = {}
for nodes in range(0,M): # The nodes are named after '0'~'N-1'
    reciever_node_degree_dict[str(nodes)] = rk[nodes]


# In[13]:


model = nx.Graph()
model.add_nodes_from(transmitter_node_degree_dict.keys(), bipartite=0)
model.add_nodes_from(reciever_node_degree_dict.keys(), bipartite=1)


# In[14]:


del tk, rk


# ### Step 4

# 建立transmitter和reciever的stub

# In[15]:


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


# In[16]:


transmitter_stubs = []
for nodes in transmitter_node_degree_dict.keys():
    for degree in range(0, transmitter_node_degree_dict[nodes]):
        transmitter_stubs.append(Stubs(nodes, 2, 1))


# In[17]:


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

# In[18]:


pivot_index = int(degree_transmitter / 2)
pivot = transmitter_stubs[pivot_index-1] # 第m/2個stub
transmitter_block1 = transmitter_stubs[0:pivot_index] # 屬於degree較大的點
transmitter_block2 = transmitter_stubs[pivot_index:] # 屬於degree較小的點


# 執行原則1

# In[19]:


transmitter_block1.extend(list(filter(lambda x: (x.get_node() == pivot.get_node()), transmitter_block2)))
transmitter_block2 = list(filter(lambda x: (x.get_node() != pivot.get_node()), transmitter_block2))


# 執行原則2與3

# In[20]:


stubs_degree_toolarge = list(filter(lambda x: (transmitter_node_degree_dict[x.get_node()] 
                                               == transmitter_node_degree_dict[pivot.get_node()]), 
                                    transmitter_block2)) 
nodes_too_large = np.unique([stubs.get_node() for stubs in stubs_degree_toolarge])
# 等等需要增加的stub數目x=要刪掉的stub數+執行原則1後所需補的數目
x = len(nodes_too_large) + len(transmitter_block1) - len(transmitter_block2)


# In[21]:


# 此stub所連的node的degree跟pivot一樣大
for stubs in transmitter_block2: # 自所有degree跟pivot一樣大的點刪掉一個stub
    this = stubs.get_node()
    if ((this in nodes_too_large) and 
        (transmitter_node_degree_dict[this] == transmitter_node_degree_dict[pivot.get_node()])):
        transmitter_node_degree_dict[this] -= 1 # update degree sequence
        transmitter_block2.remove(stubs) # 刪除的動作


# In[22]:


b = [] # 候選點
block_2_node = np.unique([stubs.get_node() for stubs in transmitter_block2])
for nodes in block_2_node:
    if ((transmitter_node_degree_dict[pivot.get_node()] - transmitter_node_degree_dict[nodes] > 1) and 
        (transmitter_node_degree_dict[pivot.get_node()] - transmitter_node_degree_dict[nodes] < 4)):
        # degree小2的可以補一個stub,degree小3的可以補兩個stub
        for i in range(0, transmitter_node_degree_dict[pivot.get_node()] - 
                       transmitter_node_degree_dict[nodes] - 1):
            b.append(Stubs(nodes, 2, 1))


# In[23]:


assert (x < len(b)), '實驗失敗 請重新生成模型' # 若2,3無法被完成，則實驗失敗


# In[24]:


while (x > 0 and len(b) > 0):
    i = random.choice(list(range(0, len(b))))
    # 自block 2隨機抽出一stub，且保證此stub所屬之點的degree之值為(pivot所屬之點的degree-2 or degree-3)
    this = b[i].get_node() 
    transmitter_node_degree_dict[this] += 1 # update degree sequence
    transmitter_block2.append(Stubs(this, 2, 1)) # 增加一個stub
    x = x - 1
    b.pop(i)


# In[25]:


for stubs in transmitter_block2:
    stubs.block = 2


# 對Reciever進行一樣的操作

# In[26]:


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
            
assert (x < len(b)), '實驗失敗 請重新生成模型' # 若2,3無法被完成，則實驗失敗

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


# In[27]:


del b, stubs_degree_toolarge, nodes_too_large


# 這時候transmitter的degree和reciever的degree不一定相同。調整的方法是：誰比較少就把誰補滿。假設今天是transmitter比較少：
# 1. block 1因為是大degree，所以亂補也沒關係
# 2. 設block 2的頂點中最大的degree是$d$，則只有補degree比$d$小的點可以補，可以補的數量是$d-$該點degree

# In[28]:


x = len(transmitter_block2) - len(reciever_block2) # 每個block要補多少


# In[29]:


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


# In[30]:


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

# In[31]:


degree_transmitter = sum(transmitter_node_degree_dict.values())
degree_reciever = sum(reciever_node_degree_dict.values())


# In[32]:


# print(degree_transmitter)
# print(degree_reciever)
# print(len(transmitter_block1))
# print(len(transmitter_block2))
# print(len(reciever_block1))
# print(len(reciever_block2))


# 指定type

# In[33]:


type1_count = int(len(transmitter_block1)*q) # type 1數量


# In[34]:


transmitter_block1_type1_stubs = random.sample(transmitter_block1, type1_count)
transmitter_block2_type1_stubs = random.sample(transmitter_block2, type1_count)
for stubs in transmitter_block1_type1_stubs:
    stubs.stubtype = 1
for stubs in transmitter_block2_type1_stubs:
    stubs.stubtype = 1
transmitter_type2_stubs = list(filter(lambda x: (x.get_stubtype() == 2), transmitter_block1+transmitter_block2))


# In[35]:


reciever_block1_type1_stubs = random.sample(reciever_block1, type1_count)
reciever_block2_type1_stubs = random.sample(reciever_block2, type1_count)
for stubs in reciever_block1_type1_stubs:
    stubs.stubtype = 1
for stubs in reciever_block2_type1_stubs:
    stubs.stubtype = 1
reciever_type2_stubs = list(filter(lambda x: (x.get_stubtype() == 2), reciever_block1+reciever_block2))


# In[36]:


# print(len(transmitter_block1_type1_stubs))
# print(len(transmitter_block2_type1_stubs))
# print(len(transmitter_type2_stubs))
# print(len(reciever_block1_type1_stubs))
# print(len(reciever_block2_type1_stubs))
# print(len(reciever_type2_stubs))


# In[37]:


transmitter_stubs = transmitter_block1_type1_stubs + transmitter_block2_type1_stubs + transmitter_type2_stubs


# ### Step 5

# In[38]:


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


# In[39]:


del transmitter_stubs, reciever_stubs, transmitter_block1, transmitter_block2, reciever_block1, reciever_block2
del transmitter_block1_type1_stubs, transmitter_block2_type1_stubs, transmitter_type2_stubs, 
del transmitter_node_degree_dict, reciever_node_degree_dict
del reciever_block1_type1_stubs, reciever_block2_type1_stubs, reciever_type2_stubs


# ### Step 6
# 1. Degree Correlation

# In[40]:


corr = nx.degree_pearson_correlation_coefficient(model)


# 2. covariance

# In[41]:


xl=[]
yl=[]
deg_sum = 0
for (v, w) in model.edges():
    deg_sum = deg_sum + model.degree[v] * model.degree[w]
    xl.append(model.degree[v])
    yl.append(model.degree[w])

EX = np.mean(xl)
EY = np.mean(yl)
EXY = deg_sum / len(model.edges)


# In[42]:


cov = EXY - EX * EY


# 3. Giant component

# In[43]:


gc = len(max(nx.connected_components(model), key=len)) / (N+M)


# ## SIC演算法

# SIC演算法的步驟：
# 1. 找出reciever中所有degree為1的點
# 2. 若$w$是這樣的點,假設他唯一的那條邊叫做$(v,w)$,$v$為transmitter上的點,那麼就把所有過$v$之邊刪除
# 3. 重複1,2直到無法再刪除,也就是reciever端不再有degree為1的點

# In[44]:


transmitter_nodes = [x for x in list(model.nodes()) if isinstance(x, int)]
reciever_nodes = [x for x in list(model.nodes()) if isinstance(x, str)]


# In[45]:


m = len(model.edges()) # 邊的總數
deg1_r = len([x for x in reciever_nodes if (model.degree(x)==1)]) # 當下reciver中degree為1的點的總數


# In[46]:


while(deg1_r != 0):
    for node in reciever_nodes:
        if (model.degree(node) == 1): # 找出reciever中degree為1的點
            w, v = list(model.edges([node]))[0] # 令這條唯一的邊叫做(w,v),v是transmitter上的點,w是reciever上的點
            remove = list(model.edges(v))
            model.remove_edges_from(remove) # 刪除這些邊
            
    deg1_r = len([x for x in reciever_nodes if (model.degree(x)==1)]) # update


# In[47]:


sic = (m - len(model.edges())) / m # packet解成功的比例


# In[48]:


throughput = len([x for x in transmitter_nodes if (model.degree(x)==0)]) / len(transmitter_nodes)


# ### 結果寫入檔案

# In[49]:


with open('./results/q=' + str(q) + '.csv',  'a', newline = '') as csvFile:
    csvWriter = csv.writer(csvFile, delimiter = ',')
    csvWriter.writerow([corr, cov, gc, sic, throughput])

