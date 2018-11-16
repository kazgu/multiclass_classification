#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[2]:


import matplotlib.pyplot as plt
def DrawGraph(scores):
    plt.rcParams['figure.figsize'] = (8.0, 4)
    d_map = {
    'svm': (scores[0], '#7199cf'),
    'randomForest': (scores[1], '#4fc4aa'),
    'XGBoost': (scores[2], '#e1a7a2')
    }
    # 整体图的标题
    fig = plt.figure('Bar chart & Pie chart')
    # 在整张图上加入一个子图，121的意思是在一个1行2列的子图中的第一张
    ax = fig.add_subplot(111)
    ax.set_title('compare')
     # 生成x轴每个元素的位置
    xticks = np.arange(3)
     # 定义柱状图每个柱的宽度
    bar_width = 0.5
    animals = d_map.keys()
    values = [x[0] for x in d_map.values()] 
    # 对应颜色
    colors = [x[1] for x in d_map.values()]
    # 画柱状图，横轴是动物标签的位置，纵轴是速度，定义柱的宽度，同时设置柱的边缘为透明
    bars = ax.bar(xticks, values, width=bar_width, edgecolor='none') 
    # 设置y轴的标题
    ax.set_ylabel('F1 Score')
    # x轴每个标签的具体位置，设置为每个柱的中央
    ax.set_xticks(xticks+bar_width/3) 
    # 设置每个标签的名字
    ax.set_xticklabels(animals)
    # 设置x轴的范围
    ax.set_xlim([bar_width/2-0.5, 3-bar_width/2]) 
    # 设置y轴的范围
    ax.set_ylim(0, +1.00) 
    # 给每个bar分配指定的颜色
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    plt.show()


# In[ ]:




