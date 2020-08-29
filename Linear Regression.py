#!/usr/bin/env python
# coding: utf-8

# In[24]:


import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
house_price = [245, 312, 279, 308, 199, 219, 405, 324, 319, 255]
size = [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700]
plt.scatter(size,house_price, color='black')
plt.ylabel('house price')
plt.xlabel('size of house')
plt.show()


# In[25]:


house_price2=  [219, 405, 324, 319, 255, 245, 312, 279, 308, 199,219, 405, 324, 319, 255 ]


# In[26]:


size2 = [ 1550, 2350, 2450, 1425, 1700, 1400, 1600, 1700, 1875, 1100, 334 , 1100 ,334, 1100]


# In[27]:


house_price2=  [219, 405, 324, 319, 255, 245, 312, 279, 308, 199,219, 405, 324, 319, 255 ]
size2 = [ 1550, 2350, 2450, 1425, 1700, 1400, 1600, 1700, 1875, 1100, 334 , 1100 ,334, 1100]
plt.scatter(size,house_price, color='black')
plt.ylabel('house price')
plt.xlabel('size of house')
plt.show()


# In[28]:


size3=np.array(size).reshape((-1, 1))


# In[29]:


regr = linear_model.LinearRegression()
regr.fit(size3, house_price)
print ('Coefficients: \n', regr.coef_)
print ('Coefficients: \n', regr.intercept_)


# In[30]:


#formula obtained for the trained model
def graph(formula, x_range):
   x = np.array(x_range)
   y = eval(formula)
   plt.plot(x, y)
#plotting the prediction line 
graph('regr.coef_*x + regr.intercept_' , range(1000, 2700))
#regr.score(size3, house_price)
plt.scatter(size3, house_price, color='black')
plt.ylabel('house price')
plt.xlabel('size of house')
plt.show()


# In[31]:


#k-mean clustring


# In[32]:


import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans 


# In[33]:


x = [1, 5, 1.5, 8,1, 9] 
y = [2, 8, 1.8, 8, 0.6, 11]
plt.scatter(x,y)
plt.show()


# In[34]:


x = np.array([[1,2],[5,8],[1.5,1.8],[8,8],[1,0.6],[9,11]])


# In[35]:


kmeans = KMeans(n_clusters=2)
kmeans.fit(x)


# In[47]:


centeroids= kmeans.cluster_centers_
labls = kmeans.labels_
print(centeroids)
print(labls)


# In[46]:


#CLUSTRING VISULALIZATION
colors = ['g','r','c.','g.']
for i in range(len(x)):
    print("coordinate:",x[i], "labels:", labls[i])
    plt.plot(x[i][0], x[i][i], colors[labls[i]], markersize = 10)

plt.scatter(centroids[:,0], centroids[:, 1], marker ="x", s=150, linewidths= 5, zorder =10)
plt.show()
    
    


# In[ ]:





# In[ ]:




