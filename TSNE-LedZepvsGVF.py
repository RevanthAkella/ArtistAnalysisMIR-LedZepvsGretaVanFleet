
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv("LedZepMSvsGVF.csv",sep=';',encoding='cp1252')
df


# In[27]:


x = df.drop(["Unnamed: 0","track_href","artists","name"], axis = 1) #Drop nonvalue columns to create dataset
x


# In[33]:


x.danceability.mean()


# In[34]:


x.energy.mean()


# In[35]:


x.instrumentalness.mean()


# In[36]:


x.liveness.mean()


# In[37]:


x.speechiness.mean()


# In[38]:


x.acousticness.mean()


# In[28]:


from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(x) #t-sne algorithm to clusterly dimensionally reduce songs to (x,y) points


# In[29]:


from matplotlib.pyplot import *
fig = figure(figsize=(20, 20))
ax = axes(frameon=False)
setp(ax, xticks=(), yticks=()) #remove axis labels
plt.title("T-SNE Scatter Plot", fontsize=20,loc='left')
scatter(X_embedded[:, 0], X_embedded[:, 1], s=45,   marker="o") #scatter plot of t-sne'd data

for row_id in range(0, len(df)):
        songpoint = df.name[row_id] + " | " + df.artists[row_id]
        xx = X_embedded[row_id, 0]
        yy = X_embedded[row_id, 1]
        plt.annotate(songpoint, (xx,yy), size=10, xytext=(-90,90), 
            textcoords='offset points', ha='center', va='bottom',
            bbox=dict(boxstyle='round4', fc='white', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5', 
                            color='red'))


# In[30]:


x.corr()


# In[31]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(x)
x_pca = pca.fit_transform(x) #PCA Dimension Reduction

from sklearn import svm #Novelty Detection Using SVM 
clf = svm.OneClassSVM(kernel="rbf", gamma=0.01) #Fit gamma to model size
clf.fit(x_pca)

xx, yy = np.meshgrid(np.linspace(-200, 800, 500), np.linspace(-200,800, 500)) #Gradient Shape to fit Classifier

Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) #Decide which songs are classified as "artist" songs
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(30, 20),)
plt.title("One Class SVM Model", fontsize= 30)
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.Blues_r) #Contour Plot Background Based on Classifier
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors='orange', alpha = 0.5) #Contour Plot Based on Classified Songs

#Plot song points back onto Contour Plot
for row_id in range(0, len(df)): 
        target_word = df.name[row_id]
        xxx = x_pca[row_id, 0]
        yyy = x_pca[row_id, 1]
        plt.annotate(target_word, (xxx,yyy), size=10, xytext=(-90,90), 
            textcoords='offset points', ha='center', va='bottom',
            bbox=dict(boxstyle='round4', fc='white', alpha=0.3),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=1', 
                            color='black'))


b1 = plt.scatter(x_pca[:, 0], x_pca[:, 1], c='white', s=100)
plt.axis('tight')
plt.xlim((-75, 100))
plt.ylim((-50, 50))

leg = plt.legend([a.collections[0], b1, ], #Make Legend
           ["Predicted Model", "Training Songs"],
           loc="upper left",prop={'size':30}, frameon=True)

plt.show()

