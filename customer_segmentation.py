
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df=pd.read_csv('customers.csv')
X=df[['age','income','spend_score','visits']]
sc=StandardScaler()
Xs=sc.fit_transform(X)
# elbow
inertias=[]
for k in range(2,8):
    km=KMeans(n_clusters=k, random_state=42)
    km.fit(Xs)
    inertias.append(km.inertia_)
plt.plot(range(2,8), inertias)
plt.savefig('elbow.png')
# final model
km=KMeans(n_clusters=4, random_state=42)
df['cluster']=km.fit_predict(Xs)
# PCA
pca=PCA(n_components=2)
X2=pca.fit_transform(Xs)
plt.figure()
plt.scatter(X2[:,0], X2[:,1], c=df['cluster'])
plt.savefig('clusters.png')
