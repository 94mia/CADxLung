import numpy as np
from sklearn.decomposition import PCA
import os
from scipy.misc import imread
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


data = []
path = "/home/zlstg1/cding0622/newfull_augdata"
files = os.listdir(path)

for img in files:
	if ".png" in img:
		flat = imread(os.path.join(path, img)).flatten()
		data.append(flat)
k = 100

data = np.array(data)
print(data.shape)

pca = PCA(n_components=k)

pca.fit_transform(data)
plt.scatter(pca.components_[0], pca.components_[1])
plt.savefig("pca.png")

g_data = pca.components_.T

estimator = GaussianMixture(n_components=6, covariance_type='tied', max_iter=2000, verbose=1, random_state=0, tol=1e-5)

estimator.fit(g_data)
mean = estimator.means_
print("means: ", mean.shape)

p = estimator.predict_proba(g_data)
print(p)

count = 0
for i in range(len(p)):
	if all(p[i] < 5e-2):
		print("low probability data point")
		count += 1
print("number of underflow threshold data points: %d" % count)
