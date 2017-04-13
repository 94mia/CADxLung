import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.colors import LogNorm
from sklearn import mixture
from sklearn.mixture import GaussianMixture
import os
import time
from scipy.misc import imread
from scipy.misc import imshow
from scipy.misc import imsave
from numpy import *
import itertools
from sklearn.manifold import TSNE
from sklearn.mixture import BayesianGaussianMixture

colors = ['navy', 'darkorange']

def make_ellipses(gmm, ax):
    for n, color in enumerate(colors):
        if gmm.covariance_type == 'full':
            covariances = gmm.covariances_[n][:2, :2]
        elif gmm.covariance_type == 'tied':
            covariances = gmm.covariances_[:2, :2]
        elif gmm.covariance_type == 'diag':
            covariances = np.diag(gmm.covariances_[n][:2])
        elif gmm.covariance_type == 'spherical':
            covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
        v, w = np.linalg.eigh(covariances)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        print ("v: ", v)
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)

full_data = []
full_label = []
path = "/home/zlstg1/cding0622/varfull_pngdata"
files = os.listdir(path)
for img in files:
	if ".png" in img:
		score = float(img.split("_")[1])
		tmp_img = imread(os.path.join(path, img))
		flat = tmp_img.flatten()
		full_data += [flat]
	#full_label += [score]
	#if 0. <= score <= 3.0:
		#full_label += [0.]
	#elif 3.0 < score <= 5.0:
		#full_label += [1.]	
full_data = np.array(full_data)
#full_label = np.array(full_label)
print("full data shape:", full_data.shape)

k = 15
Xb_data  = [] 
xbpath = "/home/zlstg1/cding0622/project/gauss_data/%dfull_data/train/benign" % k
xbfiles = os.listdir(xbpath)
for img in xbfiles:
	if ".png" in img:
		score = float(img.split("_")[1])
		tmp_img = imread(os.path.join(xbpath, img))
		flat = tmp_img.flatten()
		Xb_data += [flat]
	#full_label += [score]
	#if 0. <= score <= 3.0:
		#full_label += [0.]
	#elif 3.0 < score <= 5.0:
		#full_label += [1.]	
Xb_data = np.array(Xb_data)
print("X(benign)_data shape:", Xb_data.shape)

Xm_data  = [] 
xmpath = "/home/zlstg1/cding0622/project/gauss_data/%dfull_data/train/malignant" % k
xmfiles = os.listdir(xmpath)
for img in xmfiles:
	if ".png" in img:
		score = float(img.split("_")[1])
		tmp_img = imread(os.path.join(xmpath, img))
		flat = tmp_img.flatten()
		Xm_data += [flat]
	#full_label += [score]
	#if 0. <= score <= 3.0:
		#full_label += [0.]
	#elif 3.0 < score <= 5.0:
		#full_label += [1.]	
Xm_data = np.array(Xm_data)
print("X(malignant)_data shape:", Xm_data.shape)

#classes_list = list(range(20))
#classes_list.remove(0)
classes_list = [k]
bic = []
cov_type = "diag"
for n_classes in classes_list:
	count = 0
	print("################")
	print("covarinace type: ", cov_type)
	print ("n_classes: ", n_classes)
	estimator =  GaussianMixture(n_components=n_classes, covariance_type=cov_type, max_iter=2000, verbose=1, random_state=0, tol=1e-5, reg_covar=1)

	#n_estimators = len(estimators)
	#plt.figure(figsize=(3 * n_estimators // 2, 6))
	#plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,left=.01, right=.99)

	#model = TSNE(n_components=n_classes, random_state=0)
	#np.set_printoptions(suppress=True)
	#k = model.fit_transform(X_data)

	estimator.fit(full_data)
	means = estimator.means_
	print("low: ", np.where(estimator.weights_ < 0.05)[0])
	low = np.where(estimator.weights_ < 0.05)[0]

	pb = estimator.predict(Xb_data)
	pm = estimator.predict(Xm_data)
        
	eli_list = []
	for i in range(len(pb)):
		if pb[i] in low:
			eli_list += [os.path.join(xbpath, xbfiles[i])]
			count += 1
	for j in range(len(pm)):
		if pm[j] in low:
			eli_list += [os.path.join(xmpath, xmfiles[j])]
			count += 1
	for f in eli_list:
		os.remove(f)
	print("number of unclustered data points: ", count)
	bic += [estimator.bic(full_data)]
	print("bic: ", estimator.bic(full_data))
	
	#data_accuracy = np.mean(estimator.predict(X_data) == X_label) * 100
	#print("data accuracy: ", data_accuracy, "%")

	for i in range(means.shape[0]):
		new_img = means[i].reshape((48,48))
		#imsave("/home/zlstg1/cding0622/gauss_result/%d_component_%d.png" % (n_classes, i), new_img)

	#plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))
	#plt.show()
plt.plot(classes_list, bic)
plt.savefig("bic_%s.png" % cov_type)	
