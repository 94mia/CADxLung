import os
import numpy as np
import random
from scipy.misc import imread, imsave

np.random.seed(10)
def filter(variance, home_path, path):
	ppid = 0
	flag = 0
	#home_path = "/home/zlstg1/cding0622/newfull_augdata"
	files = os.listdir(home_path)
	files.sort()
    #for dir, subdir, files in os.walk("/home/zlstg1/cding0622/var_noRot_data"):
        #for d in dir:
            #print(d)
            #for s in subdir:
#file_list
	print("\nfolder: ", path)
	i = 0
	for f in files:
		i += 1
		os.sys.stdout.write("\r{0:.2f}% {1}/{2} loading...".format(i*100/len(files), i, len(files)))
		os.sys.stdout.flush()           
	    #print(f)
		flag = 0
		if "png" in f:
			var = float(f.split("_")[2][:4])
			score = float(f.split("_")[1])
		if 0. <= score <= 3.0:
		    	m_class = "benign"
		else:
		    	m_class = "malignant"
		ptest = 0.30
		pvalid = 0.50
		r = np.random.random()
		if r < ptest:
			folder = "test"
		elif ptest < r < pvalid:
			folder = "valid"
		else:
			folder = "train"
		img = imread(os.path.join(home_path,f))

		if folder == "test":
			imsave(os.path.join("/home/zlstg1/cding0622/project/", path, folder, m_class, str(ppid) + "_" + f), img)
			ppid += 1
			if m_class is "benign":
				continue
			# balancing dataset
			pmalignant = 0.37
			r2 = np.random.uniform(0,11)/10.0
			if m_class is "malignant" and r2 <= pmalignant:
				flag = 1
			rot_list = [1,2,3]
			for rot in rot_list:
				if flag == 1:
					continue
				rot_img = np.rot90(img, rot)
				dest_path = os.path.join("/home/zlstg1/cding0622/project/", path, folder, m_class, str(ppid) + "_" + f)
				imsave(dest_path, rot_img)
				ppid += 1
		else:
			if var >= variance:
				print("omitted variance: ", var)
				continue
			imsave(os.path.join("/home/zlstg1/cding0622/project/", path, folder, m_class, str(ppid) + "_" + f), img)
			ppid += 1
			if m_class is "benign":
				continue
			# balancing dataset
			pmalignant = 0.37
			r2 = np.random.uniform(0,11)/10.0
			if m_class is "malignant" and r2 <= pmalignant:
				flag = 1
			rot_list = [1,2,3]
			for rot in rot_list:
				if flag == 1:
					continue
				rot_img = np.rot90(img, rot)
				dest_path = os.path.join("/home/zlstg1/cding0622/project/", path, folder, m_class, str(ppid) + "_" + f)
				imsave(dest_path, rot_img)
				ppid += 1
	print("\n")

def  sub_filter(variance, home_path, folder):
	print("folder: ", folder)
	prefix = "/home/zlstg1/cding0622/project/"
	path = os.path.join(prefix, folder)
	cmd = "cp -r %s/* %s/" % (home_path, path)
	os.system(cmd)
	count = 0

	f = os.listdir(os.path.join(path, "train/benign"))
	for img in f:
		if ".png" in img:
			var = float(img.split("_")[-1][:4])
			if var >= variance:
				cmd = "rm %s" % (os.path.join(path, "train/benign", img))
				os.system(cmd)
				count += 1
				#print("removed: ", img)	

	f = os.listdir(os.path.join(path, "train/malignant"))
	for img in f:
		if ".png" in img:
			var = float(img.split("_")[-1][:4])
			if var >= variance:
				cmd = "rm %s" % (os.path.join(path, "train/malignant", img))
				os.system(cmd)
				count += 1
				#print("removed: ", img)
	print("removed %d files" % count)	

if __name__ == "__main__":
    np.random.seed(521)
    print("Start filtering...\n")
    #filter(500., "/home/zlstg1/cding0622/varfull_pngdata", "ori/full_data")
    #filter(1.0, "/home/zlstg1/cding0622/varfull_pngdata", "ori/var1.0_data")
    #filter(2.0, "/home/zlstg1/cding0622/varfull_pngdata", "ori/var2.0_data")
    #filter(2.5, "/home/zlstg1/cding0622/varfull_pngdata", "ori/var2.5_data")
    #filter(3.0, "/home/zlstg1/cding0622/varfull_pngdata", "ori/var3.0_data")
    #filter(3.3, "/home/zlstg1/cding0622/varfull_pngdata", "ori/var3.3_data")
    #filter(3.8, "/home/zlstg1/cding0622/varfull_pngdata", "ori/var3.8_data")
    #sub_filter(1.0, "/home/zlstg1/cding0622/project/ori/full_data", "ori/var1.0_data")
    #sub_filter(2.0, "/home/zlstg1/cding0622/project/ori/full_data", "ori/var2.0_data")
    #sub_filter(2.5, "/home/zlstg1/cding0622/project/ori/full_data", "ori/var2.5_data")
    #sub_filter(3.0, "/home/zlstg1/cding0622/project/ori/full_data", "ori/var3.0_data")
    #sub_filter(3.3, "/home/zlstg1/cding0622/project/ori/full_data", "ori/var3.3_data")
    #sub_filter(3.8, "/home/zlstg1/cding0622/project/ori/full_data", "ori/var3.8_data")
    filter(500., "/home/zlstg1/cding0622/newfull_augdata", "aug/full_data")
    sub_filter(1.0, "/home/zlstg1/cding0622/project/aug/full_data", "aug/var1.0_data")
    sub_filter(2.0, "/home/zlstg1/cding0622/project/aug/full_data", "aug/var2.0_data")
    sub_filter(2.5, "/home/zlstg1/cding0622/project/aug/full_data", "aug/var2.5_data")
    sub_filter(3.0, "/home/zlstg1/cding0622/project/aug/full_data", "aug/var3.0_data")
    sub_filter(3.3, "/home/zlstg1/cding0622/project/aug/full_data", "aug/var3.3_data")
    #filter(500., "/home/zlstg1/cding0622/elim_data/", "eli_data")
