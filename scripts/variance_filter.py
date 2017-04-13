import numpy as np
from scipy.misc import imsave
from scipy.misc import imread
import os

def v_label(in_path, out_path):
	prefix = "/home/zlstg1/cding0622/"
	fpath = os.path.join(prefix, out_path)
	files = os.listdir(in_path)
	files.sort()
	i = 1 
	length = len(files)
	ppid = 1
	for img in files:
		if ".png" in img:
			var = float(img.split("_")[2][:4])
			vlabel = int(np.round(var))
			if vlabel >= 1:
				vlabel = str(1)
			else:
				vlabel = str(0)
				
			pvalid = 0.3
			r = np.random.random()
			if r < pvalid:
				folder = "valid"
			else:
				folder = "train"
			f = os.path.join(in_path, img)
			cmd = "cp {0} {1}".format(f, os.path.join(fpath, folder, vlabel))
			os.system(cmd)
			if int(vlabel) >= 1:
				tmp = imread(f)
				rot_list = [1,2,3]
				for rot in rot_list:
					rot_img = np.rot90(tmp, rot)
					new_i = "rot" + str(rot) + "_" + img[:-4] +".png"
					imsave(os.path.join(fpath, folder, vlabel, new_i), rot_img)
			
			os.sys.stdout.write("\r{0:.2f}% - {1}/{2} Loading".format(i*100/length, i, length))
			os.sys.stdout.flush()
			i += 1
	print("\n")
if __name__ == "__main__":
	np.random.seed(49)
	v_label("/home/zlstg1/cding0622/varfull_pngdata", "project/manual/data")
