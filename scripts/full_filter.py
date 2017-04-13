import numpy as np
from scipy.misc import imsave
import os

def v_label(in_path, out_path):
	prefix = "/home/zlstg1/cding0622/"
	fpath = os.path.join(prefix, out_path)
	files = os.listdir(in_path)
	files.sort()
	i = 1 
	length = len(files)
	for img in files:
		if ".png" in img:
			score = float(img.split("_")[1])
			if 0. <= score <= 3.0:
				m_class = "benign"
			else:
				m_class = "malignant"

			f = os.path.join(in_path, img)
			cmd = "cp {0} {1}".format(f, os.path.join(fpath, m_class))
			os.system(cmd)
			
			os.sys.stdout.write("\r{}% - {}/{} Loading".format(i*100/length, i, length))
			os.sys.stdout.flush()
			i += 1
	print("\n")
if __name__ == "__main__":
	np.random.seed(10)
	v_label("/home/zlstg1/cding0622/varfull_pngdata", "project/all/data")
	v_label("/home/zlstg1/cding0622/newfull_augdata", "project/all/aug_data")
