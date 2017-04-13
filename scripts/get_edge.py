import xml.etree.ElementTree as et
import numbers

suffix = ""
def get_map(file_path):
	"""
	@rtype: list of edge map coords
	"""
	global suffix
	result = {}
	tree = et.parse(file_path)
	root = tree.getroot()
	for i in root.attrib:
		if ("schemaLocation" in i):
			suffix = "{" + root.attrib[i].split()[0] + "}"
	for tag in root.iter(suffix + "CXRreadingSession"):
		#adding SOP_UID to dict
		for child1 in tag.iter(suffix + "imageSOP_UID"):
			uid = child1.text
			
			if (uid in result):
				key_list = [x for x in result[uid].keys() if isinstance(x, numbers.Number)]
				if len(key_list) != 0:
					count = max(key_list) + 1
				else:
					count = 0
			else:
				count = 0
			if uid not in result:
				result[uid] = {}
			result[uid]["score"] = []
			for child2 in tag.iter(suffix + "edgeMap"):
				a = child2.find(suffix + 'xCoord').text
				b = child2.find(suffix + 'yCoord').text
				# 3 layer list so each index/image can have multiple sets of edge map
				if count in result[uid]:
					result[uid][count] += [[a,b]]
				else:
					result[uid][count] = [[a,b]]
		#adding malignancy score
			for score in tag.iter(suffix + "malignancy"):
				result[uid]["score"] += [score.text]
	if result == {}:
		for tag in root.iter(suffix + "readingSession"):
			#adding SOP_UID to dict
			for child1 in tag.iter(suffix + "imageSOP_UID"):
				uid = child1.text
				if (uid in result):
					key_list = [x for x in result[uid].keys() if isinstance(x, numbers.Number)]
					if len(key_list) != 0:
						count = max(key_list) + 1
					else:
						count = 0
				else:
					count = 0
				if uid not in result:
					result[uid] = {}
				for child2 in tag.iter(suffix + "edgeMap"):
					a = child2.find(suffix + 'xCoord').text
					b = child2.find(suffix + 'yCoord').text
					# 3 layer list so each index/image can have multiple sets of edge map
					if count in result[uid]:
						result[uid][count] += [[a,b]]
					else:
						result[uid][count] = [[a,b]]
				#adding malignancy score
				result[uid]["score"] = []
				for score in tag.iter(suffix + "malignancy"):
					result[uid]["score"] += [score.text]
	return result

def get_UID(file_path):
	"""
	#@rtype: list of imageSOP_UID corresponding to the image
	"""
	result = []
	tree = et.parse(file_path)
	root = tree.getroot()
	for i in root.attrib:
		if ("schemaLocation" in i):
			suffix = "{" + root.attrib[i].split()[0] + "}"
			for child in root.iter(suffix + "imageSOP_UID"):
				a = child.text
				result += [a]

	return result

#TODO functions: get_score
def get_score(file_path):
	"""
	#@rtype: list of imageSOP_UID corresponding to the image
	"""
	result = []
	tree = et.parse(file_path)
	root = tree.getroot()
	for i in root.attrib:
		if ("schemaLocation" in i):
			suffix = "{" + root.attrib[i].split()[0] + "}"
	for child in root.iter(suffix + "imageSOP_UID"):
		a = child.text
		result += [a]

	return result

if __name__ == "__main__":
	path = "/home/charles/bcb430/DOI/LIDC-IDRI-0008/1.3.6.1.4.1.14519.5.2.1.6279.6001.269179928127536655796924146755/1.3.6.1.4.1.14519.5.2.1.6279.6001.270306633918898619338449637639/083.xml"
	a = get_map(path)
	b = get_UID(path)
	for i in a:
		print(i)
	print(a)
