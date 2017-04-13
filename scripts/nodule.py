import os, sys, glob

import time



import math

import numpy as np



import cv2

from lxml import etree

import pydicom

import matplotlib.pyplot as plt



from multiprocessing import Pool

import itertools



# folder matching pattern

FOLDER_PAT = '/home/tim/knapsack/lung/dicom/LIDC-IDRI-%0.4d/'

# xml namespace

NS='{http://www.nih.gov}'







#### Parameters ###########



# edge finding

CON = False

C_MIN=100

C_MAX=200



# size adjusted crop

NRA = False

# default cropping radius

R = 24



class loc_entry():



  def __init__(self, xml):



    self.uid = None

    self.z = None

    self.xy = []

    self.inclusion = False

  

    self._parse_loc(xml)





  def _parse_loc(self, xml):

    

    self.uid = xml.findall(NS+'imageSOP_UID')[0].text        

    self.z = float(xml.findall(NS+'imageZposition')[0].text)



    # extract position (locus / edgeMap)

    locus = xml.findall(NS+'locus')

    em = xml.findall(NS+'edgeMap')



    include  = xml.findall(NS+'inclusion')

    if include: self.inclusion = bool(include[0].text)



    # only one of list shall not be empty

    for e in em+locus:

      x = int(e.findall(NS+'xCoord')[0].text)

      y = int(e.findall(NS+'yCoord')[0].text)

      self.xy.append((x,y))





  def __repr__(self):

    return str([self.inclusion, self.z, self.xy])







class nod_entry():



  def __init__(self, nod=False, xml=''):



    # whether entry is classified as nodule or not

    self.nod = nod

    # nodule id

    self.nid = None

    # position

    self.pos = []



    # characteristics (should be empty for nonnodule)

    self.charac = {}



    self._parse_nodule(xml)





  def _parse_nodule(self, xml):



    id = xml.findall(NS+'noduleID') + xml.findall(NS+'nonNoduleID')

    self.nid = id[0].text



    # non-Nodule

    if not self.nod: self.pos.append(loc_entry(xml))



    # nodule

    charac = xml.findall(NS+'characteristics')

    if charac:

      for char in charac[0]:

        self.charac[char.tag.strip(NS)] = int(char.text)



    for roi in xml.findall(NS+'roi'):

       self.pos.append(loc_entry(roi))





  def __repr__(self):

    

    return str([ self.nod, self.nid, self.pos, self.charac ])









class nodule():



  def __init__(self, ne):



    self.pid = None       # patient ID

    self.cent = (0,0,0)   # 3D center

    self.rad = 0          # nodule_radius

    self.mal = [-1]*4     # malignancy_score

    self.crop = None      # cropped_image

    self.nods = []        # list of node_entry



    self._centralise (ne)



  def _centralise (self, nod_entry):



    self.nods.append(nod_entry)



    z = [l.z for l in nod_entry.pos if l.inclusion]

    xys  = [l.xy for l in nod_entry.pos if l.inclusion]

     

    centers, radius = formalise(xys)

     

    i = np.argmax(radius)

     

    self.rad = radius[i]

    self.cent = np.array((centers[i][0], centers[i][1], z[i]))

    self.mal[0] = nod_entry.charac['malignancy']





  def __repr__ (self):



    return str([self.pid, self.cent, self.rad, self.mal])



      

# input: [[(x1,y1), (x2,y2), ... ], [ ] ]

def formalise(xys):



  xys = [np.array(xy) for xy in xys]



  cents = [np.mean(xy,0) for xy in xys]

  #rads = [np.mean( [dist(pt,cent) for pt in xy ]) for xy,cent in zip(xys,cents) ]

  #rads = [np.mean( dist(xy, np.tile(cent,(len(xy),1)) ),0) for xy, cent in zip(xys, cents) ]



  rads = [np.max( dist(xy, cent) ,0) for xy, cent in zip(xys, cents) ]



  return cents, rads



# either 2D or 3D

def dist(p1, p2):

  d =  len(p1.shape)-1

  return np.sqrt(np.sum( (p1-p2)**2,d))



# whether two nodules are the same / overlap

def same_nodule(n1,n2):

  return min(n1.rad, n2.rad) >= dist(n1.cent, n2.cent) 



# merge two nodules into one

def merge_nodule(n1, n2):



  n1.nods += n2.nods



  n1.mal = [m for m in n1.mal + n2.mal if m != -1]

  n1.mal += [-1]*(4-len(n1.mal))



  v1 = 4*math.pi*n1.rad/3

  v2 = 4*math.pi*n2.rad/3



  n1.cent = (n1.cent*v1 + n2.cent*v2) / (v1+v2)

  n1.rad = (n1.rad*v1 + n2.rad*v2) / (v1+v2)



  return n1





def extract_readings(fn):



  f = open(fn)



  rad = []



  tree = etree.parse(f)

  root = tree.getroot()



  for child in root.findall(NS+'readingSession'):



    pos = [nod_entry(True, ch) for ch in child.findall(NS+'unblindedReadNodule')]

    neg = [nod_entry(False, ch) for ch in child.findall(NS+'nonNodule')]



    rad.append( {'pos':pos, 'neg':neg} )

  

  return rad





def extract_nodule( reading ):



  # Taking only nodules >= 3mm

  rads = [[nodule(entry) for entry in rad['pos'] if len(entry.charac)>0] for rad in reading]



  nodules = []



  # merge nodules marked by different radiologist

  while len(rads) > 0:

    r1 = rads.pop(0)

    r2 = nodules

    

    merged = []



    while len(r1) > 0:

      n1 = r1.pop(0)

      same = False



      for n2 in r2:

        if same_nodule(n1, n2):

          merged.append( merge_nodule(n1,n2) )

          r2.remove(n2)

          same = True

          break

      

      if not same: merged.append(n1)



    nodules = merged+r2



  return nodules





def nodule_data(sample):

 

  print '[nodule] sample %d' % sample



  xmlfs = glob.glob((FOLDER_PAT % sample)+'/*/*/*.xml')[::-1]



  # in case of empty case

  if len(xmlfs) == 0:

    print 'SAMPLE %d is empty' % sample

    return []



  rs = [extract_readings(xmlf) for xmlf in xmlfs]

  ns = [extract_nodule(r) for r in rs]



  maxind = np.argmax([len(n) for n in ns])



  nodes = ns[maxind]

  dcm_folder = os.path.dirname(xmlfs[maxind])



  dcmfs = glob.glob(dcm_folder+'/*.dcm')



  # Getting z position of all slice in sample

  try:

    zs = [pydicom.read_file(dcmf)[(0x0020,0x0032)].value[2] for dcmf in dcmfs]

  except:

    print 'SAMPLE %d does not have z pos' % sample

    return []



  for node in  nodes:

    x,y,z = node.cent

    minind = np.argmin(np.abs(zs-z))



    # selecting slice of right z position

    ds = pydicom.read_file(dcmfs[minind])



    node.pid = sample

    node.slice = minind + 1

    node.actualZ = zs[minind]



    ori = ds.pixel_array



    # cutmask

    polys = itertools.chain(*[ [np.array(loc.xy) for loc in nod.pos] for nod in node.nods])

    cutmask = np.zeros_like(ori)

    for poly in polys: cv2.fillPoly(cutmask, [poly], 1)



    node.cutmask = cutmask



    cut = ori*cutmask



    # Determining crop radius

    if NRA: cr = int(np.ceil( node.rad*2+0.1))

    else: cr = R



    # Crop and resize image

    node.cutcrop = crop_resize(x,y,cr, cut)

    node.crop = crop_resize(x,y,cr, ori)

    



    # Extracting contours

    if CON and node.crop != None:

      node.canny = cv2.Canny(np.uint8(node.crop), C_MIN, C_MAX)

    else:

      node.canny = None



    



  return nodes



# crop image centered at (x,y) with side radius r

# resize cropped image to side radius R

def crop_resize(x, y, r, img):



  crop = img[y-r :y+r ,x-r:x+r]



  # in cases where the crop exist at boundary

  if crop.shape != (r*2, r*2):

    return None

  else:

    # Resizing crop back to standard size

    return cv2.resize(crop, dsize=(2*R,2*R))    







def post_process (nodes):



  print '[post_processing]'



  # putting nodule images into x and y vector

  x = [node.cutcrop for node in nodes if node.cutcrop != None]

  y = [node.mal for node in nodes if node.cutcrop != None]



  z = range(len(x))



  for i in range(len(x)):

    for d in range(3):

      x.append(np.rot90(x[i],d+1))

      y.append(y[i])

      z.append(i)



  return (np.array(x), np.array(y), np.array(z))



if __name__ == '__main__':



  show = False



  # extracting nodules from patient cases

  nodes = []

  # Full: 1012

  for i in range(1, 1012): nodes += nodule_data(i)



  x,y,z = post_process(nodes)



  np.savez('../output/data_cut_multimal_rot.npz', x=x, y=y, z=z)



  print x.shape

  print y.shape

  print z.shape



  if show:

    fig = plt.figure(0)

    plt.show(False)



  print 'NODES:'

  #for node in nodes:

  for p in x:

  #  print node, node.slice, node.actualZ



    if show:

      #fig.add_subplot(121)

      plt.imshow( p, cmap=plt.cm.gray)

      #fig.add_subplot(122)

      #plt.imshow( node.cutcrop, cmap=plt.cm.gray)

      plt.draw()

      time.sleep(0.1)

      #cv2.imwrite('../output/sample%d.png' % node.pid, node.ori)

      #plt.savefig('../output/sample%d.png' % node.pid)

      fig.clf()




