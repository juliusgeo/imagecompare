import matplotlib.pyplot as plt

import numpy as np
from skimage import io
from skimage.segmentation import slic
from skimage.color import label2rgb
from skimage.measure import label

def bdistance(l1, l2):
    b = []
    for x, y in zip(l1, l2):
        c = (1 + np.min([x, y])) / (1 + np.max([x, y]))
        b.append(c)
    return np.mean(b)
#tuning parameters
n_segs=10
cmpctness=.01
img = io.imread("testimages/IMG1_small.jpg", as_grey=True)
segments_slic = slic(img, n_segments=n_segs, compactness=cmpctness, sigma=0)
segments_array=[[]]*len(np.unique(segments_slic))
for i in np.unique(segments_slic):
	mask=img[segments_slic == i]
	segments_array[i]=mask[mask !=0]
img1 = io.imread("testimages/IMG2_small.jpg", as_grey=True)
#segments_slic1 = slic(img1, n_segments=n_segs, compactness=cmpctness, sigma=0)
segments_array1=[[]]*len(np.unique(segments_slic))
for i in np.unique(segments_slic):
	mask=img1[segments_slic == i]
	segments_array1[i]=mask[mask !=0]


segment_sim=[]
min=min([len(segments_array1),len(segments_array)])
segments_array=segments_array[0:min]
segments_array1=segments_array1[0:min]
for i in range(0,min-1):
	segment=segments_array[i]
	segment1=segments_array1[i]
	histo1 = np.histogram(segment,bins=255,range=(0,1))[0]
	histo2 = np.histogram(segment1,bins=255,range=(0,1))[0]
	#print(histo1, histo2)
	luminance1 = np.mean(np.asarray(segment))
	luminance2 = np.mean(np.asarray(segment1))
	lumdiff=abs(luminance1 - luminance2) / luminance1
	stddev1 = np.std(segment)
	stddev2 = np.std(segment1)
	condiff=abs(stddev1 - stddev2) / stddev1
	histodiff=bdistance(histo1, histo2)
	#print(np.mean([lumdiff,condiff]))
	segment_sim.append(np.mean([lumdiff,condiff, 1-histodiff]))

segment_sim=np.asarray(segment_sim)
segment_sim[segment_sim<=0]=0
delta = np.max(segment_sim) -np.min(segment_sim)
segment_sim = [segment_sim[i] - np.min(segment_sim) for i in range(len(segment_sim))]
segment_sim = [segment_sim[i] / delta for i in range(0, len(segment_sim))]
tints=[(i,0,0) for i in segment_sim]
#print(tints)
print(segment_sim)


label_image = label(segments_slic)
image_label_overlay = label2rgb(label_image, image=img, colors=tints)
fig, ax = plt.subplots(figsize=(10, 6))
io.imshow(image_label_overlay)
io.show()
