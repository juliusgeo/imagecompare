import numpy as np
import keypoint as k
path1 = "imgs/"+os.listdir('imgs')[1]
print("image every other one is being compared to:",path1)
for filename in os.listdir('imgs')[1:]:
    path2='imgs/'+filename
    emptyWeight=0
    sim=k.compareImages(path1, path2, emptyWeight)

