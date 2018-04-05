import keypoint as k
import os
diffs=[]
dir=os.listdir('testimages')
for i in range(3,len(dir)):
	path1="testimages/"+dir[i]
	path2='testimages/'+dir[i+1]
	print("Path1=" + path1)
	print("Path2=" + path2)
	k.outputImage(path1, path2)
	print("---------------------------------------------\n\n")
