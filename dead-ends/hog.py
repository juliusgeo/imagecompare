from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
#this is just my implementation of the method outlined in http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf

# defines block class--loc is a 4 element array defining the boundaries of the block and theta is a two element array
# with the components of the gradient vector
class Block:
    def __init__(self, loc, theta):
        self.loc = loc
        self.theta = theta

def extractGradients(im):
    w, h = im.size
    #casts image array as float32 as opposed to the default of unsigned ints because of compatibility issues
    # with numpy.gradient
    imarray = np.array(im.getdata()).astype(np.float32).reshape((im.size[0],im.size[1]))
    [dx, dy] = np.gradient(imarray)
    blockarr=[]
    #sets up the number of blocks
    numblocks=[(h//16),(w//16)]
    boxsize = [16, 16]
    for i in range(0, numblocks[0]):
        for n in range(0, numblocks[0]):
            #defines the box location
            box = [(boxsize[0] * i), (boxsize[1] * n), (boxsize[0] * (i + 1)), (boxsize[1] * (n + 1))]
            #defines the vertical and horizontal components of the gradient vector by finding the mean of the
            #gradients at every point in a 16x16 square
            vertw=np.mean(dy[box[0]:box[2], box[1]:box[3]])
            horzw=np.mean(dx[box[0]:box[2], box[1]:box[3]])
            theta=[horzw, vertw]
            #appends the theta array and the box location as a Block object to the block array
            blockarr.append(Block(box,theta))
    return blockarr


#uses matplotlib to create a quiver plot with the image forming the background
path1="face.jpg"
im1 = Image.open(path1).convert('L')
blockarr=extractGradients(im1)
X, Y = np.meshgrid(np.arange(0, im1.size[0], 16), np.arange(0, im1.size[1], 16))
#retrieves the theta components from the block array
U = [blockarr[i].theta[0] for i in range(len(blockarr))]
V = [blockarr[i].theta[1] for i in range(len(blockarr))]
#plots the quiver plot
plt.figure()
Q = plt.quiver(X, Y, U, V, units='xy')
#sets x and y axis lims
plt.ylim([0, im1.size[1]])
plt.xlim([0, im1.size[0]])
#sets up image background
im = plt.imread(path1)
plot = plt.imshow(im, origin='lower')
plt.gca().invert_yaxis()
plt.show()


