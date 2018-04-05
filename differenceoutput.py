from PIL import Image, ImageFilter, ImageChops, ImageOps
import numpy as np
import os
import glob
def avgOfCoords(arr):
    if(arr != [] and arr.ndim>1):
        return np.mean(arr[:,0]), np.mean(arr[:, 1])
    else:
        return [0,0]

def sumHessian(arr, center, r):
    sum = 0
    x = center[0]
    y = center[1]
    for i in range(x - r, x + r):
        for n in range(y - r, y + r):
            sum += arr[i, n]
    return sum


def differenceOfGaussians(im, r):
    [im1, im2] = [im.filter(ImageFilter.GaussianBlur(radius=r)), im.filter(ImageFilter.GaussianBlur(radius=2 * r))]
    newim = ImageChops.subtract(im2, im1)
    newim = ImageChops.invert(newim)
    return newim


def isCorner(a, b, c, coord, im, r):
    x = coord[0]
    y = coord[1]
    m = np.asarray([[a, b], [b, c]])
    bigr = np.linalg.det(m) - (.05 * (np.trace(m) ** 2))
    if (bigr > 10000):
        total = 0
        t = 40
        bo = [y + r, y - r, x + r, x - r]
        circle = [im[int(bo[0]), int(x)],
                  im[int(bo[1]), int(x)],
                  im[int(y), int(bo[2])],
                  im[int(y), int(bo[3])]]
        for i in range(len(circle)):
            if (im[int(y), int(x)] > circle[i] + t or im[int(y), int(x)] > circle[i] - t):
                total = total + 1
        if (total / len(circle) > .85):
            return True
        else:
            return False
    else:
        return False


def isKeyPoint(val, im):
    extrema = im.getextrema()
    if (extrema == None):
        return False
    elif (abs(extrema[0] - extrema[1]) <= 7):
        return False
    elif (val == extrema[0] or val == extrema[1]):
        return True
    else:
        return False


def extractKeypoints(im):
    im = im.convert('L')
    w, h = im.size
    im = differenceOfGaussians(im, 4)
    [dx, dy] = np.gradient(im)
    imarray = im.load()
    keys = [5, 5]
    corners = [0, 0]
    for i in range(6, h - 6, 1):
        for n in range(6, w - 6, 1):
            box = (n - 5, i - 5, n + 5, i + 5)
            if isKeyPoint(imarray[n, i], im.crop(box)) == True:
                keys = np.vstack((keys, [i, n]))
    for z in range(len(keys)):
        [i, n] = keys[z, :]
        dxx = dx ** 2
        dyy = dy ** 2
        dxy = dx * dy
        a = sumHessian(dxx, [i, n], 5)
        b = sumHessian(dxy, [i, n], 5)
        c = sumHessian(dyy, [i, n], 5)
        if isCorner(a, b, c, [i, n], imarray, 4):
            corners = np.vstack((corners, [i, n]))
    return corners


def cluster(arr, r):
    newarr=np.asarray([0,0])
    while len(arr)>1:
        center = arr[0, :]
        box = np.asarray([center])
        outbox=[0,0]
        for i in range(1,len(arr)):
            te = arr[i, :]
            d = np.sum(np.square(te-center))
            if(d<=r**2):
                box = np.vstack((box, te))
            else:
                outbox = np.vstack((outbox, [te]))
        outbox = np.delete(outbox, 0, axis=0)
        arr=outbox
        newarr = np.vstack((newarr, avgOfCoords(box)))
    return newarr

def compareKeypoints(path1, path2):
    size = 300, 200
    im1 = ImageOps.fit(Image.open(path1).convert('L'), size, Image.ANTIALIAS)
    im2 = ImageOps.fit(Image.open(path2).convert('L'), size, Image.ANTIALIAS)
    g = np.asarray(extractKeypoints(im1))
    e = np.asarray(extractKeypoints(im2))
    g = cluster(g,10)
    e = cluster(e,10)
    z = np.asarray(avgOfCoords(e))
    q = np.asarray(avgOfCoords(g))
    #return np.linalg.norm(z-q)
def singleKeypoint(path1):
        size = 300, 200
        im1 = ImageOps.fit(Image.open(path1).convert('L'), size, Image.ANTIALIAS)
        g = np.asarray(extractKeypoints(im1))
        g = cluster(g, 10)
        return g


#path1 = "imgs/"+os.listdir('imgs')[1]
#print("image every other one is being compared to:",path1)
diffs=[]
for filename in os.listdir('imgs')[1:]:
    path2='imgs/'+filename
    np.save(path2, singleKeypoint(path2))
print(diffs)
#np.savetxt("diffs.csv", diffs, delimiter=",")

