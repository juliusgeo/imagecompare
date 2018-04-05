from PIL import Image, ImageFilter, ImageChops, ImageOps, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
#all code created by Julius Park

#finds the sum of gradient values around the point to find the hessian
def sumHessian(arr, center, r):
    x = center[0]
    y = center[1]
    return np.sum(arr[x - r:x + r, y - r:y + r])

# finds the average of the input coordinate array
def avgOfCoords(arr):
    if (arr != [] and arr.ndim > 1):
        return np.mean(arr[:, 0]), np.mean(arr[:, 1])
    else:
        return [0, 0]

# returns an image that is the difference of gaussian blurs at r and 2r
def differenceOfGaussians(im, r):
    newim = ImageChops.invert(ImageChops.subtract(im.filter(ImageFilter.GaussianBlur(radius=2 * r)),
                                                  im.filter(ImageFilter.GaussianBlur(radius=r))))
    return newim


# adapted Harris/FAST corner detector
def isCorner(a, b, c, coord, im, r):
    x = coord[0]
    y = coord[1]
    m = np.asarray([[a, b], [b, c]])
    bigr = np.linalg.det(m) - (.05 * (np.trace(m) ** 2))
    if (bigr > 65):
        total = 0
        t = 40
        bo = [y + r, y - r, x + r, x - r]
        circle = [im[int(x), int(bo[0])],
                  im[int(x), int(bo[1])],
                  im[int(bo[2]), int(y)],
                  im[int(bo[3]), int(y)]]
        for i in range(len(circle)):
            if (im[int(x), int(y)] > circle[i] + t or im[int(x), int(y)] > circle[i] - t):
                total = total + 1
        if (total / len(circle) > .85):
            return True
        else:
            return False
    else:
        return False


# keypoint detector
def isKeyPoint(val, arr):
    extrema = [np.max(arr), np.min(arr)]
    if (extrema == None):
        return False
    elif (abs(extrema[0] - extrema[1]) <= 7):
        return False
    elif (val == extrema[0] or val == extrema[1]):
        return True
    else:
        return False


# finds significant keypoints
def extractKeypoints(im):
    w, h = im.size
    imarray = np.array(im.getdata()).astype(np.float32).reshape((im.size[0], im.size[1]))
    [dx, dy] = np.gradient(imarray)
    dxx = dx ** 2
    dyy = dy ** 2
    dxy = dx * dy
    imarray = np.asarray(im)
    # these are initalized as python lists because that saves computational time,
    # they're later converted back to numpy arrays
    keys = []
    corners = []
    for i in range(4, h - 4, 2):
        for n in range(4, w - 4, 2):
            if isKeyPoint(imarray[i, n], imarray[i - 4:i + 4, n - 4:n + 4]) == True:
                keys.append([i, n])
    keys = np.array(keys)
    for z in range(len(keys)):
        [i, n] = keys[z, :]
        a = sumHessian(dxx, [i, n], 5)
        b = sumHessian(dxy, [i, n], 5)
        c = sumHessian(dyy, [i, n], 5)
        if isCorner(a, b, c, [i, n], imarray, 4):
            corners.append([i, n])
    return np.array(corners)



# clusters keypoints by radius r
def cluster(arr, r):
    newarr = []
    while len(arr) > 1:
        center = arr[0][:]
        box = [center]
        outbox = []
        for i in range(1, len(arr)):
            te = arr[i][:]
            d = np.sum(np.square(te - center))
            if (d <= r ** 2):
                box.append(te)
            else:
                outbox.append(te)
        arr = outbox
        newarr.append(avgOfCoords(np.array(box)))
    return np.array(newarr)

def slice(im, size, keys, ssize):
    slices = []
    boxes = []
    boxsize = [size[0] // ssize, size[1] // ssize]
    keycenters = np.empty((0, 2), int)
    for i in range(0, ssize):
        for n in range(0, ssize):
            box = [(boxsize[0] * i), (boxsize[1] * n), (boxsize[0] * (i + 1) + 1), (boxsize[1] * (n + 1)) + 1]
            keys = np.array(keys)
            inbox = np.empty((0, 2), int)
            for z in range(len(keys)):
                if (box[0] <= keys[z, 1] <= box[2] and box[1] <= keys[z, 0] <= box[3]):
                    inbox = np.vstack((inbox, keys[z, :]))
            if (inbox.size == 0):
                keycenters = np.vstack((keycenters, [0, 0]))
            else:
                keycenters = np.vstack((keycenters, (avgOfCoords(inbox))))
            boxes.append(box)
            slices.append(im.crop(box=box))

    return slices, keycenters, boxes

def outputKeypointImage(path1, resize):
    size = resize
    im = ImageOps.fit(Image.open(path1).convert('L'), size, Image.ANTIALIAS)
    g = np.array(extractKeypoints(im))
    #g = cluster(g, 10)
    draw = ImageDraw.Draw(im)
    r=2
    for i in range(len(g)):
        x=g[i,1]
        y=g[i,0]
        draw.ellipse((x-r, y-r, x+r, y+r),fill='white')
    return im
import sys
if len(sys.argv) > 1:
    path=sys.argv[1]
    resize=[int(s) for s in sys.argv[2].split(',')]
else:
    print(" No arguments ")
im=outputKeypointImage(path, resize)
im.show()
