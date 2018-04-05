from PIL import Image, ImageFilter, ImageChops, ImageOps, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
# finds the Dimensional Invariant Similarity Measure https://arxiv.org/ftp/arxiv/papers/1409/1409.0923.pdf
def bdistance(l1, l2):
    b = []
    for i in range(len(l1)):
        c = (1 + min([l1[i], l2[i]])) / (1 + max([l1[i], l2[i]]))
        b.append(c)
    return np.mean(b)


def shift_l(l, shift, empty=0):
    src_index = max(-shift, 0)
    dst_index = max(shift, 0)
    length = max(len(l) - abs(shift), 0)
    new = [empty] * len(l)
    new[dst_index:dst_index + length] = l[src_index:src_index + length]
    return new


# finds the average of the input coordinate array
def avgOfCoords(arr):
    if (arr != [] and arr.ndim > 1):
        return np.mean(arr[:, 0]), np.mean(arr[:, 1])
    else:
        return [0, 0]


# finds the sum of gradient values around the point to find the hessian
def sumHessian(arr, center, r):
    x = center[0]
    y = center[1]
    return np.sum(arr[x - r:x + r, y - r:y + r])


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
    #kernel=np.array([0,0,1,0,0,0,1,2,1,0,1,2,-16,2,1,0,1,2,1,0,0,0,1,0,0])
    #print(kernel)
    #im=im.filter(ImageFilter.GaussianBlur(radius=1))
    #im = ImageOps.invert(im.filter(ImageFilter.Kernel((5,5),kernel)))
    im=differenceOfGaussians(im, 2)
    #im.show()
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


# shifts the coordinates of the two coordinate arrays so that they overlap
def shiftCoords(e, g):
    z = np.asarray(avgOfCoords(e))
    q = np.asarray(avgOfCoords(g))
    [y, x] = (z - q)
    e[:, 1] = e[:, 1] - x / 2
    e[:, 0] = e[:, 0] + y / 2
    return e, g


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


# defines block class
class Block:
    def __init__(self, loc, sim, slice1, slice2):
        self.loc = loc
        self.sim = sim
        self.slice1 = slice1
        self.slice2 = slice1


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


def compareImages(path1, path2, resize, halfweight, emptyweight):
    size = resize
    print("Finding keypoints...")
    im1 = ImageOps.fit(Image.open(path1).convert('L'), size, Image.ANTIALIAS)
    im2 = ImageOps.fit(Image.open(path2).convert('L'), size, Image.ANTIALIAS)
    g = np.array(extractKeypoints(im1))
    e = np.array(extractKeypoints(im2))
    #g = cluster(g, 10)
    #e = cluster(e, 10)
    # g = np.load('imgs/IMG_0838.npy')
    # e = np.load('imgs/IMG_0840.npy')
    z = np.asarray(avgOfCoords(e))
    q = np.asarray(avgOfCoords(g))
    [y, x] = (z - q)
    e[:, 1] = e[:, 1] - x / 2
    e[:, 0] = e[:, 0] + y / 2
    print("Found keypoints")
    print("Computing differences...")
    ssize = 20
    slices1, keycenters1, boxes1 = slice(im1, size, g, ssize)
    slices2, keycenters2, boxes2 = slice(im2, size, e, ssize)
    boxsize = [size[0] // ssize, size[1] // ssize]
    diag = np.linalg.norm([boxsize[0], boxsize[1]])
    diff = []
    outliers = [True] * ssize ** 2
    for i in range(len(boxes1)):
        t = [keycenters1[i][0] == 0, keycenters1[i][1] == 0, keycenters2[i][0] == 0, keycenters2[i][1] == 0]
        if t[0] and t[1] and t[2] == False and t[3] == False:
            diff.append(halfweight)
            outliers[i] = False
        elif t[2] and t[3] and t[0] == False and t[1] == False:
            diff.append(halfweight)
            outliers[i] = False
        elif t[2] and t[3] and t[0] and t[1]:
            diff.append(emptyweight)
        else:
            diff.append(np.linalg.norm(keycenters1[i] - keycenters2[i]) / diag)
    diff = np.asarray(diff)
    histodiffs = []
    lumdiffs = []
    condiffs = []
    blockarr = []
    print("Computing histogram, luminosity, and contrast differences...")
    for i in range(len(boxes1)):
        histo1 = slices1[i].histogram()
        histo2 = slices2[i].histogram()
        luminance1 = np.mean(np.asarray(slices1[i]))
        luminance2 = np.mean(np.asarray(slices2[i]))
        lumdiffs.append(abs(luminance1 - luminance2) / luminance1)
        stddev1 = np.std(histo1)
        stddev2 = np.std(histo2)
        condiffs.append(abs(stddev1 - stddev2) / stddev1)
        histodiffs.append(bdistance(histo1, histo2))
        blockSim = np.mean([1 - diff[i], np.mean(histodiffs), 1 - np.mean(lumdiffs), 1 - np.mean(condiffs)])
        blockarr.append(Block(boxes1[i], blockSim, slices1[i], slices2[i]))

    del slices1
    del slices2
    # print("similarity including blank sections:", 1-np.average(diff))
    # print("similarity excluding blank sections:", 1-np.average(diff[diff != emptyweight]))
    withoutoutliers = 1 - diff[np.array(outliers)]
    # print("similarity including blank sections and excluding outliers:", np.mean(withoutoutliers))
    # print("mean of histogram similarity", np.mean(histodiffs))
    avgofsims = np.mean(
        [1 - np.average(diff), 1 - np.average(diff[diff != emptyweight]), np.mean(withoutoutliers), np.mean(histodiffs),
         1 - np.mean(lumdiffs), 1 - np.mean(condiffs)])
    print("mean of all methods:", avgofsims)
    #plt.scatter(x=keycenters1[:, 1], y=keycenters1[:, 0], c='b', s=3)
    #plt.scatter(x=keycenters2[:, 1], y=keycenters2[:, 0], c='r', s=3)
    #plt.show()
    return blockarr

    # plot = plt.imshow(im1)

def outputImage(path1, path2):
    resize=[600,400]
    # cProfile.run('compareImages(path1, path2, 1, 0)')
    blocks = compareImages(path1, path2, resize, 1, 0)
    im1 = ImageOps.fit(Image.open(path1), resize, Image.NEAREST)
    im2 = ImageOps.fit(Image.open(path2), resize, Image.NEAREST)
    blank = Image.new('RGB', resize)
    draw1 = ImageDraw.Draw(blank, 'RGBA')
    simsarr = [blocks[i].sim for i in range(0, len(blocks))]
    locs = [[blocks[i].loc[k] * 1 for k in range(0, 4)] for i in range(0, len(blocks))]
    delta = max(simsarr) - min(simsarr)
    simsarr = [simsarr[i] - min(simsarr) for i in range(len(simsarr))]
    m = [simsarr[i] / delta for i in range(0, len(simsarr))]
    fnt = ImageFont.truetype('OpenSans-Bold.ttf', 10)
    for i in range(0, len(blocks)):
        blank.paste(blocks[i].slice1, box=blocks[i].loc)
        draw1.rectangle(locs[i], fill=(200, 0, 0, int(((1 - m[i]) + .1) * 255)), outline=None)
        # draw1.text((locs[i][0],locs[i][1]), str(round(m[i],2)), font=fnt, fill=(255, 255, 255, 100))
    blank.save(path2+'output.jpg')
