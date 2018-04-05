# import for PIL
from PIL import Image
import math
import numpy as np
# imports for compress
import lzma
from io import BytesIO

path1 = "imgs/IMG_0845.jpeg"
path2 = "imgs/IMG_0852.jpeg"
im1 = Image.open(path1).convert('L')
im2 = Image.open(path2).convert('L')


# im.show()
# im2.show()
# computes similarity using the cosine similarity algorithm
def cossimilarity(l1, l2):
    dot = sum([i * j for (i, j) in zip(l1, l2)])
    ml1 = math.sqrt(sum([i * j for (i, j) in zip(l1, l1)]))
    ml2 = math.sqrt(sum([i * j for (i, j) in zip(l2, l2)]))
    return dot / (ml1 * ml2)


def shift_l(l, shift, empty=0):
    src_index = max(-shift, 0)
    dst_index = max(shift, 0)
    length = max(len(l) - abs(shift), 0)
    new = [empty] * len(l)
    new[dst_index:dst_index + length] = l[src_index:src_index + length]
    return new


def histocompare(im1, im2):
    histo1 = im1.histogram()
    histo2 = im2.histogram()
    #print(histo1,histo2)
    max1 = max(range(len(histo1)), key=lambda i: histo1[i])
    max2 = max(range(len(histo2)), key=lambda i: histo2[i])
    shift = max1 - max2
    histo1 = shift_l(histo1, -shift)
    l1 = len(histo1)
    l2 = len(histo2)
    right = min(l1 - max1, l2 - max2)
    left = min(max1, max2)
    # print(left,right)
    histo1 = histo1[max1 - left:max1 + right]
    histo2 = histo2[max2 - left:max2 + right]
    if sum(histo1) > 0 and sum(histo2) > 0:
        histo1 = [i / sum(histo1) for i in histo1]
        histo2 = [i / sum(histo2) for i in histo2]
        return cossimilarity(histo1, histo2)
    else:
        return .5



def compresscompare(imone, imtwo):
    ratio = []
    ims = [imone, imtwo]
    for i in [0, 1]:
        output = BytesIO()
        ims[i].save(output, 'BMP')
        bytes_in = output.getvalue()
        c = lzma.LZMACompressor()
        bytes_out = c.compress(bytes_in)
        c.flush()
        lbi = len(bytes_in)
        lbo = len(bytes_out) + 67000
        ratio.append(lbi / lbo)
        print(lbi, lbo, ratio)
    return 1 - (ratio[0] - ratio[1]) / ratio[0]


print(100 * histocompare(im1, im2))
#print(compresscompare(im1, im2))
