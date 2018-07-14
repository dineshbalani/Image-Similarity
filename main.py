import os
import sys
import re
import io
import numpy as np
import zipfile
from tifffile import TiffFile
from pprint import pprint
from PIL import Image #this is an image reading library
from numpy.linalg import svd
from pyspark import SparkContext
from scipy import linalg
import hashlib

def getFileName(FileName):
    # print(FileName)
    return FileName.split('/')[-1]

def getOrthoTif(zfBytes):
    #given a zipfile as bytes (i.e. from reading from a binary file),
    # return a np array of rgbx values for each pixel
    bytesio = io.BytesIO(zfBytes)
    zfiles = zipfile.ZipFile(bytesio, "r")
    #find tif:
    for fn in zfiles.namelist():
        if fn[-4:] == '.tif':#found it, turn into array:
            tif = TiffFile(io.BytesIO(zfiles.open(fn).read()))
            return tif.asarray()

def splitImages(kv):
    xAxisSplits = np.split(kv[1],5)
    count = 0
    result= []
    for xAxisSplit in xAxisSplits:
        yAxisSplits = np.split(xAxisSplit,5,axis=1)
        for yAxisSplit in yAxisSplits:
            result.append((str(kv[0])+'-'+str(count),yAxisSplit))
            count+=1
    return result

def eachPixelIntensity(kv):
    eachPixel = kv[1]
    intensity = int(((eachPixel[0] + eachPixel[1] + eachPixel[2]) / 3) * (eachPixel[3] / 100))
    return(kv[0],intensity)

def calculateIntensity(kv):
    newImage = []
    for eachRow in kv[1].tolist():
        newRow  = []
        for eachPixel in eachRow:
            # print(eachPixel)
            intensity = int(((eachPixel[0]+eachPixel[1]+eachPixel[2])/3)*(eachPixel[3]/100))
            newRow.append(intensity)
        newImage.append(newRow)
    # pprint(kv[0])
    # pprint(newImage)
    return((str(kv[0])),newImage)

def reduceResolution(kv,factor):
    xAxisSplits = np.split(np.asarray(kv[1]), factor)
    result = []
    for xAxisSplit in xAxisSplits:
        yAxisSplits = np.split(xAxisSplit, factor, axis=1)
        newRow=[]
        for yAxisSplit in yAxisSplits:
            mean = np.mean(np.asanyarray(yAxisSplit).flatten())
            newRow.append(mean)
        result.append(newRow)
    return ((str(kv[0])), result)

def rowDifference(kv):
    result=[]
    rdf = np.diff(kv[1])
    for eachRow in rdf:
        newRow=[]
        for i in range(len(eachRow)):
            newValue = 0
            if eachRow[i] < -1:
                newValue = -1
            elif eachRow[i] > 1:
                newValue = 1
            else:
                newValue = 0
            newRow.append(newValue)
        result.append(newRow)
    return ((str(kv[0])), result)

def colDifference(kv):
    result = []
    transposed = np.transpose(kv[1])
    cdf = np.diff(transposed)
    for eachRow in cdf:
        newRow = []
        for i in range(len(eachRow)):
            newValue = 0
            if eachRow[i] < -1:
                newValue = -1
            elif eachRow[i] > 1:
                newValue = 1
            else:
                newValue = 0
            newRow.append(newValue)
        result.append(newRow)
    unTransposed = np.transpose(result)
    return ((str(kv[0])), unTransposed.tolist())

def similarSizedChunks(kv):
    result=[]
    bits =[]
    avg = len(kv[1]) / float(128)
    chunks = []
    last = 0.0
    while last < len(kv[1]):
        chunks.append(kv[1][int(last):int(last + avg)])
        last += avg
    # print(len(chunks))
    for eachChunk in chunks:
        m = hashlib.md5()
        m.update(np.asarray(eachChunk))
        hex_str = m.digest()
        bytes_s = [int(x) for x in hex_str]
        bits.append(int(bin(bytes_s[0]).lstrip('0b').zfill(2)[-1]))
    # print(len(kv[1]))
    # set38 = kv[1][:3496]
    # set39 = kv[1][3496:]
    #
    # splitSet38 = np.array_split(np.array(set38), 92)
    # splitSet39 = np.array_split(np.array(set39), 36)
    # for eachSplit in splitSet38:
    #     m = hashlib.md5()
    #     m.update(np.asarray(eachSplit))
    #     hex_str = m.digest()
    #     bytes_s = [int(x) for x in hex_str]
    #     bits.append(int(bin(bytes_s[0]).lstrip('0b').zfill(2)[0]))
    # for eachSplit in splitSet39:
    #     m = hashlib.md5()
    #     m.update(np.asarray(eachSplit))
    #     hex_str = m.digest()
    #     bytes_s = [int(x) for x in hex_str]
    #     bits.append(int(bin(bytes_s[0]).lstrip('0b').zfill(2)[0]))
    # # result = [set38,set39]
    return((kv[0],bits))

def createSignatures(kv):
    m = hashlib.md5()
    m.update(np.asarray(kv[1]))
    hex_str = m.digest()
    bytes_s = [int(x) for x in hex_str]
    return (str(kv[0]),bytes_s)

def createBands(kv,bands):
    bandSize = bands
    #final outcome:- (bandId, (imageName,band)
    splitArray = np.split(np.asarray(kv[1]),bandSize)
    result=[]
    for i in range(len(splitArray)):
        result.append((i,(kv[0],splitArray[i].tolist())))
    return result

def hashBands(kv,buckets):
    result=[]
    #final outcome:-  ((band,hash),(imageName))
    for i in range(len(kv[1])):
        m = hashlib.md5()
        m.update(np.asarray(kv[1][i][1]))
        hex_str = m.digest()
        bytes_s = sum([int(x) for x in hex_str])%buckets
        result.append((str(kv[0])+str(bytes_s),kv[1][i][0]))
    return result

def reshapeTheImage(kv):
    pass

def returnEachPixel(kv):
    print(kv[0])
    return kv[1]
    result=[]
    for i in kv[1]:
        result.append((str(kv[0]),i))
    return result

def calculateSVD2(kv):
    # count = 0
    img_diffs = []
    names = []
    result = []
    rr = range(len(kv[1]))
    rr = [str(iii) for iii in rr]
    rr = sorted(rr)
    for n, x in enumerate(kv[1]):
        # count += 1
        names.append(kv[0] + "-"+ rr[n])
        img_diffs.append(x)
    # print('count', count)
    mu, std = np.mean(img_diffs, axis=0), np.std(img_diffs, axis=0)
    img_diffs_zs = (img_diffs - mu) / std
    img_diffs_zs = np.nan_to_num(img_diffs_zs)
    U, s, Vh = linalg.svd(img_diffs_zs, full_matrices=1)
    low_dim_p = 10
    img_diffs_zs_lowdim = U[:, 0:low_dim_p]
    for i, x in enumerate(img_diffs_zs_lowdim):
        result.append([names[i], x])
    return result

def calculateSVD(kv):
    with open('test1.txt', "w") as tt:
        print('reached here', file=tt)
    imageNames = []
    imageDiffs = []
    results = []
    with open('test2.txt', "w") as ttt:
        print('reached here', file=ttt)
    for i in kv:
        imageNames.append(i[0])
        imageDiffs.append(i[1])
    mu, std = np.mean(imageDiffs, axis=0), np.std(imageDiffs, axis=0)
    img_diffs_zs = (imageDiffs - mu) / std
    img_diffs_zs = np.nan_to_num(img_diffs_zs)
    U, s, Vh = linalg.svd(img_diffs_zs, full_matrices=1)
    low_dim_p = 10
    img_diffs_zs_lowdim = U[:, 0:low_dim_p]
    for i,x in enumerate(img_diffs_zs_lowdim):
        results.append((imageNames[i],x))
    return results
    # count = 0
    # img_diffs = []
    # names = []
    # result = []
    # for x in kv:
    #     # count += 1
    #     names.append(x[0])
    #     img_diffs.append(x[1])
    # # print('count', count)
    # mu, std = np.mean(img_diffs, axis=0), np.std(img_diffs, axis=0)
    # img_diffs_zs = (img_diffs - mu) / std
    # img_diffs_zs = np.nan_to_num(img_diffs_zs)
    # with open('test.txt', "w") as f:
    #     print(img_diffs_zs, file=f)
    # U, s, Vh = linalg.svd(img_diffs_zs, full_matrices=1)
    # low_dim_p = 10
    # img_diffs_zs_lowdim = U[:, 0:low_dim_p]
    # for i, x in enumerate(img_diffs_zs_lowdim):
    #     result.append([names[i], x])
    # return result


def calcEuclideanDistance(kv):
    results = []
    for i in kv[1]:
        # pprint('----------------------kk'+str(i))
        a = broadCastSVD.value[kv[0]]
        b = broadCastSVD.value[i]
        ed = np.linalg.norm(a-b)
        results.append((i,ed))
    return (kv[0],results)

def sortByValues(v):
    v = sorted(v,key=lambda x:x[1])
    return v


# sc=SparkContext("local","Balani")
sc=SparkContext()

# sc=SparkContext.getOrCreate()

dataDir =r'hdfs:/data/large_sample/'
noOfBuckets = 135
noOfBands = 4
noOfPartitions = 46

# dataDir =r'hdfs:/data/small_sample/'
# noOfBuckets = 50
# noOfBands = 8
# noOfPartitions = 5
# dataDir =r'C:\Users\SSDN-Dinesh\Desktop\SBU\BDA\Assignment2\a2_small_sample'

#gives key as file Path and value as binary of file
data = sc.binaryFiles(dataDir)
#gives key as file name and value as binary of file
# fileName = data.map(lambda x:(x[0].split('/')[-1],x[1]))
# outName = 'fileName'
# out1 = fileName.collect()
# broadCastFileNames = sc.broadcast(fileNames)

#gives key as file name and array of image
fullImgs = data.map(lambda x:(x[0].split('/')[-1],getOrthoTif(x[1])))
# out1  = fullImgs.collect()
# outName = 'ImageShapeRDD'

# # imgShape = fullImgs.map(lambda x:(x[0],x[1].shape))
# # out = imgShape.collect()
# # outName = 'imgShape'
#
splitImages = fullImgs.flatMap(splitImages)

# Showing output for step1
step1 = splitImages.filter(lambda x: x if(x[0] in ['3677454_2025195.zip-0','3677454_2025195.zip-1','3677454_2025195.zip-18','3677454_2025195.zip-19']) else None).map(lambda x:(x[0],x[1][0][0]))
out1 = step1.collect()
# outName = 'step1'

#calculating Intensity of images
intensityImages = splitImages.map(calculateIntensity).persist()
# out = intensityImages.take(1)

# intensityImages = splitImages.map(returnEachPixel)
# intensityImages = splitImages.flatMap(lambda x:[(x[0],i) for i in x[1]])
# out = intensityImages.collect()
# outName = 'intensityImages'
#
# with open(outName+'.txt', "w") as f:
#     print(out,file=f)

reducedResImages = intensityImages.map(lambda x:reduceResolution(x,50))
# out = reducedResImages.take(1)
# outName = 'reducedResImages'

rowDifferenceImgs = reducedResImages.map(rowDifference)
# out = rowDifferenceImgs.take(1)
# outName = 'rowDifferenceImgs'

colDifferenceImgs = reducedResImages.map(colDifference)
# out = colDifferenceImgs.take(1)
# outName = 'colDifferenceImgs'

longVectorImgs = rowDifferenceImgs.union(colDifferenceImgs).reduceByKey(lambda x,y:np.concatenate([np.asarray(x).flatten(),np.asarray(y).flatten()]).tolist())
# out = longVectorImgs.sortByKey(ascending=True).collect()
# outName = 'longVectorImgs'

step2 = longVectorImgs.filter(lambda x: x if(x[0] in ['3677454_2025195.zip-1','3677454_2025195.zip-18']) else None).map(lambda x:(x[0],np.asarray(x[1])))
out2 = step2.collect()
# outName = 'step2'

signatures = longVectorImgs.map(similarSizedChunks)
# out = signatures.collect()
# outName = 'signatures'

# signatures = longVectorImgs.map(createSignatures)
# out = signatures.collect()
# outName = 'signatures'

bandsRDD = signatures.flatMap(lambda x:createBands(x,noOfBands)).groupByKey().map(lambda x:(x[0],list(x[1])))
# out = bandsRDD.collect()
# outName = 'bandsRDD'

hashBandsRDD = bandsRDD.flatMap(lambda x:hashBands(x,noOfBuckets)).groupByKey().map(lambda x:(x[0],list(x[1])))
hashBandsRDD0 = hashBandsRDD.filter(lambda x: x if('3677454_2025195.zip-0' in x[1]) else None).map(lambda x:('3677454_2025195.zip-0',x[1])).reduceByKey(lambda x,y:x+y).map(lambda x:(x[0],list(set(x[1]))))
hashBandsRDD1 = hashBandsRDD.filter(lambda x: x if('3677454_2025195.zip-1' in x[1]) else None).map(lambda x:('3677454_2025195.zip-1',x[1])).reduceByKey(lambda x,y:x+y).map(lambda x:(x[0],list(set(x[1]))))
hashBandsRDD18 = hashBandsRDD.filter(lambda x: x if('3677454_2025195.zip-18' in x[1]) else None).map(lambda x:('3677454_2025195.zip-18',x[1])).reduceByKey(lambda x,y:x+y).map(lambda x:(x[0],list(set(x[1]))))
hashBandsRDD19 = hashBandsRDD.filter(lambda x: x if('3677454_2025195.zip-19' in x[1]) else None).map(lambda x:('3677454_2025195.zip-19',x[1])).reduceByKey(lambda x,y:x+y).map(lambda x:(x[0],list(set(x[1]))))
# hashBandsAll = hashBandsRDD0.union(hashBandsRDD1).union(hashBandsRDD18).union(hashBandsRDD19)
# out = hashBandsAll.collect()
# outName = 'hashBandsAll'

step3b = hashBandsRDD1.union(hashBandsRDD18)
out3b = step3b.collect()
# outName = 'step3'

with open('test0.txt', "w") as tttt:
    print('reached here', file=tttt)

# step3HashBandsRDD = hashBandsRDD1.union(hashBandsRDD18)
svdRDD = longVectorImgs.sortByKey(ascending=True).repartition(noOfPartitions).mapPartitions(calculateSVD)
# svdRDD = longVectorImgs.sortByKey(ascending=True).map(lambda x:(x[0].split('-')[0],x[1])).groupByKey().map(lambda x:(x[0],list(x[1]))).flatMap(calculateSVD2)

# .map(calculateSVD)

# indexedRow = sc.parallelize(longVectorImgs)
# mat = IndexedRowMatrix(longVectorImgs.map(lambda row: IndexedRow(row[0],row[1])))
# rowsRDD = mat.rows
# out = svdRDD.collect()
# outName = 'svdRDD'
#
broadCastSVD = sc.broadcast(dict(svdRDD.collect()))
euclideanDistanceRDD1 = hashBandsRDD1.map(calcEuclideanDistance)
euclideanDistanceRDD18 = hashBandsRDD18.map(calcEuclideanDistance)
euclideanDistanceRDD = euclideanDistanceRDD1.union(euclideanDistanceRDD18).mapValues(sortByValues)

out3c = euclideanDistanceRDD.collect()
outName = 'a2_Balani_output'


with open(outName+'.txt', "w") as f:
    print(out1,file=f)
    print(out2, file=f)
    print(out3b,file=f)
    print(out3c,file=f)
    # print(bout2, file=f)
    # print(bout3b,file=f)
    # print(bout3c,file=f)

# print(out)

