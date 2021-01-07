import matplotlib.pyplot as plt
import fitsio
import numpy
from skimage.transform import rescale, rotate
from skimage.feature import canny, match_template
from skimage.filters import gaussian, sobel
import itertools
import time
from scipy import signal
import pandas as pd
import seaborn as sns
from multiprocessing import Pool
from skimage.measure import regionprops, label

#### from solid model
# center of metrology fiber to top of beta arm
met2top = 1.886  # mm
# metrology xy position in solid model
metXY = [14.314, 0]
# boss xy position in solid model
bossXY = [14.965, -0.376]
# apogee xy position in solid model
apXY = [14.965, 0.376]
# distance between fiber centers
fiber2fiber = 0.751  # mm
# distance from fiber to center of ferrule
fiber2ferrule = 0.5 * fiber2fiber / numpy.cos(numpy.radians(30))
# distance from center of ferrule to top of beta arm
ferrule2Top = met2top - fiber2ferrule
imgScale = 2.78  # microns per pixel
# radius for beta arm corners
betaArmRadius = 1.2  # mm
betaArmWidth = 3  # mm
# print("fiber2ferrule", fiber2ferrule)

data = fitsio.read("P1234_FTO00013_20201030.fits")
data = data/numpy.max(data)


def imshow(data):
    nrows, ncols = data.shape
    extent = [0, ncols, 0, nrows]
    plt.imshow(data, origin="lower", extent=extent)

############### ds9 measurements
# fiberRad = 27 # pixels
# f1 = numpy.array([826.75, 579.5])
# f2 = numpy.array([956.75, 339])
# f3 = numpy.array([1096.75, 579])

# roughXY = (f1 + f2 + f3)/3
# roughR = 0
# for f in [f1,f2,f3]:
#     roughR += numpy.linalg.norm(f-roughXY)
# roughR /= 3

# s1 = numpy.linalg.norm(f1-f2)
# s2 = numpy.linalg.norm(f2-f3)
# s3 = numpy.linalg.norm(f3-f1)
# # print(s1, s2, s3)
# meansl = numpy.mean([s1,s2,s3])
# r = 0.5*numpy.cos(numpy.radians(30))*meansl
# print("r", r)


# print(meansl)

######################

def plotCircle(x,y,r, color="red"):
    thetas = numpy.linspace(0, 2*numpy.pi, 200)
    xs = r * numpy.cos(thetas)
    ys = r * numpy.sin(thetas)
    xs = xs + x
    ys = ys + y
    plt.plot(xs,ys, color=color)


# maxData = numpy.max(data)
# thresh = 0.95*maxData
# data[data<thresh] = 0

# conv = numpy.zeros(data.shape)

# plotCircle(f1[0], f1[1], fiberRad)
# plotCircle(f2[0], f2[1], fiberRad)
# plotCircle(f3[0], f3[1], fiberRad)
# plotCircle(roughXY[0], roughXY[1], roughR)
# plotCircle(f2[0], f2[1], fiberRad)


def betaArmTemplate(
    ferruleRot=0,
    deltaFiberBundle=[0,0],
    deltaW=0,
    upsample=7,
    doFibers=True,
    imgRot=0,
    blurMag=1,
    deltaFiberRad = 0,
    deltaFiberCoreRad = 0,
    ):
    # ferruleRot: positive rotation moves fiber pattern CW about ferrule center
    # deltaFiberBundle: translate ferrule (after rotation) mm
    # deltaW vary width of beta arm in mm
    # doFibers: if True put the fibers in the template
    # upsample image to get curves outta pixels maks sure odd
    # image rotate whole template about central (upsampled) pixel (positive rotation points arm right CW)
    # blurMag gaussian blur template in original pixel scale (not upsampled pix)
    # deltaFiberRad mm delta applied to radius of fiber centers wrt ferrule center
    # deltaFiberCoreRad microns, vary the radius of the fiber core itself

    # size = int(2*roughR + 2*fiberRad) + 30
    size = int(2 * met2top * 1000 / imgScale)

    if size % 2 == 0:
        size += 1  # make it odd so a pixel is centered
    # scale = 7 # to upsample image to get curves outta pixels maks sure odd
    temp = numpy.zeros((size * upsample, size * upsample))

    # r = int(fiberRad*upsample)
    midX = int(size * upsample / 2)
    midY = midX
    deltaX = int(deltaFiberBundle[0] * 1000 / imgScale * upsample)
    deltaY = int(deltaFiberBundle[1] * 1000/ imgScale * upsample)
    # raidal distance from ferrule center to fiber center
    radFerCen2Fib = (fiber2ferrule + deltaFiberRad) * 1000 / imgScale * upsample
    fiberRadius = int((0.5 * 120 + deltaFiberCoreRad) / imgScale * upsample)  # fiber diameter is 120 microns

    if doFibers:
        for dd in [0 + 180, 120 + 180, 240 + 180]:
            xCen = numpy.int(numpy.cos(numpy.radians(dd + ferruleRot)) * radFerCen2Fib + deltaX) + midX
            yCen = numpy.int(numpy.sin(numpy.radians(dd + ferruleRot)) * radFerCen2Fib + deltaY) + midY
            for x in range(xCen - fiberRadius, xCen + fiberRadius + 1):
                for y in range(yCen - fiberRadius, yCen + fiberRadius + 1):
                    if numpy.linalg.norm([x - xCen, y - yCen]) <= fiberRadius:
                        temp[x, y] = 1

    ### draw midpoint of template (origin)
    # for x in range(midX-15, midX+15+1):
    #     for y in range(midY-15, midY+15+1):
    #         if numpy.linalg.norm([x-midX, y-midY]) <= 15:
    #             temp[x,y] = 0.5



    # draw beta arm outline
    _betaArmWidth = (betaArmWidth + deltaW) * 1000 / imgScale * upsample
    # width must be odd to remain centered
    if _betaArmWidth % 2 == 0:
        _betaArmWidth += 1  # make it odd so a pixel is centered

    # this keeps the width centered on midX
    lside = midX - int(_betaArmWidth / 2)
    rside = midX + int(_betaArmWidth / 2) + 1

    temp[:, :lside] = 1
    temp[:, rside:] = 1

    topside = midY + int((ferrule2Top) * 1000 / imgScale * upsample)
    temp[topside:, :] = 1
    # import pdb; pdb.set_trace()

    # 1.2 mm radius on the beta head, make it!
    # +y distance from
    # curveRad = 1.2  # mm
    curveRadPx = int(betaArmRadius * 1000 / imgScale * upsample)
    yoff = topside - curveRadPx
    loff = lside + curveRadPx
    roff = rside - curveRadPx

    # left side
    columns = numpy.arange(lside,loff)
    rows = numpy.arange(yoff, topside)

    # all pixels in shoulder region
    larray = numpy.array(list(itertools.product(rows, columns)))

    for row, column in larray:
        dist = numpy.linalg.norm([row - yoff, column - loff])
        if dist > curveRadPx:
            temp[row, column] = 1

    # right side
    columns = numpy.arange(roff, rside)
    rows = numpy.arange(yoff, topside)

    # all pixels in shoulder region
    rarray = numpy.array(list(itertools.product(rows, columns)))

    for row, column in rarray:
        dist = numpy.linalg.norm([row - yoff, column - roff])
        if dist > curveRadPx:
            temp[row, column] = 1
    # import pdb; pdb.set_trace()

    # apply sobel filter before rotating, this elimites extra edges
    # that pop up if you rotate before filtering
    # give it a bit of a blur by scale so it's approx blurMag pixels after downsampling
    temp = gaussian(temp, upsample * blurMag)
    temp = sobel(temp)

    # rotate whole image
    if imgRot != 0:
        temp = rotate(temp, imgRot)
    else:
        temp = rotate(temp, 1)
        temp = rotate(temp, -1)

    # scale back down to expected image size
    temp = rescale(temp, 1 / upsample, anti_aliasing=True)
    return temp


rots = numpy.linspace(-3,3,100) # even to keep from perfect alignment at rot=0, which gives higher signal?
dBAWs = numpy.linspace(-0.1,.1,51)


def multiTemplate(x):
    i,j = x
    print("ij", i, j)
    return betaArmTemplate(imgRot=rots[i], deltaW=dBAWs[j], upsample=1, doFibers=False)


def generateOuterTemplates(doFiber=False):
    defaultImg = betaArmTemplate(upsample=1)
    templates = numpy.zeros((len(rots), len(dBAWs), defaultImg.shape[0], defaultImg.shape[1]))

    ijs = []
    for ii, imgRot in enumerate(rots):
        for jj, dBAW in enumerate(dBAWs):
            ijs.append([ii,jj])

    p = Pool(12)
    t1 = time.time()
    templateList = p.map(multiTemplate, ijs)
    print("template gen took %.2f mins"%((t1-time.time())/60.))
    p.close()

    for (i, j), temp in zip(ijs, templateList):
        templates[i,j,:,:] = temp
    numpy.save("outerTemplates", templates)


def coolImage():
    imgs = []
    for imgRot in numpy.linspace(-60,60,25):
        print("imgRot", imgRot)
        temp = betaArmTemplate(0, upsample=1, imgRot=imgRot)
        imgs.append(sobel(temp))
    meanImg = numpy.sum(imgs,axis=0)
    imshow(meanImg)
    plt.show()


def correlateWithTemplate(image, template, avgPix=5):
    """
    image: image we are measuring, sobel filtered [rows,columns]
    template: a candididate template, sobel filtered [rows,columns]
    avgPix: amount of +/- pixels in row/columns to average around to find
        subpixel location of maximum response

    returns:
    maxResponse: largest signal seen in the correlation
    argPixMaxResponse: [row int, column int], location of max response in input image,
        pixels, [0,0] corresponds to LL pixel (which is really (0.5,0.5) in pixel units)
    meanPixMaxResponse: [row float, column float], fractional location of max reponse in
        input image. (0,0) is taken to be LL corner of LL pixel (0.5,0.5) is
        center of LL pixel
    """
    corr = signal.fftconvolve(image, template[::-1,::-1], mode="same")
    maxResponse = numpy.max(corr)

    argRow, argCol = numpy.unravel_index(numpy.argmax(corr), corr.shape)

    # box about where to find fractional pixel location
    corrCut = corr[argRow-avgPix:argRow+1+avgPix,argCol-avgPix:argCol+1+avgPix]

    dPix = numpy.arange(-avgPix, avgPix+1)
    totalCounts = numpy.sum(corrCut)
    # marginal distribution along columns
    margCol = numpy.sum(corrCut, axis=0) / totalCounts
    meandCol = numpy.sum(margCol*dPix)
    # marginal distribution along rows
    margRow = numpy.sum(corrCut, axis=1) / totalCounts
    meandRow = numpy.sum(margRow*dPix)

    varRow = numpy.sum(margRow*(dPix-meandRow)**2)
    varCol = numpy.sum(margCol*(dPix-meandCol)**2)

    print("means", meandCol, meandRow)

    meanRow = argRow + 0.5 + meandRow
    meanCol = argCol + 0.5 + meandCol

    return maxResponse, [argRow, argCol], [meanRow, meanCol], [varRow, varCol]


def testFit():
    rDia = 117/2/imgScale

    refImg = betaArmTemplate(imgRot=1.07, deltaW=0.025, upsample=1, doFibers=True, deltaFiberRad=-fiber2ferrule)
    bestTemplate = betaArmTemplate(imgRot=1, deltaW=0.02, upsample=1, doFibers=False)


    maxResponse, (row,col), (mRow, mCol) = correlateWithTemplate(refImg, bestTemplate)


    plt.figure()
    imshow(refImg)
    plotCircle(col+0.5, row+0.5, r=rDia)
    plotCircle(mCol, mRow, r=rDia, color="green")

    cutImg1 = refImg[205:,100:-101]

    maxResponse, (row,col), (mRow, mCol) = correlateWithTemplate(cutImg1, bestTemplate)
    plt.figure()
    imshow(cutImg1)
    plotCircle(col+0.5, row+0.5, r=rDia)
    plotCircle(mCol, mRow, r=rDia, color="green")
    plt.show()


# templates = numpy.load("outerTemplates.npy")


def doOne(x):
    iRot = x[0]
    jBaw = x[1]
    refImg = x[2]

    tempImg = templates[iRot,jBaw,:,:]
    maxResponse, [argRow, argCol], [meanRow, meanCol], [varRow, varCol] = correlateWithTemplate(refImg, tempImg)

    return rots[iRot], dBAWs[jBaw], maxResponse, argRow, argCol, meanRow, meanCol, varRow, varCol


    # conv = signal.fftconvolve(refImg, tempImg)
    # row,col = numpy.unravel_index(numpy.argmax(conv), conv.shape)
    # rotList.append(imgRot)
    # dBAWList.append(dBAW)
    # maxCorrList.append(numpy.max(conv))
    # colList.append(col)
    # rowList.append(row)
    # return rots[iRot], dBAWs[jBaw], numpy.max(conv), row, col,

def plotResults(pdSeries, refImg):

    plt.figure()
    imshow(refImg)

    for pds in pdSeries:
        imgRot = numpy.radians(pds["imgRot"])
        rotMat = numpy.array([
            [numpy.cos(imgRot), numpy.sin(imgRot)],
            [-numpy.sin(imgRot), numpy.cos(imgRot)]
        ])

        dBW = pds["dBetaWidth"]
        meanRow = pds["meanRow"]
        meanCol = pds["meanCol"]
        # carefule with rows     / cols vs xy
        centerOffset = numpy.array([meanCol, meanRow])

        yHeight = 1.5*1000/imgScale

        # central axis of beta positioner
        x1,y1 = [0, ferrule2Top*1000/imgScale]
        x2,y2 = [0, -yHeight]

        # right edge of beta arm
        betaArmRad = (betaArmWidth + dBW)/2*1000/imgScale
        x3,y3 = [betaArmRad, ferrule2Top*1000/imgScale]
        x4,y4 = [betaArmRad, -yHeight]

        # left edge of beta arm
        x5,y5 = [-betaArmRad, ferrule2Top*1000/imgScale]
        x6,y6 = [-betaArmRad, -yHeight]

        # top edge of beta arm
        x7,y7 = x5,y5
        x8,y8 = x3,y3

        # rotate lines by imgRot
        midLine_plusy = rotMat.dot([x1,y1]) + centerOffset # origin of coord sys
        midLine_minusy = rotMat.dot([x2,y2]) + centerOffset
        rightLine_plusy = rotMat.dot([x3,y3]) + centerOffset
        rightLine_minusy = rotMat.dot([x4,y4]) + centerOffset
        leftLine_plusy = rotMat.dot([x5,y5]) + centerOffset
        leftLine_minusy = rotMat.dot([x6,y6]) + centerOffset
        topLine_plusy = rotMat.dot([x7,y7]) + centerOffset
        topLine_minusy = rotMat.dot([x8,y8]) + centerOffset

        plt.plot([midLine_plusy[0], midLine_minusy[0]], [midLine_plusy[1], midLine_minusy[1]], 'c', alpha=0.5, linewidth=0.5)
        plt.plot([rightLine_plusy[0], rightLine_minusy[0]], [rightLine_plusy[1], rightLine_minusy[1]], 'r', alpha=0.5, linewidth=0.5)
        plt.plot([leftLine_plusy[0], leftLine_minusy[0]], [leftLine_plusy[1], leftLine_minusy[1]], 'g', alpha=0.5, linewidth=0.5)
        plt.plot([topLine_plusy[0], topLine_minusy[0]], [topLine_plusy[1], topLine_minusy[1]], 'b', alpha=0.5, linewidth=0.5)


def plotMarginals(df):
    plt.figure()
    sns.lineplot(x="imgRot", y="maxCorr", hue="dBetaWidth", data=df)

    plt.figure()
    sns.lineplot(x="dBetaWidth", y="maxCorr", hue="imgRot", data=df)

    plt.figure()
    sns.scatterplot(x="imgRot", y="dBetaWidth", hue="maxCorr", data=df)

    plt.figure()
    sns.scatterplot(x="meanCol", y="meanRow", hue="maxCorr", data=df)


def findMaxResponse(df, dbawDist, rotDist):
    """dbawDist is absoute deviation from dbaw at max
       rotDist is absoulte deviation from imgRot at max

       grab solutions in the locality of the max response

       returns
       argMaxSol: pandas series for parameters at maxCorrelation
       cutDF: sliced input dataframe with results falling within
            beta arm distance and rot distance constraints
    """
    amax = df["maxCorr"].idxmax() # where is the correlation maximized?
    argMaxSol = df.iloc[amax]
    dbaw = argMaxSol["dBetaWidth"]
    rot = argMaxSol["imgRot"]

    # search around the argmax to average
    df = df[numpy.abs(df["dBetaWidth"] - dbaw) <= dbawDist]
    cutDF = df[numpy.abs(df["imgRot"] - rot) <= rotDist].reset_index()

    # create an argMaxSol analog by averaging over nearby values
    avgMaxSol = {}
    for key in ["imgRot", "dBetaWidth", "argCol", "argRow", "meanCol", "meanRow"]:

        marg = cutDF.groupby([key]).sum()["maxCorr"]
        keyVal = marg.index.to_numpy()
        corrVal = marg.to_numpy()
        corrValNorm = corrVal / numpy.sum(corrVal)

        # determine expected value and variance
        meanPar = numpy.sum(keyVal*corrValNorm)
        varPar = numpy.sum((keyVal-meanPar)**2*corrValNorm)
        print("par stats", key, meanPar, varPar)


        avgMaxSol[key] = meanPar

        # plt.figure()
        # plt.plot(keyVal, corrValNorm, 'ok-')
        # plt.title(key)
        # plt.show()


        # import pdb; pdb.set_trace()



    return argMaxSol, pd.Series(avgMaxSol), cutDF


if __name__ == "__main__":


    # generateOuterTemplates()



    # refImg25sor = rotate(sobel(data), 2.5)

    # refImgsor = sobel(data)
    refImg = sobel(data)
    # fitsio.write("refImg.fits", refImg) # angle measured in ds9 90.15, 89.5, 359.9,

    # fits measured
    # a = numpy.mean([90-90.15, 90-89.5, 90-(359.9-360+90)])
    # print("a", a)




    def genDataFrame():
        indList = []
        for ii, imgRot in enumerate(rots):
            for jj, dBAW in enumerate(dBAWs):
                indList.append([ii,jj,refImg])

        pool = Pool(12)
        t1 = time.time()
        out = numpy.array(pool.map(doOne, indList))
        print("getData took %2.f mins"%((t1-time.time())/60.))
        pool.close()
        maxCorr = pd.DataFrame(out, columns=["imgRot", "dBetaWidth", "maxCorr", "argRow", "argCol", "meanRow", "meanCol", "varRow", "varCol"])
        maxCorr.to_csv("maxCorr.csv")



    # genDataFrame()

    refImg25 = rotate(refImg, 2.5)
    # # fitsio.write("refImg25.fits", refImg25) # 87.688336, 87.288246, 357.56788
    # a2 = numpy.mean([90-87.688336, 90-87.288246, 90-(357.56788-360+90)])
    # print("a2", a2)

    def genDataFrame25():
        indList = []
        for ii, imgRot in enumerate(rots):
            for jj, dBAW in enumerate(dBAWs):
                indList.append([ii,jj,refImg25])

        pool = Pool(12)
        t1 = time.time()
        out = numpy.array(pool.map(doOne, indList))
        print("genData25 took %2.f mins"%((t1-time.time())/60.))
        pool.close()
        maxCorr = pd.DataFrame(out, columns=["imgRot", "dBetaWidth", "maxCorr", "argRow", "argCol", "meanRow", "meanCol", "varRow", "varCol"])
        maxCorr.to_csv("maxCorr25.csv")

    # genDataFrame25()

    # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()

    # genDataFrame()

    # maxCorr is unrotated image
    # maxCorr25.csv is rotated by 2.5 degrees

    maxCorr = pd.read_csv("maxCorr.csv", index_col=0)

    plotMarginals(maxCorr)

    plt.show()



    maxCorr25 = pd.read_csv("maxCorr25.csv", index_col=0)

    plotMarginals(maxCorr25)

    plt.show()

    dbwDist = 0.02
    rotDist = 0.5

    sol, avgSol, maxCorrCut = findMaxResponse(maxCorr, dbwDist, rotDist)

    plotMarginals(maxCorrCut)

    plt.show()

    sol25, avgSol25, maxCorrCut25 = findMaxResponse(maxCorr25, dbwDist, rotDist)

    plotMarginals(maxCorrCut)
    plt.show()


    # amax = maxCorr["maxCorr"].idxmax()
    # sol = maxCorr.iloc[amax]

    # amax = maxCorr25["maxCorr"].idxmax()
    # sol25 = maxCorr25.iloc[amax]

    # import pdb; pdb.set_trace()

    plotResults([sol, avgSol], refImg)

    plt.show()

    plotResults([sol25, avgSol25], refImg25)


    plt.show()
    import pdb; pdb.set_trace()


