import numpy
import fitsio
from skimage.feature import canny
from skimage.filters import gaussian, sobel, scharr
from cs import imshow, imgScale, plotCircle
import matplotlib.pyplot as plt
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.draw import circle_perimeter
from skimage.measure import label, regionprops, regionprops_table, find_contours

data = fitsio.read("P1234_FTO00013_20201030.fits")
data = data/numpy.max(data)

fiberDia = 120 # microns
imgScale = 2.78 # microns per pixel
targetDia = 140/2.78
print("targetDia", targetDia)

thresh = (data > numpy.percentile(data, 95))
imshow(data)



labels = label(thresh)
props = regionprops(labels, data)
# https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.regionprops
# this seems useful?
# https://stackoverflow.com/questions/31705355/how-to-detect-circlular-region-in-images-and-centre-it-with-python

# properties = ['area', 'eccentricity', 'perimeter', 'mean_intensity']
goodProps = []
for ii, region in enumerate(props):
    ed = region.equivalent_diameter
    if ed < 45 or ed > 60 or region.eccentricity > 0.3:
        continue
    cr, cc = region.weighted_centroid
    plotCircle(cc+0.5, cr+0.5, ed/2)
    plt.text(cc,cr,"%i"%ii)
    print("%i"%ii, cr, cc, region.equivalent_diameter, region.bbox_area, region.eccentricity, region.perimeter)
    goodProps.append(region)

plt.show()

assert len(goodProps) == 3

# props = regionprops_table(label_img)

# print(props)

# refImg = sobel(data)


import pdb; pdb.set_trace()