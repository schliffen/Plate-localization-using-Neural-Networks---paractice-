#
# This code is for detecting proper region of the plate
# with the combined approach
#
# imports
import cv2
import imutils
import numpy as np
from imutils import paths
# import RDetectPlates as detplt
from imutils import perspective
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.decomposition import PCA
from imutils import perspective
from all_params import *
#from all_params import TRAIN_DATA_PATH as path
#from all_params import MODEL_CHECKPOINT_DIR as ls_path
imgs = sorted(list(paths.list_images(path)), reverse=True)
now = datetime.datetime.now()
import Regressor_01

#rnd = 10
# testing the detector:

for _ in range(len(imgs)):
    rnd = np.random.randint(0, len(imgs) - 1, 1)[0]
    gimg = cv2.imread(imgs[rnd])

    plt.imshow(gimg)
    plt.close()

    try:
        gimg = cv2.cvtColor(gimg, cv2.COLOR_BGR2GRAY)
    except:
        print('there is an error in making img gray')

    # Detecting approximate region with OCV
    retRegions = []  # this will be the return value
    gCoords = []  # this will be the return value
    retCoords = []
    poss_plates = []
    globCoord = []

    # Vertical Kernels
    vertKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))
    pKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #
    # Horizontal Kernels
    bKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    b2Kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    smallKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 3))
    HKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 4))  # the rectangle kernel
    superKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 3))  # 27,3 check 29 also
    #
    bigpics = []  # this will be the return value
    # then initialize the list of license plate regions
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5))

    # convert the image to grayscale, and apply the blackhat operation

    blackhat = cv2.morphologyEx(gimg, cv2.MORPH_BLACKHAT, rectKernel)

    # find regions in the image that are light
    light = cv2.morphologyEx(gimg, cv2.MORPH_CLOSE, rectKernel)
    light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY)[1]

    # compute the Scharr gradient representation of the blackhat image and scale the
    # resulting image into the range [0, 255]
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    # blur the gradient representation, apply a closing operation, and threshold the
    # image using Otsu's method
    gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    gradX = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # perform a series of erosions and dilations on the image
    gradX = cv2.erode(gradX, squareKernel, iterations=2)
    gradX = cv2.dilate(gradX, squareKernel, iterations=3)

    # take the bitwise 'and' between the 'light' regions of the image, then perform
    # another series of erosions and dilations
    thresh = cv2.bitwise_and(gradX, gradX, mask=light)
    thresh = cv2.erode(thresh, squareKernel, iterations=2)
    thresh = cv2.dilate(thresh, squareKernel, iterations=2)

    # find contours in the thresholded image
    _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # loop over the contours
    for c in cnts:
        # grab the bounding box associated with the contour and compute the area and
        # aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        aspectRatio = w / float(h)

        # compute the rotated bounding box of the region
        rect = cv2.minAreaRect(c)
        box = np.int0(cv2.boxPoints(rect))

        # ensure the aspect ratio, width, and height of the bounding box fall within
        # tolerable limits, then update the list of license plate regions
        if (aspectRatio > 2 and aspectRatio < 8) and h > 10 and w > 50 and h < 125 and w < 400:
            # if h > 10 and w > 50 and h < 250 and w < 750:
            bigpics.append(box)
            gCoords.append(np.array([x, y, w, h]))

    #
    # processing the resulted images
    #
    for bigpic in bigpics:
        platel = perspective.four_point_transform(gimg, bigpic)

        # 1. Classifier: setting a classifier to select figures with all kind of plates included
        #Using trained network



        # 2. Regressor: Applting a light regressor to extract the exact region of the plate

        # net is loaded and we want to use it
        filedir = "save_load/"
        # net.save_params(filename)
        regplt = Regressor_01.Regression_plt()
        #
        tr_input, tr_target, ts_input, ts_target, size_1, size_2 = regplt.Reg_Data()
        net = regplt.Model()
        ## first we should define net the same as training, then load the saved parameters
        net.load_params(filedir + "testnet20180319-081631.params", ctx=mx.cpu())

        print('model loaded')

        # Here goes to padding image and dimension reduction which is currently
        # incomplete

        # which image?
        # an integer between 0-453
        wi = 420
        pred = np.int0(net(ts_input[wi].reshape((1, -1)))[0].asnumpy())
        # Showing the results

        image = ts_input[wi].reshape([100, 200])

        # after process, padding should be removed or taget also should be padded

        # drawing rectangle
        plate = cv2.rectangle(image, (pred[0], pred[1]), \
                              (pred[2], pred[3]), (0, 200, 0), 3)

        # this is for cropping plate number
        # plt_points = np.array([[int(pred[0]), int(pred[1])],[int(pred[0]) + int(pred[2]), int(pred[1])],\
        #               [int(pred[0]), int(pred[1]) + int(pred[3])], [int(pred[0]) + int(pred[2]), int(pred[1]) + int(pred[3])]])
        # cvxh = cv2.convexHull(plt_points)
        # rect = cv2.minAreaRect(cvxh)
        # box = np.int0(cv2.boxPoints(rect))
        # platel = perspective.four_point_transform(image, box)

        plt.imshow(plate)

        # for j in range(inp_raw.shape[0]):
        #    plate = cv2.rectangle(image[j], (pred[0], pred[1]),(pred[3], pred[4]), (200,0,0), 0)
        #    plt.imshow(plate)


        # 3. Here goes to segmentation





