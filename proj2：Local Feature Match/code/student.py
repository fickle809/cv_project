import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
import cv2
import math

def ANMS (x , y, r, maximum):
    i = 0
    j = 0
    NewList = []
    while i < len(x):
        minimum = 1000000000000 
        FirstCoordinate, SecondCoordinate = x[i], y[i] # 获得一个harris角点的横纵坐标，称为基础点
        # 遍历除该角点外每一个得分（R值）更高的角点，称为比较点，找到离基础点最近的一个比较点
        # 记录下基础点的横纵坐标和与最近比较点之间的距离
        while j < len(x):
            CompareCoordinate1, CompareCoordinate2 = x[j], y[j] 
            if (FirstCoordinate != CompareCoordinate1 and SecondCoordinate != CompareCoordinate2) and r[i] < r[j]:
                distance = math.sqrt((CompareCoordinate1 - FirstCoordinate)**2 + (CompareCoordinate2 - SecondCoordinate)**2)
                if distance < minimum:
                    minimum = distance
            j = j + 1
        NewList.append([FirstCoordinate, SecondCoordinate, minimum])
        i = i + 1
        j = 0
    #根据距离大小对基础点进行排序，很显然，距离越小的点说明在该角点周围有更好的角点，所以可以适当舍弃。
    # 在舍弃一定数目的得分较小的角点后，就得到了非最大抑制后的harris角点坐标。
    NewList.sort(key = lambda t: t[2])
    NewList = NewList[len(NewList)-maximum:len(NewList)]
    return NewList


def get_interest_points(image, feature_width):
    '''
    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width:

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # TODO: Your implementation here!
    alpha = 0.04
    threshold = 10000

    XCorners = []
    YCorners = []
    RValues = []
    ImageRows = image.shape[0]
    ImageColumns = image.shape[1]
    # 使用sobel算子处理计算图像在x和y方向上的导数
    Xderivative = cv2.Sobel(image, cv2.CV_64F,1,0,ksize=5)
    Yderivative = cv2.Sobel(image, cv2.CV_64F,0,1,ksize=5)
    #计算Ixx, Iyy, Ixy
    Ixx = (Xderivative)*(Xderivative)
    Iyy = (Yderivative)*(Yderivative)
    Ixy = (Xderivative)*(Yderivative)
    #遍历图像以计算每个像素的角度得分
    for i in range(16, ImageRows - 16):
        for j in range(16, ImageColumns - 16):
            Ixx1 = Ixx[i-1:i+1, j-1:j+1]
            Iyy1 = Iyy[i-1:i+1, j-1:j+1]
            Ixy1 = Ixy[i-1:i+1, j-1:j+1]

            Ixxsum = Ixx1.sum()
            Iyysum = Iyy1.sum()
            Ixysum = Ixy1.sum()

            Determinant = Ixxsum*Iyysum - Ixysum**2 # M的行列式
            Trace = Ixxsum + Iyysum # M的迹
            R = Determinant - alpha*(Trace**2) # 角点响应函数R
            if R > threshold:
                XCorners.append(j)
                YCorners.append(i)
                RValues.append(R)

    XCorners = np.asarray(XCorners)
    YCorners = np.asarray(YCorners)
    RValues = np.asarray(RValues)

    #Use ANMS to evenly distribute the corners in the image.

    NewCorners = ANMS(XCorners, YCorners, RValues, 5000) # 第三个参数是最终筛选出来的点个数
    NewCorners = np.asarray(NewCorners)

    x = NewCorners[:,0]
    y = NewCorners[:,1]

    return x,y


def get_features(image, x, y, feature_width):
    '''
    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''

    # TODO: Your implementation here! 
    # 初始化高斯滤波器，使用高斯模糊降低图像中的噪点，并获得图像的尺寸
    cutoff_frequency = 10
    filter1 = cv2.getGaussianKernel(ksize=4,sigma=cutoff_frequency)
    filter1 = np.dot(filter1, filter1.T)
    image = cv2.filter2D(image, -1, filter1)
    ImageRows = image.shape[0]
    ImageColumns = image.shape[1]
    Xcoordinates = len(x)
    FeatureVectorIn = np.ones((Xcoordinates,128))
    NormalizedFeature = np.zeros((Xcoordinates,128))

    for i in range(Xcoordinates):
        temp1 = int(x[i])
        temp2 = int(y[i])
        Window = image[temp2-8:temp2 + 8, temp1-8:temp1 + 8]
        WindowRows = Window.shape[0]
        WindowColumns = Window.shape[1]
        # 将window切分成16个4*4大小的window
        for p in range(4):
            for q in range(4):
                WindowCut = Window[p*4:p*4 +4,q*4: q*4+4] # 切分相应的坐标变换
                NewWindowCut = cv2.copyMakeBorder(WindowCut, 1, 1, 1, 1, cv2.BORDER_REFLECT)
                Magnitude = np.zeros((4,4))
                Orientation = np.zeros((4,4))

                for r in range(WindowCut.shape[0]):
                    for s in range(WindowCut.shape[1]):
                        Magnitude[r,s] = math.sqrt((NewWindowCut[r+1,s] - NewWindowCut[r-1,s])**2 + (NewWindowCut[r,s+1] - NewWindowCut[r,s-1])**2)
                        Orientation[r,s] = np.arctan2((NewWindowCut[r+1,s] - NewWindowCut[r-1,s]),(NewWindowCut[r,s+1] - NewWindowCut[r,s-1]))

                Magnitude = Magnitude
                OrientationNew = Orientation*(180/(math.pi))
                hist, edges = np.histogram(OrientationNew, bins = 8, range = (-180,180), weights = Magnitude)
                for t in range(8):
                    l = t+p*32+q*8
                    FeatureVectorIn[i,l] = hist[t]

    for a in range(FeatureVectorIn.shape[0]):
        sum1 = 0
        for b in range(FeatureVectorIn.shape[1]):
            sum1 = sum1 + (FeatureVectorIn[a][b])*(FeatureVectorIn[a][b])
        sum1 = math.sqrt(sum1)

        for c in range(FeatureVectorIn.shape[1]):
            NormalizedFeature[a][c] = FeatureVectorIn[a][c]/sum1

    features = NormalizedFeature
    return features

def match_features(im1_features, im2_features):
    '''
    :params:
    :im1_features: an np array of features returned from get_features() for interest points in image1
    :im2_features: an np array of features returned from get_features() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''
    # TODO: Your implementation here!
    Distance = np.zeros((im1_features.shape[0], im2_features.shape[0]))
    Value = []
    Hitx = []
    Hity = []
    for x in range(im1_features.shape[0]):
        for y in range(im2_features.shape[0]):
            ExtractedRow1 = im1_features[[x],:]
            ExtractedRow2 = im2_features[[y],:]
            SubtractedRow = ExtractedRow1 - ExtractedRow2
            Square = SubtractedRow*SubtractedRow
            Sum = Square.sum()
            Sum = math.sqrt(Sum)
            Distance[x,y] = Sum
        IndexPosition = np.argsort(Distance[x,:])
        d1 = IndexPosition[0]
        d2 = IndexPosition[1]
        Position1 = Distance[x,d1]
        Position2 = Distance[x,d2]
        ratio = Position1/Position2
        if ratio<0.8:  #Change to 0.9 while running Mount Rushmore
            Hitx.append(x)
            Hity.append(d1)
            Value.append(Position1)
    Xposition = np.asarray(Hitx)
    Yposition = np.asarray(Hity)
    matches = np.stack((Xposition,Yposition), axis = -1)
    confidences = np.asarray(Value)
    return matches, confidences
