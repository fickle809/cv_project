# UESTC计算机视觉与模式识别课程作业

因为上课都在摸鱼，所以记录一下在做这门课课程作业时遇到的一些问题与学习过程。



### Proj 1: Image Filtering and Hybrid Images

项目内容：

- 基于skimage中所包含的常见滤波器，对图像（学生自定义）进行滤波，直观地对比不同滤波器的效果
- 编写my_filter()函数，实现高通滤波和低通滤波两种滤波形式
- 通过my_filter()函数对图像进行高低通滤波，并且对图像分别在高低频进行融合。

![image-20230418165939485](https://raw.githubusercontent.com/fickle809/cv_project/main/image/image-20230418165939485.png)

两个主要的任务：**图像过滤**和**混合图像**

#### 项目原理：

1. 图像的**频率**：频率是灰度值**变化剧烈程度**的指标，是灰度在平面空间上的梯度。图像中的低频信号和高频信号也叫做低频分量和高频分量。

2. **低频**:低频就是颜色缓慢地变化，也就是灰度缓慢地变化，就代表着那是连续渐变的一块区域，这部分就是低频。 对于一幅图像来说，除去高频的就是低频了，也就是边缘以内的内容为低频，而边缘内的内容就是图像的大部分信息，即图像的大致概貌和轮廓，是图像的近似信息。所以说低频分量主要对整幅图像的**强度**的综合度量。

3. **高频**:反过来，高频就是频率变化快。当相邻区域之间灰度相差很大，这就是变化得快。图像中，一个影像与背景的边缘部位，通常会有明显的差别，也就是说变化那条边线那里，灰度变化很快，也即是变化频率高的部位。因此，图像边缘的灰度值变化快，就对应着频率高，即高频显示图像边缘。图像的细节处也是属于灰度值急剧变化的区域，正是因为灰度值的急剧变化，才会出现细节。所以说高频分量主要是对图像边缘和轮廓的度量。

   另外噪声（即噪点）也是这样，当一个像素所在的位置与正常的点颜色不一样（该像素点灰度值明显不一样，也就是灰度有快速地变化）时可视为噪声。



#### my_imfilter函数测试

```python
test_image = load_image('../data/cat.bmp')
print(test_image.shape)  # (361,410,3)
test_image = rescale(test_image, [0.7, 0.7, 1], mode='reflect') # 按一定比例缩放图像，根据给定的模式reflect填充输入边界之外的点
print(test_image.shape)  # (253,287,3)
```

rescale():按比例缩放

函数格式为skimage.transform.rescale(image, scale[, ...])

scale参数可以是单个float数，表示缩放的倍数，也可以是一个float型的tuple，如[0.2,0.5],表示将行列数分开进行缩放

```python
identity_filter = np.asarray([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
identity_image = my_imfilter(test_image, identity_filter)
plt.imshow(identity_image)
done = save_image('../results/identity_image.jpg', identity_image)
```

my_imfilter就是我们需要在student.py中补全的函数

功能：对图像应用滤波，返回过滤后的图像。
输入:

- image -> numpy nd-array (m, n, c)
- filter -> numpy array of odd dim (k, l)
返回
- filtered_image -> numpy nd- dim (m, n, c)
错误:
- filter有任何偶数维度->抛出一个异常，并给出合适的错误消息。

##### 图像滤波处理公式：

![image-20230419221206978](https://raw.githubusercontent.com/fickle809/cv_project/main/image/image-20230419221206978.png)

参照课件上的公式，因为要返回的filtered_image的维度和处理前的image是相同的，所以要根据filter的大小来填充image_padding，即在image的四周补0，使得其维度变为(m+k,n+l,c)（第三维度不用改变）

my_imfilter函数内部如下：

```python
image_padding = np.zeros([image.shape[0]+(filter.shape[0]-1),image.shape[1]+(filter.shape[1]-1),image.shape[2]])
  filtered_image = np.zeros(image.shape)
  image_padding[(filter.shape[0]-1)//2:(filter.shape[0]-1)//2+image.shape[0],(filter.shape[1]-1)//2:(filter.shape[1]-1)//2+image.shape[1]]=image

  for k in range(image.shape[2]):
      for i in range(image.shape[0]):
          for j in range(image.shape[1]):
              convolute_image = image_padding[i:i+filter.shape[0],j:j+filter.shape[1],k]
              # reshape_image = convolute_image.reshape(-1,1)
              # filtered_image[i][j][k] = sum(np.multiply(reshape_image,filter))
              filtered_image[i][j][k] = sum(sum(np.multiply(convolute_image,filter)))
```

具体处理时用到了np.multiply函数，注意这里不是矩阵乘法，而是两个维度相同的矩阵对应位置做乘积（哈达玛积），最后我们需要将做乘积后的结果矩阵每个位置上的数加起来，作为`filtered_image[i][j][k]`的值。

（注释部分时刚开始写的另一种处理方法，但是好像在后面高通低通图像融合的时候结果会有问题-_-）

这部分处理后的图像：

1. 不做处理

<img src="C:\Users\FTCY\Desktop\Projects\image\不做处理.png" style="zoom:50%;" />

2. small blur（低通模糊处理）

<img src="C:\Users\FTCY\Desktop\Projects\image\small blur.png" style="zoom:50%;" />

3. large blur（高糊）用到了高斯模糊

高斯滤波器是一种线性滤波器，能够有效的抑制噪声，平滑图像。其作用原理和均值滤波器类似，都是取滤波器窗口内的像素的均值作为输出。其窗口模板的系数和均值滤波器不同，均值滤波器的模板系数都是相同的为1;而高斯滤波器的模板系数，则随着距离模板中心的增大而系数减小。所以，高斯滤波器相比于均值滤波器对图像个模糊程度较小。

<img src="C:\Users\FTCY\Desktop\Projects\image\large blur.png" style="zoom:50%;" />

4. sobel算子（高通处理）

参考了[这篇文章](https://blog.csdn.net/qq_43010987/article/details/121641734)，之前做的时候感觉这个滤波就像是朝着竖直方向的，果然看了这篇文章后发现sobel算子有竖直和水平两个方向的处理：

<img src="C:\Users\FTCY\Desktop\Projects\image\sobel算子.png" style="zoom:80%;" />

<img src="C:\Users\FTCY\Desktop\Projects\image\sobel 竖直方向.png" style="zoom:50%;" />

<img src="C:\Users\FTCY\Desktop\Projects\image\sobel水平.png" style="zoom:50%;" />

5. 拉普拉斯算子（高通）

拉普拉斯算子是图像邻域内像素灰度差分计算的基础，通过二阶微分推导出的一种图像邻域增强算法。它的基本思想是当邻域的中心像素灰度低于它所在邻域内的其他像素的平均灰度时，此中心像素的灰度应该进一步降低；当高于时进一步提高中心像素的灰度，从而实现图像锐化处理。
在算法实现过程中，通过对邻域中心像素的四方向或八方向求梯度，并将梯度和相加来判断中心像素灰度与邻域内其他像素灰度的关系，并用梯度运算的结果对像素灰度进行调整。
[参考](https://blog.csdn.net/weixin_42415138/article/details/108574657)

<img src="C:\Users\FTCY\Desktop\Projects\image\拉普拉斯1.png" style="zoom:50%;" />

<img src="C:\Users\FTCY\Desktop\Projects\image\拉普拉斯2.png" style="zoom:50%;" />

#### gen_hybrid_image函数部分

其实哪怕没听课，把my_imfilter函数处理部分写出来，这个project也就差不多了，因为哪怕不清楚高通低通filter的实现，也能够直接调用给出的filter处理得到结果。嗯，因为我也只是为了完成project也只是随便了解了一下没去深究。

```python
# Your code here:
  large_blur_image1 = my_imfilter(image1,kernel)
  low_frequencies = large_blur_image1 

  large_blur_image2 = my_imfilter(image2,kernel)
  high_frequencies = image2 - large_blur_image2 

  hybrid_image = low_frequencies + high_frequencies

  for i in range(hybrid_image.shape[0]):
     for j in range(hybrid_image.shape[1]):
        for k in range(hybrid_image.shape[2]):
           if(hybrid_image[i][j][k]>1.0):
              hybrid_image[i][j][k]=1.0
           if(hybrid_image[i][j][k]<0.0):
              hybrid_image[i][j][k]=0.0
```

混合图像这部分：高斯kernel已经给了，所以我们要做的就是四步

1. 对image1进行高斯低通处理
2. 对image2进行高斯低通处理后，用原始图像image2-低通处理的图像得到高通处理的图像
3. 将低通处理后的image1和image2直接相加
4. clip处理

结果：

<img src="C:\Users\FTCY\Desktop\Projects\image\image1.png" style="zoom:50%;" />

<img src="C:\Users\FTCY\Desktop\Projects\image\image2.png" alt="image2" style="zoom:50%;" />

<img src="C:\Users\FTCY\Desktop\Projects\image\混合.png" alt="混合" style="zoom:50%;" />



### Proj 2: Local Feature Match

项目内容：

- 实现对SIFT特征的具体细节理解和简单的距离计算
- 关键技术：Harris兴趣点提取->基于SIFT的特征描述->特征匹配

![image-20230419233548317](C:\Users\FTCY\AppData\Roaming\Typora\typora-user-images\image-20230419233548317.png)

1. 加载和调整图像大小
2. 在这些图像中找到兴趣点(编写代码)
3. 用本地特性描述每个兴趣点(编写代码)
4. 查找匹配的特性(编写代码)
5. 可视化匹配
6. 基于基础真值对应来评估匹配

#### get_interest_points函数部分

<img src="C:\Users\FTCY\AppData\Roaming\Typora\typora-user-images\image-20230420221721039.png" alt="image-20230420221721039" style="zoom:50%;" />



##### Harris角点检测

算法原理：

算法的核心是利用局部窗口在图像上进行移动，判断灰度是否发生较大的变化。如果窗口内的灰度值（在梯度图上）都有较大的变化，那么这个窗口所在区域就存在角点。

这样就可以将 Harris 角点检测算法分为以下三步：

- 当窗口（局部区域）同时向 x （水平）和 y（垂直） 两个方向移动时，计算窗口内部的像素值变化量 **E(x,y)** 
- 对于每个窗口，都计算其对应的一个角点响应函数 **R**
- 然后对该函数进行**阈值处理**，如果 R>threshold，表示该窗口对应一个角点特征



<img src="C:\Users\FTCY\AppData\Roaming\Typora\typora-user-images\image-20230420222050521.png" alt="image-20230420222050521" style="zoom:50%;" />

**对上图公式的解释如下：**

建立数学模型，确定哪些窗口会引起较大的灰度值变化。 让一个窗口的中心位于灰度图像的一个位置 $(x,y)$，这个位置的像素灰度值为 $I(x,y)$ ，如果这个窗口分别向 $x$ 和 $y$ 方向移动一个小的位移 $u$ 和 $v$ ，到一个新的位置 $(x+u,y+v)$，这个位置的像素灰度值就是 $I(x+u,y+v)$。$|I(x+u,y+v)-I(x,y)|$就是窗口移动引起的灰度值的变化值。

设 $w(x,y)$ 为位置$(x,y)$处的窗口函数，表示窗口内各像素的权重，最简单的就是把窗口内所有像素的权重都设为1，即一个均值滤波核。

当然，也可以把 $w(x,y)$ 设定为以窗口中心为原点的高斯分布，即一个高斯核。如果窗口中心点像素是角点，那么窗口移动前后，中心点的灰度值变化非常强烈，所以该点权重系数应该设大一点，表示该点对灰度变化的贡献较大；而离窗口中心（角点）较远的点，这些点的灰度变化比较小，于是将权重系数设小一点，表示该点对灰度变化的贡献较小。



若窗口内是一个角点，则E(u,v)的计算结果将会很大。

为了提高计算效率，对上述公式进行简化，利用泰勒级数展开来得到这个公式的近似形式：

<img src="https://pic2.zhimg.com/v2-2ae0bb13dafd974423d527a11ccd26f1_r.jpg" alt="img" style="zoom:50%;" />



<img src="C:\Users\FTCY\AppData\Roaming\Typora\typora-user-images\image-20230420225402468.png" alt="image-20230420225402468" style="zoom:50%;" />

![image-20230425173533119](C:\Users\FTCY\AppData\Roaming\Typora\typora-user-images\image-20230425173533119.png)

则根据两个特征值得到结论

- 如果矩阵对应的两个特征值都较大，那么窗口内含有角点
- 如果特征值一个大一个小，那么窗口内含有线性边缘
- 如果两个特征值都很小，那么窗口内为平坦区域

下面再定义一个式子是角点响应函数R

![image-20230425173645156](C:\Users\FTCY\AppData\Roaming\Typora\typora-user-images\image-20230425173645156.png)

同理可得：

- 角点：R 为大数值正数
- 边缘：R为大数值负数
- 平坦区：R为小数值



接下来就是基于上面的公式写代码了：

```python
def get_interest_points(image, feature_width):
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
```

使用Harris Corner Detector生成的角点不是均匀分布的。在许多情况下角点可能集中在图像的特定区域，从而导致准确性的下降。NMS算法通过在特征函数中寻找局部最大值并丢弃剩余的次大值（即图像中的临近角点）。通过使用该算法，可以提高特征匹配的准确性，但是传统的NMS具有某些局限性，例如图像中特征点在处理后可能会不均匀分布。为了避免这种情况，所以使用ANMS算法（自适应非最大值抑制），基本思想是仅保留r个像素附近最大的那些点。

```python
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
```



<img src="C:\Users\FTCY\AppData\Roaming\Typora\typora-user-images\image-20230421002451977.png" alt="image-20230421002451977" style="zoom:67%;" />

<img src="C:\Users\FTCY\AppData\Roaming\Typora\typora-user-images\image-20230421002512135.png" alt="image-20230421002512135" style="zoom:67%;" />

#### get_features函数部分

目标：返回一组给定兴趣点的特征描述符。

输入：image图像，兴趣点的坐标，局部特征维度

输出：规范化特征向量

- 初始化高斯滤波器，使用高斯模糊降低图像中的噪点，并获得图像的尺寸

```python
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
```

- 遍历Harris算法得到的每一个角点坐标，提取以该角点坐标为中心的16x16像素的一个局部图像，对于每一个局部图像，拆分成16个4x4像素的窗口，计算窗口内每个像素的大小和方向。

```python
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
```

你可能会疑惑最后两行是如何冒出来的，可以参考一下[这篇文章](https://blog.csdn.net/weixin_48167570/article/details/123704075?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522168242153216800215035827%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=168242153216800215035827&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-123704075-null-null.142^v86^control,239^v2^insert_chatgpt&utm_term=SIFT&spm=1018.2226.3001.4187)，有一个推导（但是我没看），最终的公式是：

<img src="C:\Users\FTCY\AppData\Roaming\Typora\typora-user-images\image-20230425192449963.png" alt="image-20230425192449963" style="zoom:80%;" />

将区域划分为4x4的子块，对每一个子块进行8个方向的直方图统计操作，获得每个方向的梯度幅值，总共可以组成128维描述向量。

对于每一个关键点，都拥有位置、尺度以及方向三个信息。为每个关键点建立一个描述符，用一组向量将这个关键点描述出来，使其不随各种变化而改变，比如光照变化、视角变化等等。这个描述子不但包括关键点，也包含关键点周围对其有贡献的像素点，并且描述符应该有较高的独特性，以便于提高特征点正确匹配的概率

```python
for p in range(4):
	for q in range(4):
        WindowCut = Window[p*4:p*4 +4,q*4: q*4+4] # 切分相应的坐标变换
        NewWindowCut = cv2.copyMakeBorder(WindowCut, 1, 1, 1, 1, cv2.BORDER_REFLECT)
        Magnitude = np.zeros((4,4))
        Orientation = np.zeros((4,4))
```

- 正则化特征向量后返回值

```python
for a in range(FeatureVectorIn.shape[0]):
        sum1 = 0
        for b in range(FeatureVectorIn.shape[1]):
            sum1 = sum1 + (FeatureVectorIn[a][b])*(FeatureVectorIn[a][b])
        sum1 = math.sqrt(sum1)
        for c in range(FeatureVectorIn.shape[1]):
            NormalizedFeature[a][c] = FeatureVectorIn[a][c]/sum1

    features = NormalizedFeature
    return features
```

#### match_features函数部分

计算第一图像中每个关键点的每个特征向量到第二图像中每个特征向量的距离。然后对距离进行排序，并获取和比较两个最小距离（欧式距离）。为了使关键点与第二个图像中的另一个关键点精确匹配，计算并检查两个最小距离的比率是否大于指定的阈值，之后将其视为关键点。

```python
def match_features(im1_features, im2_features):
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

```

![image-20230425231013418](C:\Users\FTCY\AppData\Roaming\Typora\typora-user-images\image-20230425231013418.png)

![img](file:///C:\Users\FTCY\AppData\Local\Temp\ksohtml14764\wps2.jpg)

总结：SIFT特征具有稳定性和不变性，在图像处理和计算机视觉领域有着很重要的作用，其本身也是非常复杂的。





后面两个proj相对来说代码量比较少，也比较简单，没花太多时间，就直接看代码吧

### Proj 3: Face Detection and Swap with OpenCV+Dlib

项目内容：

- 基于opencv和Dlib实现面部识别，并且实现换脸

![image-20230419233715759](C:\Users\FTCY\AppData\Roaming\Typora\typora-user-images\image-20230419233715759.png)

### Proj 4: Scene Recognition with Bag of Words

项目内容：

- 分别利用 Tiny+KNN 和 Bags of Words (SIFT) + SVM 实现对场景的分类，并且比较不同特征和不同分类方法对最终精度的影响

![image-20230419233746108](C:\Users\FTCY\AppData\Roaming\Typora\typora-user-images\image-20230419233746108.png)

​	这个实验是使用词袋模型对十五个场景数据库进行场景识别，达到图像分类的目的。

<img src="C:\Users\FTCY\AppData\Roaming\Typora\typora-user-images\image-20230430092120346.png" alt="image-20230430092120346" style="zoom: 50%;" />

​	词袋模型的步骤如上图所示：

- 特征提取
- 学习视觉词典
- 通过视觉词典量化特征
- 用视觉词典的频率来表示图像

