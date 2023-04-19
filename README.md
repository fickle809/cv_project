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
