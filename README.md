# 电子科技大学计算机视觉与模式识别课程作业

因为上课都在摸鱼，所以记录一下在做这门课课程作业时遇到的一些问题与学习过程。



### Proj 1: Image Filtering and Hybrid Images

项目内容：

- 基于skimage中所包含的常见滤波器，对图像（学生自定义）进行滤波，直观地对比不同滤波器的效果
- 编写my_filter()函数，实现高通滤波和低通滤波两种滤波形式
- 通过my_filter()函数对图像进行高低通滤波，并且对图像分别在高低频进行融合。

![image-20230418165939485](C:\Users\FTCY\Desktop\Projects\image\image-20230418165939485.png)

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

![image-20230419221206978](C:\Users\FTCY\Desktop\Projects\image\image-20230419221206978.png)