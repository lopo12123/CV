## CV
    use lib: opencv-python

    Search.py
    输入两张图片img1(source), img2(part);
    在img1中寻找并框出img2的位置
    - 返回坐标
    - 没有匹配到则返回'未找到'

    Compare.py
    输入两张图片ph1, ph2;
    五种方式比较相似度
    - 返回相似度
    

### 滤波器
    dst = cv.bilateralFilter(img_gray, 9, 75, 75)  # 强化边界，模糊纹理