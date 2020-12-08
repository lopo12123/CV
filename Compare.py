'''-*- coding: utf-8 -*-'''
import cv2
import numpy as np
'''计算两张图片的相似度'''


def aHash(img):
    '''均值哈希算法 average hash'''
    img_resized = cv2.resize(img, (8, 8))  # 缩放为 8*8
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

    gray_sum = 0  # s 为像素和（初始为0）
    hash_str = ''  # hash_str 为hash值（初始为空）

    for i in range(8):  # 遍历求像素和
        for j in range(8):
            gray_sum = gray_sum + img_gray[i, j]
    gray_average = gray_sum / 64  # 求平均灰度

    for i in range(8):  # 灰度大于平均则取1，相反则取0。生成图片的hash值
        for j in range(8):
            if img_gray[i, j] > gray_average:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'

    return hash_str  # 64位 8*8


def dHash(img):
    '''差值哈希算法 difference hash'''
    img_resized = cv2.resize(img, (8, 8))  # 缩放为 8*8
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

    hash_str = ''  # hash_str 为hash值（初始为空）

    for i in range(8):  # 每行前一个像素大于后一个像素为1，相反为0。生成图片的hash值
        for j in range(7):
            if img_gray[i, j] > img_gray[i, j + 1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'

    return hash_str  # 56位 8*7


def pHash(img):
    '''感知哈希 Perceptual hash'''
    img_resized = cv2.resize(img, (32, 32))  # 缩放为 32*32
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)  # 转换为灰度图

    dct = cv2.dct(np.float32(img_gray))  # 将灰度图转为浮点型，再进行dct变换获得img_gray的频域图像 / 可以使用cv2.idct（dct）恢复出原图像（有损）
    dct_roi = dct[0:8, 0:8]  # opencv 实现的掩码操作

    hash_str = []  # hash_str 为hash值（初始为空）
    average = np.mean(dct_roi)
    for i in range(dct_roi.shape[0]):
        for j in range(dct_roi.shape[1]):
            if dct_roi[i, j] > average:
                hash_str.append(1)
            else:
                hash_str.append(0)

    return hash_str  # 数组类型


def calculate(image1, image2):
    '''灰度单通道直方图'''
    '''范围:[0, 1]  值越大，相似度越高'''
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])  # 计算单通道的直方图的相似值
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])

    degree = 0  # 计算直方图的重合度
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)

    return degree  # 范围[0, 1]


def three_calculate(image1, image2, size=(256, 256)):
    '''RGB三通道直方图相似度'''
    '''范围:[0, 1]  值越大，相似度越高'''
    image1 = cv2.resize(image1, size)  # 将图像resize
    image2 = cv2.resize(image2, size)

    sub_image1 = cv2.split(image1)  # 分离为 RGB 三个通道
    sub_image2 = cv2.split(image2)

    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3

    return sub_data  # 范围[0, 1]


def cmp_Hash(hash1, hash2):
    '''Hash值对比'''
    '''两张图片通过相同算法获得一组哈希值。相同的位数越多，则图片越相似'''
    n = 0

    if len(hash1) != len(hash2):  # 如果两个hash长度不同则无法对比，返回-1报错
        return -1

    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:  # 有不同位则 n 计数 +1
            n += 1

    return n  # n越小则相似度越高


def Compare(img1, img2, num):
    '''img1/img2 是图片存储的路径'''
    '''num 是对比的方式: 1 - aHash
                        2 - dHash
                        3 - pHash
                        4 - 单通道
                        5 - 三通道'''
    route1 = img1  # 获取图片位置
    route2 = img2

    p1 = cv2.imread(route1)  # 读取图片
    p2 = cv2.imread(route2)

    if num == 1:  # aHash 方式对比
        result1 = aHash(p1)
        result2 = aHash(p2)
        result = cmp_Hash(result1, result2)
    elif num == 2:  # dHash 方式对比
        result1 = dHash(p1)
        result2 = dHash(p2)
        result = cmp_Hash(result1, result2)
    elif num == 3:  # pHash 方式对比
        result1 = pHash(p1)
        result2 = pHash(p2)
        result = cmp_Hash(result1, result2)
    elif num == 4:  # 单通道 方式对比
        result = calculate(p1, p2)
    elif num == 5:  # 三通道 方式对比
        result = three_calculate(p1, p2)

    return result


if __name__ == '__main__':
    ph1 = 'g1.jpg'
    ph2 = 'g2.jpg'
    result = Compare(ph1, ph2, 1)
    print('相似度：' + str(result))

    '''my_result = []
    my_result.append(Compare(ph1, ph2, 1))
    my_result.append(Compare(ph1, ph2, 2))
    my_result.append(Compare(ph1, ph2, 3))
    my_result.append(Compare(ph1, ph2, 4))
    my_result.append(Compare(ph1, ph2, 5))

    print('相似度：' + str(my_result[0]))
    print('相似度：' + str(my_result[1]))
    print('相似度：' + str(my_result[2]))
    print('相似度：' + str(my_result[3]))
    print('相似度：' + str(my_result[4]))'''

    # cv2.waitKey(0)
