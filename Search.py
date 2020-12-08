import cv2
import numpy as np


def Search(my_img, my_template, my_threshold=0.9):
    '''在大图中找小图，并返回坐标点位, my_threshold 是查询的精度(默认为0.9))'''
    '''img 是原始大图 p1; template 是需要找的小图 p2'''
    scale = 1  # 缩放因子

    route1 = my_img  # 获取图片位置
    route2 = my_template

    p1 = cv2.imread(route1)  # 读取图片
    p2 = cv2.imread(route2)

    p1 = cv2.resize(p1, (0, 0), fx=scale, fy=scale)  # 按比例缩放
    p2 = cv2.resize(p2, (0, 0), fx=scale, fy=scale)
    p2_size = p2.shape[:2]

    p1_gray = cv2.cvtColor(p1, cv2.COLOR_BGR2GRAY)  # 转换为灰度
    p2_gray = cv2.cvtColor(p2, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(p1_gray, p2_gray, cv2.TM_CCOEFF_NORMED)  # 匹配

    threshold = my_threshold  # res 大于 my_threshold(0-1)
    loc = np.where(result >= threshold)  # 使用灰度图像中的坐标对原始RGB图像进行标记

    point = ()
    for pt in zip(*loc[::-1]):  # 单个*用法：1、元组、字典等类型自动解包（此处用法）；2、接收任意多个参数并存入元组中
        # cv2.rectangle(img, (bbox.left, bbox.top), (bbox.right, bbox.bottom), (B, G, R), width)
        cv2.rectangle(p1, pt, (pt[0] + p2_size[1], pt[1] + + p2_size[0]), (0, 255, 0), 1)  # 缺点：会重复画框，没有找出最优解
        point = pt
    if point == ():  # 没有匹配到，即不包含目标图片
        return None, None, None
    return p1, point[0], point[1]  # 返回加上标记（矩形框）的图片和标记矩形框的左上角坐标


if __name__ == '__main__':
    pp1 = 'search/Source.png'
    pp2 = 'search/part.png'

    img, x, y = Search(pp1, pp2, 0.9)  # 第三位数据为精度，需要进行调整

    if img is None:
        print('没找到')
    else:
        # cv2.imshow('11', img)
        cv2.imwrite('result/re.png', img)

    cv2.waitKey(0)
