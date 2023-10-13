"""
Author:      郑辉
StudentID:   23320231154460
Date:        2023-10-12
File:        abstract_a4.py
Description: 计算机视觉课程作业一：识别A4纸并进行透视变换
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

A4_SIZE = (210, 297)  # A4纸的尺寸，单位毫米


def read_img(img_path: str) -> np.ndarray:
    """
    读取图片,返回图片BGR数据
    :param img_path:
    :return: BGR三维数组
    """
    img = cv2.imread(img_path)
    return img


def show_imgs(lined_img: np.ndarray, a4_img: np.ndarray) -> None:
    """
    显示标注a4区域的图片与透视变换后的图片
    :param lined_img: 标注a4区域的图片
    :param a4_img: 透视变换后的图片
    :return: None
    """
    plt.subplot(121)
    plt.imshow(lined_img)
    plt.subplot(122)
    plt.imshow(a4_img)
    plt.show()


def img_preprocess(img_data: np.ndarray) -> np.ndarray:
    """
    图片预处理
    :param img_data: BGR三维数组
    :return: 边缘检测后的灰度图二维数组
    """
    gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(gray, 30, 100)
    # 对图像进行膨胀操作，使得轮廓更加清晰
    edge = cv2.dilate(edge, None)
    # show_gray(edge)
    return edge


def get_area_points(edges: np.ndarray) -> np.ndarray:
    """
    获取A4纸的四个角点
    :param edges: 边缘检测后的灰度图二维数组
    :return: A4纸的四个角点
    """
    # 获取轮廓
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 按轮廓面积降序排序
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 获取轮廓的近似多边形
    cnt = contours[0]
    # epsilon为逼近精度，一般情况下，使用0.02*周长作为逼近精度
    epsilon = 0.02*cv2.arcLength(cnt, True)
    # approx为逼近后的四个角点坐标
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    # 对四个角点进行排序，得到[左上，右上，右下，左下]
    # 首先判断各点距离左上角点的距离，距离越小，越靠近左上角
    # 确定左上和右下后，判断右上和左下，y坐标越小，越靠近左上和右上
    pts1 = np.int32(
        sorted(approx, key=lambda x: (x[0][0] ** 2 + x[0][1] ** 2)))
    pts1 = pts1[[0, 1, 3, 2],
                0] if pts1[1][0][1] < pts1[2][0][1] else pts1[[0, 2, 3, 1], 0]
    return pts1


def get_area_lengths(area_points: np.ndarray) -> list[float]:
    """
    计算选取区域的四条边的长度
    :param area_points: 选取区域的四个角点
    :return: 四条边的长度
    """
    lengths = []

    for i in range(4):
        x1, y1 = area_points[i][0], area_points[i][1]
        x2, y2 = area_points[(i+1) % 4][0], area_points[(i+1) % 4][1]
        length = np.sqrt((x1-x2)**2 + (y1-y2)**2)
        lengths.append(length)

    return lengths


def get_page_wh(area_lengths: list[float]) -> tuple[int, int]:
    """
    计算新的纸张图片宽高，当纸张横放时，宽边被压缩，长边不变，
    以长边为基准计算新的宽高，反之亦然
    :param area_lengths: 选取区域的四条边的长度
    :return: 新的纸张图片宽高（默认竖放）
    """
    sorted_lengths = sorted(area_lengths, reverse=True)
    # 判断哪边被压缩，当长/宽 > A4纸长/宽时，宽边被压缩
    if sorted_lengths[0] / sorted_lengths[2] > A4_SIZE[1] / A4_SIZE[0]:
        # 纸张横着放
        width = sorted_lengths[0]
        height = sorted_lengths[0]*(A4_SIZE[1]/A4_SIZE[0])
    else:
        # 纸张竖着放
        width = sorted_lengths[2]
        height = sorted_lengths[2]*(A4_SIZE[1]/A4_SIZE[0])
    return int(width), int(height)


def get_aligned_paper(img_data: np.ndarray, area_points: np.ndarray) -> np.ndarray:
    """
    获取矫正后的纸张图片
    ---0---
    |     |
    3     1
    |     |
    ---2---
    :param img_data: BGR三维数组
    :param area_points: 选取区域的四个角点
    :return: 矫正后的纸张图片
    """
    area_lengths = get_area_lengths(area_points)
    # 判断纸张横放还是竖放
    if area_lengths[0] < area_lengths[1]:
        # 纸张竖放
        width, height = get_page_wh(area_lengths)
    else:
        # 纸张横放
        height, width = get_page_wh(area_lengths)

    # 定义A4纸的四个角点
    paper_corners = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    # 透视变换矩阵
    matrix = cv2.getPerspectiveTransform(
        np.float32(area_points), paper_corners)
    # 进行透视变换
    aligned_paper = cv2.warpPerspective(img_data, matrix, (width, height))
    return aligned_paper


def add_lines(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    给选取区域的四条边加上红色的线
    :param img: BGR三维数组
    :param points: 选取区域的四个角点
    :return: 加上红色线的图片
    """
    for i in range(4):
        x1, y1 = points[i][0], points[i][1]
        x2, y2 = points[(i+1) % 4][0], points[(i+1) % 4][1]
        img = cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 7)
    return img


if __name__ == "__main__":
    img_path = 'img1.jpg'
    img_data = read_img(img_path)
    edges = img_preprocess(img_data)
    area_points = get_area_points(edges)
    aligned_paper = get_aligned_paper(img_data, area_points)
    lined_img = add_lines(img_data, area_points)
    show_imgs(lined_img, aligned_paper)
    cv2.imwrite(f'a4_{img_path}', aligned_paper)
