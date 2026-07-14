import cv2
import numpy as np
import scipy.signal
import os
import time


# --- 1. PCQI 计算函数 ---
def PCQI(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    window = np.multiply(cv2.getGaussianKernel(11, 1.5), (cv2.getGaussianKernel(11, 1.5)).T)
    L = 256
    window = window / np.sum(np.sum(window))

    mu1 = scipy.signal.correlate2d(img1, window, 'valid')
    mu2 = scipy.signal.correlate2d(img2, window, 'valid')
    mu1_sq = mu1 * mu1

    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = scipy.signal.correlate2d(img1 * img1, window, 'valid') - mu1_sq
    sigma2_sq = scipy.signal.correlate2d(img2 * img2, window, 'valid') - mu2_sq
    sigma12 = scipy.signal.correlate2d(img1 * img2, window, 'valid') - mu1_mu2

    sigma1_sq[sigma1_sq < 0] = 0
    sigma2_sq[sigma2_sq < 0] = 0

    C = 3

    pcqi_map = (4 / np.pi) * np.arctan((sigma12 + C) / (sigma1_sq + C))
    pcqi_map = pcqi_map * ((sigma12 + C) / (np.sqrt(sigma1_sq) * np.sqrt(sigma2_sq) + C))
    pcqi_map = pcqi_map * np.exp(-abs(mu1 - mu2) / L)

    mpcqi = np.mean(pcqi_map)

    return mpcqi


# --- 2. AG (平均梯度) 计算函数 ---
def calculate_ag(img):
    img = img.astype(np.float64)
    dx, dy = np.gradient(img)
    gradient_mag = np.sqrt((dx ** 2 + dy ** 2) / 2)
    ag = np.mean(gradient_mag)
    return ag


# --- 3. EI (边缘强度) 计算函数 ---
def calculate_ei(img):
    img = img.astype(np.float64)
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    ei = np.mean(magnitude)
    return ei


# --- 4. 批量处理逻辑 ---
def evaluate_images(ref_dir, dist_dir):
    """
    ref_dir: 原始图像文件夹路径 (Reference)
    dist_dir: 处理后图像文件夹路径 (Distorted/Enhanced)
    """

    if not os.path.exists(ref_dir) or not os.path.exists(dist_dir):
        print("错误：文件夹路径不存在，请检查路径。")
        return

    file_list = os.listdir(ref_dir)
    img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    img_files = [f for f in file_list if f.lower().endswith(img_extensions)]

    scores = []
    ag_scores_ref = []
    ag_scores_dist = []
    ei_scores_ref = []
    ei_scores_dist = []

    print(f"开始处理... 共有 {len(img_files)} 张图片待处理。\n")
    print(f"{'文件名':<30} | {'PCQI得分':<10} | {'AG(原图)':<8} {'AG(处理)':<8} | {'EI(原图)':<8} {'EI(处理)':<8}")
    print("-" * 80)

    start_time = time.time()

    for filename in img_files:
        ref_path = os.path.join(ref_dir, filename)
        dist_path = os.path.join(dist_dir, filename)

        if not os.path.exists(dist_path):
            print(f"{filename:<30} | 跳过 (在目标文件夹中未找到对应文件)")
            continue

        ref_img = cv2.imread(ref_path, 0)
        dist_img = cv2.imread(dist_path, 0)

        if ref_img is None or dist_img is None:
            print(f"{filename:<30} | 跳过 (文件读取失败)")
            continue

        if ref_img.shape != dist_img.shape:
            dist_img = cv2.resize(dist_img, (ref_img.shape[1], ref_img.shape[0]))

        try:
            # 计算 PCQI
            pcqi_score = PCQI(ref_img, dist_img)
            scores.append(pcqi_score)

            # 计算 AG 和 EI
            ag_ref = calculate_ag(ref_img)
            ag_dist = calculate_ag(dist_img)
            ei_ref = calculate_ei(ref_img)
            ei_dist = calculate_ei(dist_img)

            ag_scores_ref.append(ag_ref)
            ag_scores_dist.append(ag_dist)
            ei_scores_ref.append(ei_ref)
            ei_scores_dist.append(ei_dist)

            print(f"{filename:<30} | {pcqi_score:.4f} | {ag_ref:6.2f} -> {ag_dist:6.2f} | {ei_ref:6.2f} -> {ei_dist:6.2f}")
        except Exception as e:
            print(f"{filename:<30} | 错误: {e}")

    print("-" * 80)
    if len(scores) > 0:
        avg_pcqi = np.mean(scores)
        avg_ag_ref = np.mean(ag_scores_ref)
        avg_ag_dist = np.mean(ag_scores_dist)
        avg_ei_ref = np.mean(ei_scores_ref)
        avg_ei_dist = np.mean(ei_scores_dist)

        print(f"处理完成。")
        print(f"成功处理图片数: {len(scores)}")
        print(f"平均 PCQI 得分: {avg_pcqi:.4f}")
        print(f"平均 AG (原图): {avg_ag_ref:.2f}, 平均 AG (处理后): {avg_ag_dist:.2f}")
        print(f"平均 EI (原图): {avg_ei_ref:.2f}, 平均 EI (处理后): {avg_ei_dist:.2f}")

        # 简单的提升分析
        print("\n--- 分析结论 ---")
        ag_gain = (avg_ag_dist - avg_ag_ref) / avg_ag_ref * 100
        ei_gain = (avg_ei_dist - avg_ei_ref) / avg_ei_ref * 100
        print(f"AG (清晰度) 变化: {'提升' if ag_gain > 0 else '下降'} {abs(ag_gain):.2f}%")
        print(f"EI (边缘强度) 变化: {'提升' if ei_gain > 0 else '下降'} {abs(ei_gain):.2f}%")
    else:
        print("未成功计算任何图片，请检查文件夹路径和文件名是否匹配。")

    print(f"总耗时: {time.time() - start_time:.2f} 秒")


# --- 主程序入口 ---
if __name__ == '__main__':
    # 修改文件夹路径
    # reference_folder = "UIQS_sum"  # 原始图像文件夹
    # target_folder = "UIQS_sum_results_CLAHE"  # 处理后图像文件夹
    # reference_folder = "UCCS_sum"  # 原始图像文件夹
    # target_folder = "UCCS_sum_results_GDCP"  # 处理后图像文件夹
    reference_folder = "Realraw-890"  # 原始图像文件夹
    target_folder = "Realraw-890_results_004"  # 处理后图像文件夹

    evaluate_images(reference_folder, target_folder)
