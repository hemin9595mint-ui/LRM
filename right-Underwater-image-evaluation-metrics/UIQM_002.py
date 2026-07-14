import os
import cv2
import numpy as np
import time
from uiqm_utils import getUIQM  # 从你提供的工具文件中导入核心函数


def evaluate_uiqm_folder(folder_path):
    """
    遍历文件夹计算 UIQM 并求平均值
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹路径不存在 -> {folder_path}")
        return

    # 支持的图片格式
    img_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    # 获取所有图片文件
    file_list = [f for f in os.listdir(folder_path) if f.lower().endswith(img_extensions)]
    file_list.sort()  # 按文件名排序

    if len(file_list) == 0:
        print("未在文件夹中找到支持的图片文件。")
        return

    print(f"开始计算 UIQM... 目标文件夹: {folder_path}")
    print(f"共有 {len(file_list)} 张图片。\n")
    print(f"{'文件名':<30} | {'UIQM得分':<10}")
    print("-" * 45)

    scores = []
    start_time = time.time()

    for filename in file_list:
        file_path = os.path.join(folder_path, filename)

        # 1. 读取图片
        # 注意：OpenCV默认读取为 BGR，而 UIQM 计算通常基于 RGB 逻辑
        # (虽然部分无参指标对通道顺序不敏感，但为了严谨，我们转为 RGB)
        img = cv2.imread(file_path)

        if img is None:
            print(f"{filename:<30} | 读取失败 (可能是损坏的文件)")
            continue

        # 2. 转换颜色空间 BGR -> RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. 调用工具函数计算
        try:
            # uqim_utils 里的代码期望的是 numpy array，这里直接传入即可
            # 注意：某些版本的 uiqm 代码对图像尺寸有整除要求，但你提供的 utils 内部做了切片处理，应该没问题
            score = getUIQM(img_rgb)

            # 检查是否计算出 NaN 或 Inf
            if np.isnan(score) or np.isinf(score):
                print(f"{filename:<30} | 跳过 (数值异常: {score})")
            else:
                scores.append(score)
                print(f"{filename:<30} | {score:.4f}")

        except Exception as e:
            print(f"{filename:<30} | 计算出错: {e}")

    # --- 统计结果 ---
    print("-" * 45)
    if len(scores) > 0:
        avg_score = np.mean(scores)
        print(f"处理完成。")
        print(f"成功处理: {len(scores)} / {len(file_list)}")
        print(f"平均 UIQM: {avg_score:.4f}")

        # 可选：计算最大值和最小值对应的图片
        max_idx = np.argmax(scores)
        min_idx = np.argmin(scores)
        print(f"最高分: {scores[max_idx]:.4f} ({file_list[max_idx]})")
        print(f"最低分: {scores[min_idx]:.4f} ({file_list[min_idx]})")
    else:
        print("未能计算出任何有效分数。")

    print(f"总耗时: {time.time() - start_time:.2f} 秒")


if __name__ == '__main__':
    # ==========================================
    # 在这里修改你的图片文件夹路径
    # Windows系统建议在路径前加 r，如 r'D:\Dataset\Test'
    # ==========================================
    target_folder = r'UIQS_sum_results_TSA'

    evaluate_uiqm_folder(target_folder)

    '''
    python UIQM_002.py
    '''