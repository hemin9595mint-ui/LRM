import os
import cv2
import csv
import math
import numpy as np
from glob import glob


# =========================
# Safe image reading (supports Windows paths containing non-ASCII characters)
# =========================
def safe_imread(img_path):
    if not os.path.exists(img_path):
        return None
    try:
        data = np.fromfile(img_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


# =========================
# Sobel3
# =========================
def sobel(x):
    x = x.astype(np.float64)
    dx = cv2.Sobel(x, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(x, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(dx * dx + dy * dy)
    return mag


# =========================
# UICM
# =========================
def mu_a(x, alpha_L=0.1, alpha_R=0.1):
    x = sorted(x)
    K = len(x)
    if K == 0:
        return 0.0
    T_a_L = math.ceil(alpha_L * K)
    T_a_R = math.floor(alpha_R * K)
    denom = (K - T_a_L - T_a_R)
    if denom <= 0:
        return float(np.mean(x))
    weight = 1.0 / denom
    s = int(T_a_L + 1)
    e = int(K - T_a_R)
    val = sum(x[s:e])
    val = weight * val
    return val


def s_a(x, mu):
    if len(x) == 0:
        return 0.0
    val = 0.0
    for pixel in x:
        val += math.pow((pixel - mu), 2)
    return val / len(x)


def _uicm(x_rgb):
    # x_rgb: RGB
    R = x_rgb[:, :, 0].astype(np.float64).flatten()
    G = x_rgb[:, :, 1].astype(np.float64).flatten()
    B = x_rgb[:, :, 2].astype(np.float64).flatten()

    RG = R - G
    YB = ((R + G) / 2.0) - B

    mu_a_RG = mu_a(RG)
    mu_a_YB = mu_a(YB)

    s_a_RG = s_a(RG, mu_a_RG)
    s_a_YB = s_a(YB, mu_a_YB)

    l = math.sqrt((math.pow(mu_a_RG, 2) + math.pow(mu_a_YB, 2)))
    r = math.sqrt(s_a_RG + s_a_YB)

    return (-0.0268 * l) + (0.1586 * r)


# =========================
# UISM
# =========================
def eme(x, window_size):
    """
    x.shape[0] = height
    x.shape[1] = width
    """
    h, w = x.shape[:2]
    k1 = w // window_size
    k2 = h // window_size

    if k1 == 0 or k2 == 0:
        return 0.0

    weight = 2.0 / (k1 * k2)

    x = x[:k2 * window_size, :k1 * window_size]

    val = 0.0
    for l in range(k1):
        for k in range(k2):
            block = x[
                k * window_size: window_size * (k + 1),
                l * window_size: window_size * (l + 1)
            ]
            max_ = np.max(block)
            min_ = np.min(block)

            if min_ <= 0.0 or max_ <= 0.0:
                val += 0.0
            else:
                val += math.log(max_ / min_)

    return weight * val


def _uism(x_rgb):
    R = x_rgb[:, :, 0].astype(np.float64)
    G = x_rgb[:, :, 1].astype(np.float64)
    B = x_rgb[:, :, 2].astype(np.float64)

    Rs = sobel(R)
    Gs = sobel(G)
    Bs = sobel(B)

    R_edge_map = np.multiply(Rs, R)
    G_edge_map = np.multiply(Gs, G)
    B_edge_map = np.multiply(Bs, B)

    r_eme = eme(R_edge_map, 10)
    g_eme = eme(G_edge_map, 10)
    b_eme = eme(B_edge_map, 10)

    lambda_r = 0.299
    lambda_g = 0.587
    lambda_b = 0.144

    return (lambda_r * r_eme) + (lambda_g * g_eme) + (lambda_b * b_eme)


# =========================
# UICONM
# =========================
def _uiconm(x_rgb, window_size):
    """
    Underwater image contrast measure
    """
    h, w = x_rgb.shape[:2]
    k1 = w // window_size
    k2 = h // window_size

    if k1 == 0 or k2 == 0:
        return 0.0

    weight = -1.0 / (k1 * k2)

    x = x_rgb[:k2 * window_size, :k1 * window_size, :]

    alpha = 1.0
    val = 0.0

    for l in range(k1):
        for k in range(k2):
            block = x[
                k * window_size: window_size * (k + 1),
                l * window_size: window_size * (l + 1),
                :
            ]
            max_ = np.max(block)
            min_ = np.min(block)

            top = max_ - min_
            bot = max_ + min_

            if math.isnan(top) or math.isnan(bot) or bot == 0.0 or top == 0.0:
                val += 0.0
            else:
                ratio = top / bot
                val += alpha * math.pow(ratio, alpha) * math.log(ratio)

    return weight * val


# =========================
# UIQM
# =========================
def get_uiqm(img_bgr):
    # The formula assumes RGB channel order
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)

    uicm = _uicm(img_rgb)
    uism = _uism(img_rgb)
    uiconm = _uiconm(img_rgb, 10)

    # Standard UIQM weighted combination
    uiqm = (0.0282 * uicm) + (0.2953 * uism) + (3.5753 * uiconm)
    return uiqm, uicm, uism, uiconm


# =========================
# UCIQE
# =========================
def get_uciqe_from_img(img_bgr):
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    img_lab = np.array(img_lab, dtype=np.float64)

    coe_metric = [0.4680, 0.2745, 0.2576]

    img_lum = img_lab[:, :, 0] / 255.0
    img_a = img_lab[:, :, 1] / 255.0
    img_b = img_lab[:, :, 2] / 255.0

    chroma = np.sqrt(np.square(img_a) + np.square(img_b))
    sigma_c = np.std(chroma)

    img_lum_flat = img_lum.flatten()
    if len(img_lum_flat) == 0:
        return 0.0

    sorted_index = np.argsort(img_lum_flat)

    top_index = sorted_index[int(len(img_lum_flat) * 0.99)]
    bottom_index = sorted_index[int(len(img_lum_flat) * 0.01)]
    con_lum = img_lum_flat[top_index] - img_lum_flat[bottom_index]

    chroma_flat = chroma.flatten()
    sat = np.divide(
        chroma_flat,
        img_lum_flat,
        out=np.zeros_like(chroma_flat, dtype=np.float64),
        where=img_lum_flat != 0
    )
    avg_sat = np.mean(sat)

    uciqe = sigma_c * coe_metric[0] + con_lum * coe_metric[1] + avg_sat * coe_metric[2]
    return float(uciqe)


# =========================
# Collect image files
# =========================
def collect_images(folder):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    imgs = []
    for ext in exts:
        imgs.extend(glob(os.path.join(folder, ext)))
    return sorted(imgs)


# =========================
# Calculate the average metrics for all images in one subfolder
# =========================
def calculate_folder_metrics(folder):
    img_paths = collect_images(folder)

    if len(img_paths) == 0:
        print(f"[WARN] No images found in folder: {folder}")
        return None

    uciqe_list = []
    uiqm_list = []
    uicm_list = []
    uism_list = []
    uiconm_list = []

    bad_files = []

    for img_path in img_paths:
        img = safe_imread(img_path)
        if img is None:
            bad_files.append(img_path)
            continue

        try:
            uciqe_val = get_uciqe_from_img(img)
            uiqm_val, uicm_val, uism_val, uiconm_val = get_uiqm(img)

            uciqe_list.append(uciqe_val)
            uiqm_list.append(uiqm_val)
            uicm_list.append(uicm_val)
            uism_list.append(uism_val)
            uiconm_list.append(uiconm_val)

        except Exception as e:
            print(f"[WARN] Metric calculation failed: {img_path}")
            print(f"       Reason: {e}")
            bad_files.append(img_path)
            continue

    if len(uciqe_list) == 0:
        print(f"[WARN] No valid images are available in folder: {folder}")
        return None

    if len(bad_files) > 0:
        print(f"[WARN] Failed to read or process {len(bad_files)} image(s) in {folder}")
        for bf in bad_files[:5]:
            print("   ->", bf)
        if len(bad_files) > 5:
            print("   ...")

    return {
        "UCIQE": float(np.mean(uciqe_list)),
        "UIQM": float(np.mean(uiqm_list)),
        "UICM": float(np.mean(uicm_list)),
        "UISM": float(np.mean(uism_list)),
        "UICONM": float(np.mean(uiconm_list)),
        "num_images": len(uciqe_list)
    }


# =========================
# Evaluate the entire root directory
# =========================
def evaluate_all(root, save_csv="metrics_summary.csv"):
    algo_dirs = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

    all_rows = []

    for algo in algo_dirs:
        algo_path = os.path.join(root, algo)
        subfolders = sorted([d for d in os.listdir(algo_path) if os.path.isdir(os.path.join(algo_path, d))])

        print("\n==============================")
        print("Algorithm:", algo)

        sub_metrics = []

        for sub in subfolders:
            sub_path = os.path.join(algo_path, sub)
            metrics = calculate_folder_metrics(sub_path)

            if metrics is None:
                continue

            sub_metrics.append(metrics)

            print(f"  Subfolder: {sub}")
            print(f"    UCIQE : {metrics['UCIQE']:.4f}")
            print(f"    UIQM  : {metrics['UIQM']:.4f}")
            print(f"    UICM  : {metrics['UICM']:.4f}")
            print(f"    UISM  : {metrics['UISM']:.4f}")
            print(f"    UICONM: {metrics['UICONM']:.4f}")
            print(f"    Number of valid images: {metrics['num_images']}")

            all_rows.append([
                algo, sub,
                metrics["UCIQE"],
                metrics["UIQM"],
                metrics["UICM"],
                metrics["UISM"],
                metrics["UICONM"],
                metrics["num_images"]
            ])

        if len(sub_metrics) == 0:
            print(f"[WARN] No valid subfolders found for algorithm {algo}")
            continue

        avg = {
            "UCIQE": np.mean([m["UCIQE"] for m in sub_metrics]),
            "UIQM": np.mean([m["UIQM"] for m in sub_metrics]),
            "UICM": np.mean([m["UICM"] for m in sub_metrics]),
            "UISM": np.mean([m["UISM"] for m in sub_metrics]),
            "UICONM": np.mean([m["UICONM"] for m in sub_metrics]),
        }

        print("  ---- Algorithm Average ----")
        print(f"  UCIQE : {avg['UCIQE']:.4f}")
        print(f"  UIQM  : {avg['UIQM']:.4f}")
        print(f"  UICM  : {avg['UICM']:.4f}")
        print(f"  UISM  : {avg['UISM']:.4f}")
        print(f"  UICONM: {avg['UICONM']:.4f}")

        all_rows.append([
            algo, "AVERAGE",
            avg["UCIQE"],
            avg["UIQM"],
            avg["UICM"],
            avg["UISM"],
            avg["UICONM"],
            ""
        ])

    with open(save_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Algorithm", "Subfolder",
            "UCIQE", "UIQM", "UICM", "UISM", "UICONM", "ValidImages"
        ])
        writer.writerows(all_rows)

    print(f"\nResults saved to: {save_csv}")


if __name__ == "__main__":
    root = r""
    evaluate_all(root, save_csv="metrics_summary.csv")