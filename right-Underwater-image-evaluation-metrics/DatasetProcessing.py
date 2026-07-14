import os

def rename_images_in_folder(folder_path):
    # 获取文件夹内所有文件
    files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]

    # 按字母排序文件
    files.sort()

    # 重命名文件
    for idx, file_name in enumerate(files, start=1):
        # 新文件名，保持6位数格式
        new_name = f"{str(idx).zfill(6)}.png"

        # 获取文件的完整路径
        old_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, new_name)

        # 重命名
        os.rename(old_path, new_path)
        print(f"Renamed: {file_name} -> {new_name}")


# 设定目标文件夹路径
folder_path = "UCCS_sum"  # 修改为你实际的文件夹路径

# 调用重命名函数
rename_images_in_folder(folder_path)
