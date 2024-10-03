import os

# 数据文件夹路径
data_dir = '../data'

# 创建一个空的列表，用于存储png文件路径和标签
file_list = []

# 遍历数据文件夹
for root, dirs, files in os.walk(data_dir):
    # 遍历文件
    for file in files:
        # 如果文件是png文件
        if file.endswith('.png'):
            # 获取文件的相对路径
            file_path = os.path.relpath(os.path.join(root, file), data_dir)
            # 获取文件所在文件夹的名称（frontal或profile）
            folder_name = os.path.basename(os.path.dirname(file_path))
            # 设置标签
            if folder_name == 'frontal':
                label = '4'
            elif folder_name == 'profile':
                label = '1'
            # 将路径中的反斜杠替换为正斜杠
            file_path = file_path.replace("\\", "/")
            # 添加到文件列表中，加上"../data/"前缀
            file_list.append(f"../data/{file_path} {label}\n")

# 写入list.txt文件
with open('../my_CR-GAN/list.txt', 'w') as f:
    f.writelines(file_list)
