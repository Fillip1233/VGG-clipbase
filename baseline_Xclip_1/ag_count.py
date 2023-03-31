import os

def count_dirs_files(path):
    """
    统计指定路径下所有子文件夹的文件数量，并返回文件数量分布字典和文件总数
    """
    file_counts = {}
    file_count_sum = 0
    for root, dirs, files in os.walk(path):
        file_count = len(files)
        if file_count not in file_counts:
            file_counts[file_count] = 1
        else:
            file_counts[file_count] += 1
        file_count_sum += file_count
    return file_counts, file_count_sum

if __name__ == '__main__':
    path = '/mnt/cephfs/dataset/zhenjie/agtraindata/frames'
    file_counts, file_count_sum = count_dirs_files(path)
    print(f"文件数量分布：{file_counts}")
    print(f"子文件夹总数：{sum(file_counts.values())}")
    print(f"文件总数：{file_count_sum}")
