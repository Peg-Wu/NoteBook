from Bio import SeqIO
import csv
import os
import pandas as pd
import numpy as np

def fa2csv(fasta_file, csv_file):
    """
    该函数用于将fasta文件转换成csv文件
    :param fasta_file: fasta文件路径
    :param csv_file: csv文件路径
    :return: csv文件
    """
    with open(csv_file, 'w', newline='') as csv_file:
        fieldnames = ['id', 'seq']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in SeqIO.parse(fasta_file, "fasta"):
            writer.writerow({'id': record.id, 'seq': str(record.seq)})

def suffix2csv(fasta_file_list):
    """
    该函数用于将后缀为.fasta的文件'名称'转换成后缀为.csv
    :param fasta_file_list: fasta文件列表
    :return: csv文件列表
    """
    assert fasta_file_list[0].endswith(".fasta")
    return [os.path.splitext(file_name)[0] + ".csv" for file_name in fasta_file_list]

def process(fasta_file_path, csv_file_path):
    """
    主程序：该函数用于数据处理，具体包含fasta转csv文件，以及标签(label)的提取
    :param fasta_file_path: fasta文件路径
    :param csv_file_path: csv文件路径
    :return: 处理后的csv文件
    """
    os.makedirs(csv_file_path, exist_ok=True)

    # fasta转csv
    fasta_files = [os.path.join(fasta_file_path, each) for each in os.listdir(fasta_file_path)]
    csv_files = [os.path.join(csv_file_path, each) for each in suffix2csv(os.listdir(fasta_file_path))]
    fasta_csv_pairs = zip(fasta_files, csv_files)
    for each in fasta_csv_pairs:
        fa2csv(each[0], each[1])

    # 提取label
    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        labels = [each.split('|')[1] for each in data.id.values.tolist()]
        data['label'] = np.array(labels)
        data.to_csv(csv_file, index=False)


if __name__ == '__main__':
    # 文件路径
    fasta_file_path_train = "../data/2OM_Train/fasta"
    csv_file_path_train = "../data/2OM_Train/csv"

    fasta_file_path_test = "../data/2OM_Test/fasta"
    csv_file_path_test = "../data/2OM_Test/csv"

    # 预处理
    process(fasta_file_path_train, csv_file_path_train)
    process(fasta_file_path_test, csv_file_path_test)

    print("Finish！")
