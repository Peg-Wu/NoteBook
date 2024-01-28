# cd tools/
# python file_process.py --fasta_path='...' --csv_path='...' --need_convert_U2T=True/False

# fasta file: >sequence_name|label|
# Note: The sequence_name should not contain the character '|'

# If you want to process the original data in the repository, run the following command:
# cd tools/
# python file_process.py --fasta_path="../data/2OM_Train/fasta" --csv_path="../data/2OM_Train/csv" --need_convert_U2T=True
# python file_process.py --fasta_path="../data/2OM_Test/fasta" --csv_path="../data/2OM_Test/csv" --need_convert_U2T=True

from Bio import SeqIO
import csv
import os
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fasta_path', type=str)
parser.add_argument('--csv_path', type=str)
parser.add_argument('--need_convert_U2T', type=bool, default=False,
                    help="If your sequences contain 'U', you should convert 'U' to 'T', "
                         "please set --need_convert_U2T=True")
opt = parser.parse_args()

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

def main(fasta_file_path, csv_file_path, need_convert_U2T):
    """
    该函数用于数据处理，具体包含fasta转csv文件，以及标签(label)的提取
    :param fasta_file_path: fasta文件路径
    :param csv_file_path: csv文件路径
    :param need_convert_U2T: 'U'是否转换为'T'
    :return: 处理后的csv文件
    """
    os.makedirs(csv_file_path, exist_ok=True)

    # fasta转csv
    fasta_files = [os.path.join(fasta_file_path, each) for each in os.listdir(fasta_file_path)]
    csv_files = [os.path.join(csv_file_path, each) for each in suffix2csv(os.listdir(fasta_file_path))]
    fasta_csv_pairs = zip(fasta_files, csv_files)
    for each in fasta_csv_pairs:
        fa2csv(each[0], each[1])

    # 将U转换成T，并提取label
    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        if need_convert_U2T:
            data['seq'] = data['seq'].apply(lambda x: x.replace('U', 'T'))
        labels = [each.split('|')[1] for each in data.id.values.tolist()]
        data['label'] = np.array(labels)
        data.to_csv(csv_file, index=False)


if __name__ == '__main__':
    main(opt.fasta_path, opt.csv_path, opt.need_convert_U2T)
    print('Finish!')
