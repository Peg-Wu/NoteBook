import os
import re
import h5py
import time
import pickle

class Conductor:
    """Operate h5 files."""
    def __init__(self, file_path):
        self.file_path = file_path

    # Display the content of a specific dataset.
    def show_data(self, dataset_path):
        with h5py.File(self.file_path, 'r') as f:
            dataset = f[dataset_path][:]
        return dataset

    @property
    def get_structure_all(self):
        with h5py.File(self.file_path, 'r') as f:
            structure_all = []
            f.visit(lambda x: structure_all.append(x))
        return structure_all

    @property
    def get_structure_datasets(self):
        structure_all = self.get_structure_all
        with h5py.File(self.file_path, 'r') as f:
            structure_datasets = [path for path in structure_all if isinstance(f[path], h5py.Dataset)]
        return structure_datasets

    @property
    def get_structure_groups(self):
        structure_all = self.get_structure_all
        with h5py.File(self.file_path, 'r') as f:
            structure_groups = [path for path in structure_all if isinstance(f[path], h5py.Group)]
        return structure_groups

    def test(self):
        """
        Function:
        1. Test whether len(structure_all) == len(structure_datasets) + len(structure_groups) or not.
        2. For examining if existing other data format.
        """
        print(len(self.get_structure_all) == len(self.get_structure_datasets) + len(self.get_structure_groups))


def str_process(original_string):
    """
    Function: Process Strings
    1. Remove .and-after substring
    2. String.lower()
    """
    assert type(original_string) in (str, list, tuple)

    def operation(original_string):
        return re.sub(b'\..*$', b'', original_string).lower()

    if type(original_string) == str:
        return operation(original_string)
    else:
        return [operation(each) for each in original_string]


if __name__ == '__main__':
    start_time = time.time()
    print("----------开始处理----------")

    root = "../process/data"
    root_list = [os.path.join(root, each) for each in os.listdir(root)]
    print("样本数: %d" % len(root_list))

    h5_files = []
    for sample in root_list:
        sample_files = [os.path.join(sample, each) for each in os.listdir(sample)]
        h5_file = [file for file in sample_files if file.endswith('expression.h5')]
        h5_files.extend(h5_file)

    print(f"待处理的h5文件数: {len(h5_files)}")

    # --------------------开始处理每一个h5文件-------------------- #
    all_genes = []
    for h5 in h5_files:
        conductor = Conductor(h5)
        gene_name_location = 'matrix/features/name'
        assert gene_name_location in conductor.get_structure_datasets
        all_str = str_process(conductor.show_data(gene_name_location).tolist())
        # 去除all_str中的motif序列 (类似于: b'agcttgcatt-12')
        gene_name_list = [each_str for each_str in all_str if re.match(br'[agct]+[-]\d+', each_str) is None]
        all_genes.extend(gene_name_list)

    print(f"基因数: {len(all_genes)}")
    all_genes = list(set(all_genes))
    print(f"去重后基因数: {len(all_genes)}")

    print("----------处理完成----------")
    end_time = time.time()
    print(f"处理时间: {end_time - start_time}s")

    # 保存为.pickle
    save_path = './Gene_names.pkl'

    with open(save_path, 'wb') as file:
        pickle.dump(all_genes, file)

    print(f'文件已保存! 程序结束! 保存路径: {save_path}')