import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
from .h5 import Conductor
import os
import warnings
warnings.filterwarnings("ignore")

def get_X_from_h5(h5_file, embed_type: [str, list[str]], bases: str, train=True) -> list:
    assert isinstance(embed_type, (str, list))
    embed_types = ['OneHotEmbedder',
                   'DNABert2Embedder',
                   'DNABertEmbedder/3mer',
                   'DNABertEmbedder/4mer',
                   'DNABertEmbedder/5mer',
                   'DNABertEmbedder/6mer',
                   'NucleotideTransformerEmbedder',
                   'GENALMEmbedder/bigbird',
                   'GENALMEmbedder/bert',
                   'GROVEREmbedder']

    if isinstance(embed_type, str):
        assert embed_type in embed_types
    else:
        for embed in embed_type:
            assert isinstance(embed, str) and embed in embed_types

    for base in bases:
        assert base in 'ACGU'

    if train:
        sub = 'train'
    else:
        sub = 'test'

    conductor = Conductor(h5_file)
    if type(embed_type) == str:
        targets_candidate = [path for path in conductor.get_structure_datasets
                             if (sub in path) and (embed_type in path)]
        X = []
        for base in bases:
            targets = [path for path in targets_candidate if (base + '_X') in path]
            assert len(targets) == 1
            X.append(torch.from_numpy(conductor.show_data(targets[0])).float())
        assert len(X) >= 1
        return X
    else:
        targets_candidate = [path for path in conductor.get_structure_datasets
                             for embed in embed_type
                             if (sub in path) and (embed in path)]
        X = []
        for base in bases:
            specific_base_of_all_embeds = []
            for embed in embed_type:
                targets = [path for path in targets_candidate if (embed in path) and ((base + '_X') in path)]
                assert len(targets) == 1
                specific_base_of_all_embeds.append(torch.from_numpy(conductor.show_data(targets[0])).float())
            assert len(specific_base_of_all_embeds) >= 1
            X.append(torch.cat(specific_base_of_all_embeds, dim=-1))
        assert len(X) >= 1
        return X

def get_y_from_csv(data_root, bases: str, train=True) -> list:
    for base in bases:
        assert base in 'ACGU'

    if train:
        sub = '2OM_Train/csv'
        sub2 = 'train'
    else:
        sub = '2OM_Test/csv'
        sub2 = 'test'

    csv_root = os.path.join(data_root, sub)

    y = []
    for base in bases:
        data = pd.read_csv(os.path.join(csv_root, f'{base}2OM_{sub2}.csv'))
        y.append(torch.from_numpy(data.label.values).long())
    return y

def get_datasets(data_root, h5_file, embed_type: str, bases: str, train=True) -> list:
    X = get_X_from_h5(h5_file, embed_type, bases, train)
    y = get_y_from_csv(data_root, bases, train)
    datasets = []
    for each_x, each_y in zip(X, y):
        datasets.append(TensorDataset(each_x, each_y))
    return datasets

def split_dataset_2parts(dataset, ratio_part1=0.7):
    part1_len = int(len(dataset) * ratio_part1)
    part2_len = len(dataset) - part1_len
    part1, part2 = random_split(dataset, [part1_len, part2_len])
    return part1, part2

def merge_datasets(datasets: list) -> torch.utils.data.Dataset:
    dataset = datasets[0]
    for i in range(len(datasets)):
        if i == 0:
            continue
        else:
            dataset += datasets[i]
    return dataset

def calc_embed_dims(embed_type: [str, list[str]]):
    assert isinstance(embed_type, (str, list))
    dims = {'OneHotEmbedder': 4,
            'DNABert2Embedder': 768,
            'DNABertEmbedder/3mer': 768,
            'DNABertEmbedder/4mer': 768,
            'DNABertEmbedder/5mer': 768,
            'DNABertEmbedder/6mer': 768,
            'NucleotideTransformerEmbedder': 1280,
            'GENALMEmbedder/bigbird': 768,
            'GENALMEmbedder/bert': 768,
            'GROVEREmbedder': 768}
    if type(embed_type) == str:
        return dims[embed_type]
    else:
        return sum([dims[each] for each in embed_type])


if __name__ == '__main__':
    pass
