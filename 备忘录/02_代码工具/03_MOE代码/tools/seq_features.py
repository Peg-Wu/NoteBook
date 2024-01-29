import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import datetime
from collections import Counter

def fft_features(lis, max_len=41):
    print('[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ']', end='\t')
    print('Extract spectrum features...')
    res = []
    for item in lis:
        missing = 0
        element = list(set(list(item)))
        ele_basic = ['A', 'C', 'G', 'T']
        if len(element) < 4:
            missing = 1
            missing_index = []
            for ele in ele_basic:
                if ele not in element:
                    missing_index.append(ele_basic.index(ele))
        item = np.array(list(item), str)
        item = LabelEncoder().fit_transform(item).reshape(-1, 1)
        item = OneHotEncoder().fit_transform(item).todense()
        item = item.transpose()
        if missing:
            l = item.shape[1]
            item = list(np.array(item))
            for _ in missing_index:
                item.insert(_, np.zeros(l, np.int8))
            item = np.array(item)
        length_original = item.shape[1]
        if length_original <= max_len:
            padding = np.zeros((4, max_len - item.shape[1]))
            item = np.concatenate((item, padding), axis=1)
        else:
            item = item[:, :max_len]
            length_original = max_len
        length = item.shape[1]
        FFT = np.fft.fft(item, axis=1) / length_original
        FFT = FFT[:, 1:length // 2 + 1] * 2
        FFT[:, -1] = FFT[:, -1] / 2 if length % 2 == 0 else FFT[:, -1]
        FFT_real = np.real(FFT)
        FFT_imag = np.imag(FFT)
        item = np.array([FFT_real, FFT_imag])
        res.append(item)
    return np.array(res).astype(np.float32).swapaxes(1, 3)

def K_mer_features(lis, k=3):
    print('[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ']', end='\t')
    print('Extract ' + str(k) + '-mer features...')
    res = []
    ele_base = ['A', 'C', 'G', 'T']
    element = ['A', 'C', 'G', 'T']
    k_ = k - 1
    while k_:
        k_ -= 1
        Num = len(ele_base)
        for i in range(Num):
            for ele in element:
                ele_base.append(ele_base[i] + ele)
        ele_base = ele_base[Num:]
    for item in lis:
        seq_lis = []
        for i in range(len(item) - k + 1):
            seq_lis.append(item[i:i + k])
        Length = len(seq_lis)
        seq_lis = Counter(seq_lis)
        feature = []
        for ele in ele_base:
            feature.append(seq_lis[ele] / Length) if ele in seq_lis else feature.append(0)
        res.append(feature)
    return (np.array(res)).astype(np.float32)

if __name__ == '__main__':
    pass
