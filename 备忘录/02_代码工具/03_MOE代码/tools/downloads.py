# cd tools/
# python downloads.py --save_path='../pretrained_models'

# Mirror Website: https://hf-mirror.com/, all models are downloaded from here.

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default='../pretrained_models')
opt = parser.parse_args()

# model
models = ['DNABert2Embedder',
          'DNABertEmbedder',  # kmer = 3, 4, 5, 6
          'NucleotideTransformerEmbedder',
          'GENALMEmbedder',  # bigbird, bert
          'GROVEREmbedder']

for model in models:
    locals()[model] = os.path.join(opt.save_path, model)
    os.makedirs(locals()[model], exist_ok=True)



# DNABert2Embedder
files = ['bert_layers.py',
         'bert_padding.py',
         'config.json',
         'configuration_bert.py',
         'flash_attn_triton.py',
         'generation_config.json',
         'pytorch_model.bin',
         'tokenizer.json',
         'tokenizer_config.json']

for file in files:
    url = f'https://hf-mirror.com/zhihan1996/DNABERT-2-117M/resolve/main/{file}'
    os.system(f'wget {url} -P {DNABert2Embedder}/')



# DNABertEmbedder/3mer
files = ['config.json',
         'configuration_bert.py',
         'dnabert_layer.py',
         'pytorch_model.bin',
         'special_tokens_map.json',
         'tokenizer_config.json',
         'vocab.txt']
for file in files:
    url = f'https://hf-mirror.com/zhihan1996/DNA_bert_3/resolve/main/{file}'
    os.system(f'wget {url} -P {DNABertEmbedder}/3mer/')



# DNABertEmbedder/4mer
files = ['config.json',
         'configuration_bert.py',
         'dnabert_layer.py',
         'pytorch_model.bin',
         'special_tokens_map.json',
         'tokenizer_config.json',
         'vocab.txt']
for file in files:
    url = f'https://hf-mirror.com/zhihan1996/DNA_bert_4/resolve/main/{file}'
    os.system(f'wget {url} -P {DNABertEmbedder}/4mer/')



# DNABertEmbedder/5mer
files = ['config.json',
         'configuration_bert.py',
         'dnabert_layer.py',
         'pytorch_model.bin',
         'special_tokens_map.json',
         'tokenizer_config.json',
         'vocab.txt']
for file in files:
    url = f'https://hf-mirror.com/zhihan1996/DNA_bert_5/resolve/main/{file}'
    os.system(f'wget {url} -P {DNABertEmbedder}/5mer/')



# DNABertEmbedder/6mer
files = ['config.json',
         'configuration_bert.py',
         'dnabert_layer.py',
         'pytorch_model.bin',
         'special_tokens_map.json',
         'tokenizer_config.json',
         'vocab.txt']

for file in files:
    url = f'https://hf-mirror.com/zhihan1996/DNA_bert_6/resolve/main/{file}'
    os.system(f'wget {url} -P {DNABertEmbedder}/6mer')



# NucleotideTransformerEmbedder
files = ['config.json',
         'pytorch_model.bin',
         'special_tokens_map.json',
         'tf_model.h5',
         'tokenizer_config.json',
         'vocab.txt']

for file in files:
    url = f'https://hf-mirror.com/InstaDeepAI/nucleotide-transformer-500m-human-ref/resolve/main/{file}'
    os.system(f'wget {url} -P {NucleotideTransformerEmbedder}/')



# GENALMEmbedder/bigbird
files = ['config.json',
         'pytorch_model.bin',
         'special_tokens_map.json',
         'tokenizer.json',
         'tokenizer_config.json']

for file in files:
    url = f'https://hf-mirror.com/AIRI-Institute/gena-lm-bigbird-base-t2t/resolve/main/{file}'
    os.system(f'wget {url} -P {GENALMEmbedder}/bigbird/')



# GENALMEmbedder/bert
files = ['config.json',
         'modeling_bert.py',
         'pytorch_model.bin',
         'special_tokens_map.json',
         'tokenizer.json',
         'tokenizer_config.json']

for file in files:
    url = f'https://hf-mirror.com/AIRI-Institute/gena-lm-bert-base-t2t/resolve/main/{file}'
    os.system(f'wget {url} -P {GENALMEmbedder}/bert/')



# GROVEREmbedder
files = ['config.json',
         'pytorch_model.bin',
         'special_tokens_map.json',
         'tokenizer_config.json',
         'vocab.txt']

for file in files:
    url = f'https://zenodo.org/records/8373117/files/{file}'
    os.system(f'wget {url} -P {GROVEREmbedder}/')


if __name__ == '__main__':
    pass
