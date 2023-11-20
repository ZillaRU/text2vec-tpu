import sys
sys.path.append('/data/aigc/rzy_dev/airbox_other/text2vec')
from text2vec import SentenceModel
import numpy as np
import time 
import torch
import os

sentences = ['在实现pytorch_model.bin模型文件之前，我们首先需要明确一些概念。pytorch_model.bin是PyTorch中保存训练好的模型权重的文件格式。通常，在模型训练过程中，我们会使用PyTorch的torch.nn.Module来定义模型的结构，并使用torch.optim来定义优化算法。在训练完成后，我们可以通过调用save_state_dict方法将模型权重保存为pytorch_model.bin文件',
             '我们可以通过使用torch.save方法将模型保存为pytorch_model.bin文件',
             '我们可以通过使用torch.load方法加载模型权重',
             '我们可以通过使用torch.load_state_dict方法加载模型权重']

# /home/aigc/.local/lib/python3.6/site-packages/transformers/models/bert/modeling_bert.py
model = SentenceModel('shibing624/text2vec-base-chinese-paraphrase')
embeddings = model.encode(sentences)
print(embeddings)

model = SentenceModel('shibing624/text2vec-bge-large-chinese', device='tpu')
embeddings_tpu = model.encode_tpu(sentences)
print(embeddings_tpu)

print(np.mean(np.abs(embeddings_tpu-embeddings)))