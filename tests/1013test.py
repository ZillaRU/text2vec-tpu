import sys
sys.path.append('/data/aigc/rzy_dev/airbox_other/text2vec')
from text2vec import SentenceModel
import numpy as np
import time 
import torch
import os
# # import sophon.sail as sail 
# from tpu_perf.infer import SGInfer

# class EngineOV:
    
#     def __init__(self, model_path="", batch=1,device_id=1) :
#         # 如果环境变量中没有设置device_id，则使用默认值
#         if "DEVICE_ID" in os.environ:
#             device_id = int(os.environ["DEVICE_ID"])
#             # print(">>>> device_id is in os.environ. and device_id = ",device_id)
#         self.model = SGInfer(model_path , batch=batch, devices=[device_id])
        
#     def __str__(self):
#         return "EngineOV: model_path={}, device_id={}".format(self.model_path,self.device_id)
    
#     def generate_randome_data(self):
#         info = self.model.get_input_info()
#         # {'latent.1': {'scale': 1.0, 'dtype': 0, 'shape': [2, 4, 128, 128]}, 't.1': {'scale': 1.0, 'dtype': 0, 'shape': [1]}, 'prompt_embeds.1': {'scale': 1.0, 'dtype': 0, 'shape': [2, 77, 2048]}, 'add_text_embeds.1': {'scale': 1.0, 'dtype': 0, 'shape': [2, 1280]}, 'add_time_ids.1': {'scale': 1.0, 'dtype': 0, 'shape': [2, 6]}}
#         res = {}
#         for k,v in info.items():
#             res[k] = generate_func(v["shape"], typemap[v['dtype']], 1)
#         return list(res.values())
    
        
#     def __call__(self, args):
#         start = time.time()
#         if isinstance(args, list):
#             values = args
#         elif isinstance(args, dict):
#             values = list(args.values())
#         else:
#             raise TypeError("args is not list or dict")
#             # print(values)
#         # print(time.time() - start)
#         start = time.time()
#         task_id = self.model.put(*values)
#         # print("put time : ",time.time() - start)
#         task_id, results, valid = self.model.get()
#         return results
    
#     def self_run(self):
#         # data = self.generate_randome_data()
#         data = [np.load("./test/testdata/L.npy")]
#         res  = self(data)
#         # for i in res:
#         #     # print(i.shape)
#         #     # print(np.mean(i))
#         #     np.save("res.npy",i)
#         return res
# class EngineOV:
    
#     def __init__(self, model_path="",output_names="",device_id=5) :
#         # 如果环境变量中没有设置device_id，则使用默认值
#         if "DEVICE_ID" in os.environ:
#             device_id = int(os.environ["DEVICE_ID"])
#             print(">>>> device_id is in os.environ. and device_id = ",device_id)
#         self.model_path = model_path
#         self.device_id = device_id
#         try:
#             self.model = sail.Engine(model_path, device_id, sail.IOMode.SYSIO)
#         except Exception as e:
#             print("load model error; please check model path and device status;")
#             print(">>>> model_path: ",model_path)
#             print(">>>> device_id: ",device_id)
#             print(">>>> sail.Engine error: ",e)
#             raise e
#         sail.set_print_flag(True)
#         self.graph_name = self.model.get_graph_names()[0]
#         self.input_name = self.model.get_input_names(self.graph_name)
#         self.output_name= self.model.get_output_names(self.graph_name)

#     def __str__(self):
#         return "EngineOV: model_path={}, device_id={}".format(self.model_path,self.device_id)
    
#     def __call__(self, args):
#         if isinstance(args, list):
#             values = args
#         elif isinstance(args, dict):
#             values = list(args.values())
#         else:
#             raise TypeError("args is not list or dict")
#         args = {}
#         for i in range(len(values)):
#             args[self.input_name[i]] = values[i]
#         output = self.model.process(self.graph_name, args)
#         res = []

#         for name in self.output_name:
#             res.append(output[name])
#         return res

# bert = EngineOV('/data/aigc/rzy_dev/airbox_other/text2vec/model_file/bert_bge-4_512.bmodel',
#                         device_id=0)
# input_ids = np.zeros((4, 512), dtype=np.float16)
# attention_mask = np.zeros((4, 512), dtype=np.float16)
# token_type_ids = np.zeros((4, 512), dtype=np.float16)
# print(id(input_ids), id(attention_mask), id(token_type_ids))
# # import pdb;pdb.set_trace()
# model_output = bert({'input_ids.1': input_ids,
#                           'attention_mask.1': attention_mask,
#                           'token_type_ids.1': token_type_ids}) 

sentences = ['在实现pytorch_model.bin模型文件之前，我们首先需要明确一些概念。pytorch_model.bin是PyTorch中保存训练好的模型权重的文件格式。通常，在模型训练过程中，我们会使用PyTorch的torch.nn.Module来定义模型的结构，并使用torch.optim来定义优化算法。在训练完成后，我们可以通过调用save_state_dict方法将模型权重保存为pytorch_model.bin文件',
             '我们可以通过使用torch.save方法将模型保存为pytorch_model.bin文件',
             '我们可以通过使用torch.load方法加载模型权重',
             '我们可以通过使用torch.load_state_dict方法加载模型权重']

# /home/aigc/.local/lib/python3.6/site-packages/transformers/models/bert/modeling_bert.py

model = SentenceModel('shibing624/text2vec-bge-large-chinese')
embeddings = model.encode(sentences)
print(embeddings)

model = SentenceModel('shibing624/text2vec-bge-large-chinese', device='tpu')
embeddings_tpu = model.encode_tpu(sentences)
print(embeddings_tpu)

print(np.mean(np.abs(embeddings_tpu-embeddings)))