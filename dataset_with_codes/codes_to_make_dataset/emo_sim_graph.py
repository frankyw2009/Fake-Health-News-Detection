import torch
from torch_geometric.data import InMemoryDataset
import numpy as np
import re
# from transformers import BertTokenizer, BertModel
from torch_geometric.data import Data

# from transformers import AutoModelForSequenceClassification
# from transformers import AutoTokenizer
import torch.nn as nn
import torch.nn.functional as F

# 根据模型名称加载
# 第一次会在线加载模型，并且保存至用户子目录"\.cache\torch\transformers\"
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert = BertModel.from_pretrained('bert-base-uncased')
#
# emotion_tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
# emotion_model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
# emotion_model.classifier.out_proj = nn.Sequential()

def remove_special_chars(text):
    # 定义正则表达式模式，匹配特殊符号、换行符和网址
    pattern = r'[^a-zA-Z0-9\s@.,:!;\'\"]|https?://\S+|www\.|com|@\w+'
    return re.sub(pattern, '', text)

class NewsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(NewsDataset, self).__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])
        # For PyG<2.4:
        # self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['fake_news_detection.dataset']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    # the following lines shall be disabled after finish making the graph dataset
    def process(self):
        news_graph_list = []
        file_list = ['/content/final_dataset.npy']
        for file in file_list:
          print(file)
          graph_data = np.load(file,allow_pickle=True)

          for i in range(0, len(graph_data)):
              # i = 114 # at i = 114 the token exceeds 512
              print(f'the number of graph:{i}')
              if graph_data[i]:
                # print(i)

                root_news = graph_data[i]['root_news']
                news = remove_special_chars(root_news)

                inputs = tokenizer(news, return_tensors="pt")  # "pt"表示"pytorch"
                emotion_inputs = emotion_tokenizer(news, return_tensors='pt')
                if inputs['input_ids'].size(1) >510:
                  inputs = tokenizer(news, return_tensors="pt", max_length=510, padding="max_length", truncation=True)
                if emotion_inputs['input_ids'].size(1) > 510:
                  emotion_inputs = emotion_tokenizer(news, return_tensors="pt", max_length=510, padding="max_length", truncation=True)

                outputs = bert(**inputs)
                emotion_outputs = emotion_model(**emotion_inputs)
                root_news_feature = outputs.pooler_output
                print(f'root_news_feature shape:{root_news_feature.size()}')
                root_news_emotion_feature = emotion_outputs['logits']
                print(f'root_news_emotion_feature shape:{root_news_emotion_feature.size()}')
                # data[i]
                concatenated_tensor = torch.cat((root_news_feature, root_news_emotion_feature), dim=1)
                print(f'concat_tensor shape:{concatenated_tensor.size()}')
                news_mean_list = [concatenated_tensor]

                for j in range(10):
                  news_list = []
                  for k in range(10):
                    if 'tweet_id_' + str(k) in graph_data[i]['rts_' + str(j)]:
                      news_piece = graph_data[i]['rts_' + str(j)]['tweet_id_' + str(k)]
                      news_piece = remove_special_chars(news_piece)

                      inputs = tokenizer(news_piece, return_tensors="pt")  # "pt"表示"pytorch"
                      emotion_inputs = emotion_tokenizer(news_piece, return_tensors="pt")

                      if inputs['input_ids'].size(1) > 510:
                        inputs = tokenizer(news_piece, return_tensors="pt", max_length=510, padding="max_length", truncation=True)
                      if emotion_inputs['input_ids'].size(1) > 510:
                        emotion_inputs = emotion_tokenizer(news_piece, return_tensors="pt", max_length=510, padding="max_length", truncation=True)

                      outputs = bert(**inputs)
                      emotion_outputs = emotion_model(**emotion_inputs)
                      concatenated_outputs = torch.cat((outputs.pooler_output, emotion_outputs['logits']), dim=1)
                      news_list.append(concatenated_outputs)

                  for p in range(len(news_list)):
                    if p == 0:
                      news_mean = news_list[p]
                    else:
                      news_mean += news_list[p]

                  if len(news_list) > 0:
                    news_mean = news_mean / len(news_list)
                    news_mean_list.append(news_mean)

                # node_feature = torch.tensor(news_mean_list)
                node_feature = torch.tensor([item.detach().numpy() for item in news_mean_list]).squeeze(dim = 1)

                edge_index = []
                edge_attr = []
                for j in range(1, len(news_mean_list)): #one more edge
                  edge_index.append([0,j])
                  cosine_sim = F.cosine_similarity(news_mean_list[0], news_mean_list[j])
                  edge_attr.append(cosine_sim)

                print(f'edge_attr length: {len(edge_attr)}')

                edge_index = torch.tensor(edge_index, dtype=torch.long)  # .contiguous()
                edge_index = edge_index.transpose(1, 0)
                edge_attr = torch.Tensor(edge_attr)

                data = Data(x=node_feature, edge_index=edge_index, edge_attr = edge_attr, y=graph_data[i]['label'])
                news_graph_list.append(data)
                print('done')
                # breakpoint()
                # a = 1
        data, slices = self.collate(news_graph_list)
        torch.save((data, slices), self.processed_paths[0])