"""
Many parts of the fake health news detection program are interchangeable.
For example, there are two different model structures using GCN and SAGE.
On the other hand, the dataset can be HNDataset-BASE and HNDataset-Emotion.
Sometimes, UPFD dataset also get used to be train dataset with HNDataset as test dataset.

As a result, annotated alternatives codes to different sections of the model
are included in this program file.
"""
import torch
from torch_geometric.datasets import UPFD
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix
import os
from base_graph import NewsDataset
# from  similarity_graph import NewsDataset
# from emo_graph import NewsDataset
# from emo_sim_graph import NewsDataset
from sklearn.model_selection import train_test_split


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
politifact_dataset = UPFD(root='graph_data/UPFD', name='politifact', feature='bert', split='test')
NewsData = NewsDataset(root='NewsData_BASE/')
# NewsData = NewsDataset(root='NewsData_BASE_emo/')

# a = NewsData[0]

graphs_list = []
labels_list = []

for i in range(len(NewsData)):
    graphs_list.append(NewsData[i])
    labels_list.append(NewsData[i].y)

# Assuming `graphs` is a list of graph objects and `labels` is a list of corresponding labels
X_train, X_test, y_train, y_test = train_test_split(
    graphs_list, labels_list, test_size=0.4, random_state=42
)
a = 1
# X is the graph, Y is the labels


# Visualization function for NX graph or PyTorch tensor
def visualize(h, color, epoch=None, loss=None, accuracy=None):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None and accuracy['train'] is not None and accuracy['val'] is not None:
            plt.xlabel((f'Epoch: {epoch}, Loss: {loss.item():.4f} \n'
                       f'Training Accuracy: {accuracy["train"]*100:.2f}% \n'
                       f' Validation Accuracy: {accuracy["val"]*100:.2f}%'),
                       fontsize=16)
    else:
        nx.draw_networkx(h, pos=nx.spring_layout(h), with_labels=False,
                         node_color=color, cmap="Set2")
    plt.savefig(os.path.join("Figures", "base_3"), dpi=350)
    plt.show()


from torch_geometric.utils import to_networkx
#
# data = NewsData[2]
# #
# G = to_networkx(data, to_undirected=True)
# col = torch.randn(data.num_nodes)
# visualize(h=G, color=col)

# print(f'Number of nodes: {data.num_nodes}')
# print(f'Number of edges: {data.num_edges}')
# print(f'Average node degree: {(data.num_edges) / data.num_nodes:.2f}')
# print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
# print(f'Contains self-loops: {data.has_self_loops()}')
# print(f'Is undirected: {data.is_undirected()}')


# train_dataset = train_politifact_dataset + train_gossipcop_dataset
# train_loader = DataLoader(train_politifact_dataset  + train_gossipcop_dataset, batch_size=64, shuffle=True)
# train_politifact_dataset = UPFD(root='graph_data/UPFD', name='politifact', feature='bert', split='test')
# test_loader = DataLoader(train_politifact_dataset, batch_size=64, shuffle=True)
train_loader = DataLoader(X_train, batch_size=64, shuffle=True)
# train_loader = DataLoader(politifact_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(X_test, batch_size=16, shuffle=False)


# for step, data in enumerate(train_loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     print(data)
#     print()
#
# for step, data in enumerate(test_loader):
#     print(f'Step {step + 1}:')
#     print('=======')
#     print(f'Number of graphs in the current batch: {data.num_graphs}')
#     print(data)
#     print()

# GCN model
# class GCN(torch.nn.Module):
#     def __init__(self):
#         super(GCN, self).__init__()
#         torch.manual_seed(12345)
#         self.conv1 = GCNConv(768, 256)
#         self.conv2 = GCNConv(256, 128)
#         # self.conv3 = GCNConv(192, 96)
#         self.linear1 = Linear(128, 64)
#         self.linear2 = Linear(64, 2)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x, edge_index, batch):
#         # print(batch)
#         # 1. Obtain node embeddings
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = self.conv2(x, edge_index)
#         x = x.relu()
#         # x = self.conv3(x, edge_index)
#
#         # 2. Readout layer
#         x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
#
#         # 3. Apply a final classifier
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.linear1(x)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.linear2(x)
#         x = self.softmax(x)
#
#         return x

# SAGE model
# class SAGE(torch.nn.Module):
#     def __init__(self):
#         super(SAGE, self).__init__()
#         torch.manual_seed(12345)
#         self.conv1 = SAGEConv(768, 256)
#         self.conv2 = SAGEConv(256, 128)
#         # self.conv3 = GCNConv(192, 96)
#         self.linear1 = Linear(128, 64)
#         self.linear2 = Linear(64, 2)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x, edge_index, batch):
#         # print(batch)
#         # 1. Obtain node embeddings
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = self.conv2(x, edge_index)
#         x = x.relu()
#         # x = self.conv3(x, edge_index)
#
#         # 2. Readout layer
#         x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
#
#         # 3. Apply a final classifier
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.linear1(x)
#         x = F.dropout(x, p=0.5, training=self.training)
#         x = self.linear2(x)
#         x = self.softmax(x)
#
#         return x

# GAT model
class GAT(torch.nn.Module):
    def __init__(self):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        # self.conv1 = SAGEConv(1536, 768)
        self.conv2 = GCNConv(768, 256)
        self.conv3 = GCNConv(256, 128)
        # self.conv3 = GCNConv(192, 96)
        self.linear1 = Linear(128, 64)
        self.linear2 = Linear(64, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, batch):
        # print(batch)
        # 1. Obtain node embeddings
        # x = self.conv1(x, edge_index)
        # x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear1(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.linear2(x)
        x = self.softmax(x)

        return x


label_a = [['fake'], ['true']]
encoder = OneHotEncoder(sparse=False)
data_array = np.array(label_a)
onehot_encoded = encoder.fit_transform(data_array)


def convert_to_OneHot(x):
    if x == 0:
        return onehot_encoded[0]
    else:
        return onehot_encoded[1]


# label = 0
# onehot_label = convert_to_OneHot(label)

model = GAT().to(device)
# have to be modified to:
# model = GCN().to(device)
# model = SAGE().to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) # learning rate is 0.0001
criterion = torch.nn.CrossEntropyLoss().to(device)


def train():
    model.train()
    # 把模型设置为训练模式
    all_label = []
    t = 0
    for data in train_loader:
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        # data.edge_attr = data.edge_attr.to(device)
        data.batch = data.batch.to(device)
        data.y = data.y.to(device)
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        if t == 0:
            temp_store = out[:64]
            label_array = data.y.cpu().numpy().tolist()
            all_label = label_array[:64]
            for i in range(len(all_label)):
                # label_array[i] = convert_to_OneHot(label_array[i])
                if all_label[i] == 1:
                    all_label[i] = 'cyan'
                else:
                    all_label[i] = 'blueviolet'
            # data.y = torch.Tensor(np.array(label_array))

        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        #  计算梯度
        optimizer.step()  # Update parameters based on gradients.
        #  根据上面计算的梯度更新参数
        optimizer.zero_grad()  # Clear gradients.
        t += 1
    #  清除梯度，为下一个批次的数据做准备，相当于从头开始
    return loss, temp_store, all_label


def test(loader):
    model.eval()
    # evaluate model

    correct = 0
    # correct number of predictions
    # y_true = []
    # y_pre = []
    for data in loader:  # Iterate in batches over the training/test dataset.
        data.x = data.x.to(device)
        data.edge_index = data.edge_index.to(device)
        # data.edge_attr = data.edge_attr.to(device)
        data.batch = data.batch.to(device)
        # y_true.append(data.y)

        out = model(data.x, data.edge_index, data.batch)
        for i in range(data.num_graphs):
            # u = data.y
            # f = out[i][0]
            if out[i][0] > out[i][1]:
                pred = 0
            else:
                pred = 1
            # y_pre.append(pred)
            if pred == data.y[i]:
                correct += 1

        # #  预测的输出值
        # pred = out.argmax(dim=1)  # Use the class with highest probability.
        # #  每个类别对应一个概率，概率最大的就是对应的预测值
        # correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    #  如果一样，就是True，也就是1，correct就+1
    # 准确率就是正确的/总的

    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


# def visualize(h, color, epoch=None, loss=None, train_accuracy=None, test_accuracy=None):
#     plt.figure(figsize=(7, 7))
#     plt.xticks([])
#     plt.yticks([])
#
#     if torch.is_tensor(h):
#         h = h.detach().cpu().numpy()
#         plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
#         if epoch is not None and loss is not None and train_accuracy is not None and test_accuracy is not None:
#             plt.xlabel((f'Epoch: {epoch}, Loss: {loss.item():.4f} \n'
#                         f'Training Accuracy: {train_accuracy[epoch-1] * 100:.2f}% \n'
#                         f'Test Accuracy: {test_accuracy[epoch-1] * 100:.2f}%'),
#                        fontsize=16)
#
#     else:
#         nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), with_labels=False,
#                          node_color=color, cmap="Set2")
#     # plt.savefig(os.path.join('Figures', f'politifact_{epoch}_epoch_training_graph.jpg'), dpi=300)
#     plt.show()

x = []
y1 = []
y2 = []

for epoch in range(0, 250):
    loss, out, label_array = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    x.append(epoch)
    y1.append(train_acc)
    y2.append(test_acc)

    # if epoch % 5 == 0:
    #     visualize(out, color=label_array, epoch=epoch, loss=loss, train_accuracy=y1)

# with open('GCN_politifact_base.txt', 'w') as f:
#     for item in y2:
#         f.write(str(item) + '\n')

plt.figure(figsize=(8, 5))

plt.plot(x, y1, label="train accuracy", c="g")
plt.plot(x, y2, label="test accuracy", c="b")

# draw graph
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("GAT Politifact&Base Accuracy")
plt.legend()  # add legend

# for x1, y1 in zip(x, y):
#     plt.text(x1, y1, str(y1), ha='center', va='bottom', fontsize=16)

plt.savefig(os.path.join('Figures','GAT_Politifact_base.jpg'), dpi=350)
plt.show()
#
torch.save(model, os.path.join('output','GAT_Politifact_base.pth'))
