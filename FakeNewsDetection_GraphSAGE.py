import torch
from torch_geometric.datasets import UPFD
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import global_mean_pool
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix
import os

train_politifact_dataset = UPFD(root='graph_data/UPFD', name='politifact', feature='bert', split='test')
test_politifact_dataset = UPFD(root='graph_data/UPFD', name='politifact', feature='bert', split='train')
val_politifact_dataset = UPFD(root='graph_data/UPFD', name='politifact', feature='bert', split='val')

train_gossipcop_dataset = UPFD(root='graph_data/UPFD', name='gossipcop', feature='bert', split='train')
test_gossipcop_dataset = UPFD(root='graph_data/UPFD', name='gossipcop', feature='bert', split='test')
val_gossipcop_dataset = UPFD(root='graph_data/UPFD', name='gossipcop', feature='bert', split='val')

# Helper function for visualization.

# Visualization function for NX graph or PyTorch tensor
# def visualize(h, color, epoch=None, loss=None, accuracy=None):
#     plt.figure(figsize=(7,7))
#     plt.xticks([])
#     plt.yticks([])
#
#     if torch.is_tensor(h):
#         h = h.detach().cpu().numpy()
#         plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
#         if epoch is not None and loss is not None and accuracy['train'] is not None and accuracy['val'] is not None:
#             plt.xlabel((f'Epoch: {epoch}, Loss: {loss.item():.4f} \n'
#                        f'Training Accuracy: {accuracy["train"]*100:.2f}% \n'
#                        f' Validation Accuracy: {accuracy["val"]*100:.2f}%'),
#                        fontsize=16)
#     else:
#         nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), with_labels=False,
#                          node_color=color, cmap="Set2")
#     plt.savefig(fname='graph9.jpg')
#     plt.show()
#
# from torch_geometric.utils import to_networkx
#
# for i in range(62):
#     data = train_politifact_dataset[i]
#     if data.y == 0:
#         print(i)

# G = to_networkx(data, to_undirected=True)
# col = torch.randn(90)
# visualize(h=G, color=col)
# print(f'Number of nodes: {data.num_nodes}')
# print(f'Number of edges: {data.num_edges}')
# print(f'Average node degree: {(data.num_edges) / data.num_nodes:.2f}')
# print(f'Contains isolated nodes: {data.has_isolated_nodes()}')
# print(f'Contains self-loops: {data.has_self_loops()}')
# print(f'Is undirected: {data.is_undirected()}')


# train_dataset = train_politifact_dataset + train_gossipcop_dataset
# train_loader = DataLoader(train_politifact_dataset  + train_gossipcop_dataset, batch_size=64, shuffle=True)
train_loader = DataLoader(train_politifact_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_politifact_dataset + val_politifact_dataset, batch_size=64, shuffle=True)


# k = []
# for data in test_loader:
#     k.append(data.num_nodes)
#
# y = max(k)
# l=1


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

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = SAGEConv(768, 256)
        self.conv2 = SAGEConv(256, 128)
        # self.conv3 = GCNConv(192, 96)
        self.linear1 = Linear(128, 64)
        self.linear2 = Linear(64, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, batch):
        # print(batch)
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        # x = self.conv3(x, edge_index)

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


label = 0
onehot_label = convert_to_OneHot(label)

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()


def train():
    model.train()
    # 把模型设置为训练模式
    all_label = []
    for data in train_loader:
        # Iterate in batches over the training dataset.
        label_array = data.y.numpy().tolist()
        all_label = label_array[:]
        for i in range(len(label_array)):
            label_array[i] = convert_to_OneHot(label_array[i])
            if all_label[i] == 1:
                all_label[i] = 'cyan'
            else:
                all_label[i] = 'blueviolet'
        data.y = torch.Tensor(np.array(label_array))

        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        #  计算梯度
        optimizer.step()  # Update parameters based on gradients.
        #  根据上面计算的梯度更新参数
        optimizer.zero_grad()  # Clear gradients.
    #  清除梯度，为下一个批次的数据做准备，相当于从头开始
    return loss, out, all_label


def test(loader):
    model.eval()
    # 把模型设置为评估模式

    correct = 0
    #  初始化correct为0，表示预测对的个数
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        for i in range(data.num_graphs):
            # u = data.y
            # f = out[i][0]
            if out[i][0] > out[i][1]:
                pred = 0
            else:
                pred = 1
            if pred == data.y[i]:
                correct += 1

        # #  预测的输出值
        # pred = out.argmax(dim=1)  # Use the class with highest probability.
        # #  每个类别对应一个概率，概率最大的就是对应的预测值
        # correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    #  如果一样，就是True，也就是1，correct就+1
    # 准确率就是正确的/总的
    
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


def visualize(h, color, epoch=None, loss=None, train_accuracy=None, test_accuracy=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    if torch.is_tensor(h):
        h = h.detach().cpu().numpy()
        plt.scatter(h[:, 0], h[:, 1], s=140, c=color, cmap="Set2")
        if epoch is not None and loss is not None and train_accuracy is not None and test_accuracy is not None:
            plt.xlabel((f'Epoch: {epoch}, Loss: {loss.item():.4f} \n'
                        f'Training Accuracy: {train_accuracy[epoch-1] * 100:.2f}% \n'
                        f'Test Accuracy: {test_accuracy[epoch-1] * 100:.2f}%'),
                       fontsize=16)

    else:
        nx.draw_networkx(h, pos=nx.spring_layout(h, seed=42), with_labels=False,
                         node_color=color, cmap="Set2")
    plt.savefig(os.path.join('Figures', f'GraphSAGE_politifact_{epoch}_epoch_training_graph.jpg'),dpi=350)
    # plt.show()

x = []
y1 = []
y2 = []

for epoch in range(1, 101):
    loss, out, label_array = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    x.append(epoch)
    y1.append(train_acc)
    y2.append(test_acc)
    if epoch % 5 == 0:
        visualize(out, color=label_array, epoch=epoch, loss=loss, train_accuracy=y1, test_accuracy=y2)
        # plt.savefig(os.path.join('Figures', f'GraphSAGE_politifact_{epoch}_epoch_training_graph.jpg'))

plt.figure(figsize=(8, 5))

# x = ["Mon", "Tues", "Wed", "Thur", "Fri", "Sat", "Sun"]
# y = [20, 40, 35, 55, 42, 80, 50]

plt.plot(x, y1, label="train accuracy", c="g")
plt.plot(x, y2, label="test accuracy", c="b")

# 绘制坐标轴标签
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("Politifact Training Accuracy(SAGEConv)")
plt.legend()  # 加图例

# for x1, y1 in zip(x, y):
#     plt.text(x1, y1, str(y1), ha='center', va='bottom', fontsize=16)

plt.savefig('GraphSAGE_politifact_training_graph.jpg',dpi=350)
plt.show()

torch.save(model, 'GCN_politifact_model.pth')
