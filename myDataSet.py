import pandas as pd
import dgl
from dgl.data import DGLDataset
import torch
import numpy as np
import os

class myERDataset(DGLDataset):
    def __init__(self, days: list, length: int, read_path:str, save_dir: str):
        """
        arg:
        days: list 需要提取的天数 如 [7, 14, 15, 16]
        length: int 是指之前模拟传播了多少次,用于定位下标
        read_path: str rawdata文件夹
        save_path: str data存在哪里
        """
        self.days = days
        self.length = length  
        self.read_path = read_path
        super().__init__(name='ER_Graph', save_dir=save_dir)

    def process(self):
        edges_data = pd.read_csv(self.read_path + 'edges.csv')
        
        node_features = []
        node_labels = []
        for i in range(self.length):
            label_data = pd.read_csv(self.read_path + str(i) + '/nodes' + str(1) +'.csv')
            label = label_data['state'].to_numpy()
            for j in self.days:
                nodes_data = pd.read_csv(self.read_path + str(i) + '/nodes' + str(j) +'.csv')
                node_features.append(nodes_data['state'].to_numpy()) # 某天的状态 # append 一行
                node_labels.append(label)

        # edge_features = torch.from_numpy(edges_data['weight'].to_numpy())
        edges_src = torch.from_numpy(edges_data['src'].to_numpy())
        edges_dst = torch.from_numpy(edges_data['dst'].to_numpy())
        n_nodes = node_labels[0].shape[0]
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=n_nodes)
        node_features = np.array(node_features).reshape(-1, n_nodes).T
        node_labels = np.array(node_labels).reshape(-1, n_nodes).T
        self.graph.ndata['feat'] = torch.from_numpy(node_features)
        self.graph.ndata['label'] = torch.from_numpy(node_labels)
        # self.graph.edata['weight'] = edge_features

        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.

        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self):
        return self.graph

    def __len__(self):
        return len(self.days) * self.length
        
    def save(self):
        pass

if __name__ == "__main__":
    save_path = './dataset/test'
    os.makedirs(save_path)
    myERDataset([7, 10, 13, 16], 10, './rawdata/er/', save_path).save()