# import torch
# from torch.utils.data import Dataset, DataLoader

# # 예제 데이터셋 사용 예시
# class CustomDataset(Dataset):
#     def __init__(self, num_node, num_features):
#         self.num_node = num_node
#         self.num_features = num_features
        
#     def __len__(self):
#         return len(self.num_node)
    
#     def __getitem__(self, index):
#         graph = torch.randn((self.num_node[index],self.num_features))
#         label = torch.randint(0,1,(1,))
#         return graph, label
    

# def collate_fn(batch):
#     # batch는 (sequence, label)의 리스트입니다.
#     sequences = [item[0] for item in batch]
#     labels = [item[1] for item in batch]
    
#     return sequences, labels


# if __name__ == '__main__':
#     import numpy as np
#     src_seq_len = np.arange(10,100)
#     trg_seq_len = np.arange(100,10, -1)
#     dataset = CustomDataset(src_seq_len, trg_seq_len)
#     dataloader = DataLoader(dataset=dataset,
#                             batch_size=16,
#                             collate_fn=collate_fn)
    
#     src, trg = next(iter(dataloader))

#     print(src.shape)
#     print(trg.shape)


import torch
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, num_graphs, max_nodes, num_features):
        self.num_graphs = num_graphs
        self.max_nodes = max_nodes
        self.num_features = num_features
        
    def __len__(self):
        return self.num_graphs
    
    def __getitem__(self, index):
        num_nodes = torch.randint(1, self.max_nodes + 1, (1,)).item()
        graph = torch.randn((num_nodes, self.num_features))
        
        # Create a random adjacency matrix for the graph
        adj_matrix = torch.randint(0, 2, (num_nodes, num_nodes)).float()
        adj_matrix = (adj_matrix + adj_matrix.t()) / 2  # Symmetrize the matrix
        adj_matrix.fill_diagonal_(1)  # Ensure self-loops
        
        label = torch.randint(0, 2, (1,))  # Binary label (0 or 1)
        
        return graph, adj_matrix, label

def collate_fn(batch):
    graphs = [item[0] for item in batch]
    adj_matrices = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    
    return graphs, adj_matrices, labels

if __name__ == '__main__':
    dataset = CustomDataset(num_graphs=100, max_nodes=30, num_features=5)
    dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

    graphs, adj_matrices, labels = next(iter(dataloader))

    print(graphs)