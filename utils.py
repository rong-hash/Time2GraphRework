import numpy as np
import torch, os, random
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from dgl import graph
import dgl

def write_log(log_filename, message):
    with open(log_filename, 'a') as log:
        log.write(message)

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    dgl.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    

def read_dataset(dataset_name):
    data = np.load('/data/chenzy/ucr/%s.npz' % dataset_name)
    return (data[file] for file in data.files)

def matrix2index(matrix):
    return matrix.nonzero().T

# class GraphDataset(Dataset):
#     def __init__(self, node_features, edge_matrices, labels, args):
#         super(GraphDataset, self).__init__()
#         node_features, edge_matrices, labels = map(lambda x: torch.tensor(x).to(args.device), 
#                                                    (node_features, edge_matrices, labels))
#         self.data = []
#         for node_feature, edge_matrix, label in zip(node_features, edge_matrices, labels):
#             self.data.append(Data(x=node_feature.float(), 
#                                   edge_index=matrix2index(edge_matrix), 
#                                   y=label, num_nodes=node_feature.shape[0]))
#     def __len__(self):
#         return len(self.data)
#     def __getitem__(self, idx):
#         return self.data[idx]

class GraphDataset(Dataset):
    def __init__(self, node_features, edge_matrices, labels, args):
        super(GraphDataset, self).__init__()
        node_features, edge_matrices, labels = map(lambda x: torch.tensor(x).to(args.device), 
                                                   (node_features, edge_matrices, labels))
        self.data = []
        for node_feature, edge_matrix, label in zip(node_features, edge_matrices, labels):
            edge_index = matrix2index(edge_matrix)
            graphdata = graph((edge_index[0], edge_index[1]), num_nodes=node_feature.shape[0])
            graphdata.ndata['ft'] = node_feature.float()
            self.data.append((graphdata, label))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def graph_dataloader(node_features, edge_matrices, labels, args):
    dataset = GraphDataset(node_features, edge_matrices, labels, args)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=torch.cuda.device_count())
