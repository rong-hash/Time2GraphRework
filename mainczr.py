import torch, warnings, argparse
from tqdm import tqdm
from utils import write_log, seed_torch, read_dataset, GraphDataset
seed_torch(42)
from construct_graph import extract_shapelets, embed_series, adjacency_matrix
from network import NeuralNetwork, train, test
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from gat import GATClassifier, collate


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()

# DATA LOADING & SHAPELET EMBEDDING
# DATA LOADING
#parser.add_argument('dataset', help='Path of [.npz] dataset with *ordered* fields "train_data", "train_label", "test_data", "test_label"')
parser.add_argument('--seed',               type=int,   default=42,   help='Random seed')
parser.add_argument('--dataset',            type=str,   default='Coffee', help='Name of UCR dataset')
parser.add_argument('--device',             type=str,   default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')

# SHAPELET EMBEDDING
parser.add_argument('--num-shapelets',      type=int,   default=30,   help='Number of shapelets to extract')
parser.add_argument('--num-segments',       type=int,   default=20,   help='Number of segments for shapelet mapping')
parser.add_argument('--pruning-percentile', type=int,   default=30,   help='Percentile for pruning weak edges')

# GAT NETWORK & GRAPH EMBEDDING
parser.add_argument('--hidden-dim',         type=int,   default=256,  help='Hidden dimension of GAT')
parser.add_argument('--embed-dim',          type=int,   default=64,   help='Embedding dimension of graph (output dim. of GAT)')
parser.add_argument('--num-layers',         type=int,   default=4,    help='Number of layers in GAT')
parser.add_argument('--heads',              type=int,   default=8,    help='Number of attention heads in GAT')
parser.add_argument('--neg-slope',          type=float, default=.2,   help='Negative slope of leakyReLU')
parser.add_argument('--dropout',            type=float, default=.5,   help='Dropout rate in training')
parser.add_argument('--tail',               type=str,   default='resnet', help='Type of prediction tail: [none, mlp, resnet]')

# TRAINING & ENHANCEMENT OPTIONS
parser.add_argument('--epochs',             type=int,   default=100,  help='Number of epochs')
parser.add_argument('--batch-size',         type=int,   default=8,    help='Batch size')
parser.add_argument('--lr',                 type=float, default=.001, help='Learning rate')
parser.add_argument('--weight-decay',       type=float, default=.001, help='Weight decay')
parser.add_argument('--ts2vec',             action='store_true', default=True, help='Switch for using TS2VEC')
parser.add_argument('--dtw',                action='store_true', default=True, help='Switch for using DTW')

args = parser.parse_args()
warnings.filterwarnings('ignore')

log_filename = 'Kmeanstime2graph0723.csv'

def write_log(message):
    with open(log_filename, 'a') as log:
        log.write(message)

write_log('dataset,shapelet_num,shapelet_len,train_acc,test_acc\n')
write_log('%s,%d,%d,' % (args.dataset, args.num_shapelets, args.num_segments))

# DATA LOADING & PREPARATION (utils.py)
train_data, train_label, test_data, test_label = read_dataset(args.dataset)
if train_label.min() == 1: train_label -= 1
if test_label.min() == 1: test_label -= 1
num_classes = 2
len_shapelet = int(train_data.shape[1] / args.num_segments)

# SHAPELET EMBEDDING & GRAPH GENERATION (construct_graph.py, ts2vec.py, kmeans.py)
shapelets = extract_shapelets(train_data,    len_shapelet,  args)
train_node_features = embed_series(train_data,  shapelets,  args)
test_node_features  = embed_series(test_data,   shapelets,  args)
train_edge_matrices = adjacency_matrix(train_node_features, args)
test_edge_matrices  = adjacency_matrix(test_node_features,  args)


trainset = GraphDataset(train_node_features, train_edge_matrices, train_label, args)
testset = GraphDataset(test_node_features, test_edge_matrices, test_label, args)

data_loader = DataLoader(trainset, batch_size=1, shuffle=True,
                         collate_fn=collate)

# # Create model
model = GATClassifier(20, 16, 8, 2)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-3) 
model.to(device)
model.train()
print(model.parameters())
epoch_losses = []
for epoch in range(80):
    epoch_loss = 0
    for iter, (bg, label) in enumerate(data_loader):
        bg = bg.to(device)
        label = label.to(device)
        prediction = model(bg)
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    torch.cuda.empty_cache()
    epoch_loss /= (iter + 1)
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)

model.eval()

result = 0
for i in range(len(testset)):
    bg = testset[i][0].to(device)
    label = testset[i][1].to(device)
    prediction = model(bg)
    out = prediction.argmax()
    if out == label.item():
        result += 1
testacc = result / len(testset)
train_result = 0
for i in range(len(trainset)):
    bg = trainset[i][0].to(device)
    label = trainset[i][1].to(device)
    prediction = model(bg)
    out = prediction.argmax()
    if out == label.item():
        train_result += 1
trainacc = train_result / len(trainset)
print("test accuracy = ", testacc)
print("train accuracy = ", trainacc)
write_log('%f,%f\n' % (trainacc, testacc))