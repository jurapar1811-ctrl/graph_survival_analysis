import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Dropout
from torch_geometric.nn import (
    EdgePooling,
    GraphConv,
    JumpingKnowledge,
    global_max_pool, global_mean_pool)
from torch_geometric.nn import GraphConv, GATv2Conv, GlobalAttention, SAGEConv, EdgeConv, GINConv, NNConv, Sequential, GATConv

###
class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, input_size, edge_weights, sep_edge_weights, GNN1lay):#, GNN1lay):
        super(GNN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(input_size, hidden_channels)
        #self.conv2 = GraphConv(hidden_channels, hidden_channels) #Enelvée car pas utilisée par la suite
        #self.conv3 = GraphConv(512, hidden_channels)
        self.conv4 = GraphConv(hidden_channels, 1)
        #self.lin = Linear(hidden_channels, 1)
        self.edge_weights = edge_weights

    def forward(self, x, edge_index, edge_weight, batch):
        if(self.edge_weights):
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            x = self.conv4(x, edge_index, edge_weight)
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv4(x, edge_index)
        #x = self.conv2(x, edge_index)
        #x = F.relu(x)
        #x = F.relu(x)
        #x = self.conv3(x, edge_index)
        #x = F.relu(x)
        #x = self.conv4(x, edge_index)

        x = global_max_pool(x, batch)

        return x

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels, input_size, MLP2lay, output_size=1, CoxTime=False):
        super(MLP, self).__init__()
        self.CoxTimemodel = CoxTime
        bias_output = not CoxTime

        self.input_size = input_size
        self.hidden_size = hidden_channels
        self.MLP2lay = MLP2lay
        # self.batchnorm0 = torch.nn.BatchNorm1d(self.input_size)
        # self.dropout0 = nn.Dropout(p=0.3)
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        # self.batchnorm1 = torch.nn.BatchNorm1d(self.hidden_size)
        # self.dropout1 = nn.Dropout(p=0.2)
        self.relu = torch.nn.ReLU()
        if(not MLP2lay): self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        # self.batchnorm2 = torch.nn.BatchNorm1d(self.hidden_size)
        # self.dropout2 = nn.Dropout(p=0.2)
        # self.fc21 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc22 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        # self.fc23 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = torch.nn.Linear(self.hidden_size, output_size, bias=bias_output)


    def forward(self, x, time=None, batch=None):
        if(self.CoxTimemodel): x = torch.cat([x, time], dim=1)
        # x = self.batchnorm0(x)
        # x = self.dropout0(x)
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.batchnorm1(x)
        if(not self.MLP2lay):
            # x = self.dropout2(x) 
            x = self.fc2(x)
            x = self.relu(x)
        # x = self.dropout0(x)
        # x = self.fc21(x)
        # x = self.relu(x)
        # x = self.fc22(x)
        # x = self.relu(x)
        # x = self.dropout1(x)
        output = self.fc3(x)
        # output = global_max_pool(output, batch) #N'a aucun sens ??

        return output


class GAT(torch.nn.Module): #Graph Attention Network
    def __init__(self,  hidden_channels, input_size, edge_weights, sep_edge_weights, GNN1lay, heads=2):
        super(GAT, self).__init__()
        torch.manual_seed(12345)
        edge_dim=2 if(sep_edge_weights) else 1
        self.conv1 = GATv2Conv(input_size, hidden_channels, heads=heads, add_self_loops=False, edge_dim=edge_dim)
        # self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.conv4 = GATv2Conv(hidden_channels*heads, 1, heads=1, add_self_loops=False, edge_dim=edge_dim)
        self.lin = Linear(hidden_channels, 1)
        self.edge_weights = edge_weights
        self.GNN1lay = GNN1lay

    def forward(self, x, edge_index, edge_weight, batch):
        if(self.edge_weights):
            x = self.conv1(x, edge_index, edge_weight)
            x = F.relu(x)
            if(not self.GNN1lay):
                x = self.conv4(x, edge_index, edge_weight)
        else:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            if(not self.GNN1lay):
                x = self.conv4(x, edge_index)
        # x = F.relu(x)
        # x = self.conv3(x, edge_index)
        # x = F.relu(x)
        # x = self.conv4(x, edge_index)
        
        x = global_max_pool(x, batch)
        if(self.GNN1lay):
            x = self.lin(x)            
        return x

class EdgePool(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers = 1, mode = 'cat'):
        super().__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(15, hidden_channels, aggr= 'mean')
        self.convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.convs.extend([
            GraphConv(hidden_channels, hidden_channels, aggr= 'mean')
            for i in range(num_layers - 1)
        ])
        self.pools.extend(
            [EdgePooling(hidden_channels) for i in range((num_layers) // 2)])
        self.jump = JumpingKnowledge(mode='cat')#'lstm', channels=hidden_channels, num_layers = num_layers)
        if mode == 'cat':
            self.lin1 = GraphConv(num_layers * hidden_channels, hidden_channels)
        else:
            self.lin1 = GraphConv(hidden_channels, hidden_channels)
        self.lin2 = GraphConv(hidden_channels, 1)


    def forward(self, x, edge_index, batch):

        x = F.relu(self.conv1(x, edge_index))
        xs = [global_mean_pool(x, batch)]
        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x, edge_index))
            xs += [global_mean_pool(x, batch)]
            if i % 2 == 0 and i < len(self.convs) - 1:
                pool = self.pools[i // 2]
                x, edge_index, batch, _ = pool(x, edge_index, batch=batch)
        x = self.jump(xs)
        x = F.relu(self.lin1(x, edge_index))
        x = self.lin2(x, edge_index)
        return x

class Net(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(Net, self).__init__()
        torch.manual_seed(12345)
        numNodesFeatures = 15
        numEdgeFeatures = 2
        numLayers = 1
        nn1 = nn.Sequential(GraphConv(numEdgeFeatures, numLayers), nn.ReLU(), GraphConv(numLayers, numNodesFeatures * hidden_channels))
        self.conv1 = NNConv(numNodesFeatures, hidden_channels, nn1, aggr='mean')

        nn2 = nn.Sequential(GraphConv(numEdgeFeatures, numLayers), nn.ReLU(), GraphConv(numLayers, hidden_channels * hidden_channels))
        self.conv2 = NNConv(hidden_channels, 1, nn2, aggr='mean')

        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = F.relu(self.conv1(x=x, edge_index=edge_index, edge_attr=edge_attr))
        x = self.conv2(x=x, edge_index=edge_index, edge_attr=edge_attr)
        #x = self.conv2(x, data.edge_index, data.edge_attr)
        x = global_max_pool(x, batch)

        # x = F.dropout(x, p=0.1, training=self.training)
        #x = self.lin(x)
        #x = nn.Softmax(dim=1)(x)
        return x


class cnn3d(nn.Module):

    def __init__(self):
        super(cnn3d, self).__init__()
        self.conv1 = self._conv_layer_set(1, 16)
        self.conv2 = self._conv_layer_set(16, 32)
        self.conv3 = self._conv_layer_set(32, 64)
        self.fc1 = nn.Linear(4*4*4*64, 128)
        self.fc2 = nn.Linear(128, 1)
        self.relu = nn.LeakyReLU()
        self.conv1_bn = nn.BatchNorm3d(16)
        self.conv2_bn = nn.BatchNorm3d(32)
        self.conv3_bn = nn.BatchNorm3d(64)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.3)

    def _conv_layer_set(self, in_channels, out_channels):
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=(3, 3, 3),
                stride=1,
                padding=0,
                ),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
            )
        return conv_layer

    def forward(self, x):
        #print('input shape:', x.shape)
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        #x = self.fc1_bn(x)
        x = self.drop(x)
        x = self.fc2(x)
        #print('output shape:', x.shape)

        return x


class MILModel(nn.Module):
    '''
    Model for solving MIL problems
    Args:
      prepNN: neural network created by user processing input before aggregation function (subclass of torch.nn.Module)
      afterNN: neural network created by user processing output of aggregation function and outputing final output of BagModel (subclass of torch.nn.Module)
      aggregation_func: mil.max and mil.mean supported, any aggregation function with argument 'dim' and same behaviour as torch.mean can be used
    Returns:
      Output of forward function.
    '''

    def __init__(self, prepNN, afterNN, aggregation_func):
        super().__init__()

        self.prepNN = prepNN
        self.aggregation_func = aggregation_func
        self.afterNN = afterNN

    def forward(self, input):
        ids = input[1]
        input = input[0]
        # Modify shape of bagids if only 1d tensor
        if (len(ids.shape) == 1):
            ids.resize_(1, len(ids))

        inner_ids = ids[len(ids) - 1]


        NN_out = self.prepNN(input)

        unique, inverse, counts = torch.unique(inner_ids, sorted=True, return_inverse=True, return_counts=True)
        idx = torch.cat([(inverse == x).nonzero()[0] for x in range(len(unique))]).sort()[1]
        bags = unique[idx]
        counts = counts[idx]

        output = torch.empty((len(bags), len(NN_out[0])))

        for i, bag in enumerate(bags):
            output[i] = self.aggregation_func(NN_out[inner_ids == bag], dim=0)[0]

        output = self.afterNN(output)

        if (ids.shape[0] == 1):
            return output
        else:
            ids = ids[:len(ids) - 1]
            mask = torch.empty(0).long()
            for i in range(len(counts)):
                mask = torch.cat((mask, torch.sum(counts[:i], dtype=torch.int64).reshape(1)))
            return (output, ids[:, mask])


class XGBCoxTime(nn.Module):  ### NOT DONE (=MLPVanillaCoxTime for now, no XGB)
    """A version of torchtuples.practical.MLPVanilla that works for CoxTime.
    The difference is that it takes `time` as an additional input and removes the output bias and
    output activation.
    """
    def __init__(self, in_features, num_nodes, batch_norm=True, dropout=None, activation=nn.ReLU,
                 w_init_=lambda w: nn.init.kaiming_normal_(w, nonlinearity='relu')):
        super().__init__()
        in_features += 1
        out_features = 1
        output_activation = None
        output_bias=False
        self.net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout,
                                           activation, output_activation, output_bias, w_init_)

    def forward(self, input, time):
        input = torch.cat([input, time], dim=1)
        return self.net(input)


###########################################################################################################################################################
# Test nb parameters:

if __name__=="__main__": #10 for MLP_clin, 13 for clin_graph, 15 for img

    import sys
    sys.path.insert(1, './') 
    from main.parameters import parameters
    _parameters = parameters()

    nb_params = 0
    for i in range(len(_parameters.params_MLP["hidden_channels"])):
        hidden_channels = _parameters.params_MLP["hidden_channels"][i]
        model = MLP(hidden_channels=hidden_channels, input_size=10)
        for params in model.parameters():
            if(len(params.shape)==1):
                nb_params = nb_params+params.shape[0]
            elif(len(params.shape)==2):
                nb_params = nb_params+params.shape[0]*params.shape[1]
        print(f"Model MLP_clin with hidden_channels={hidden_channels} et input_size=10 has {nb_params} parameters.")

    nb_params = 0
    for i in range(len(_parameters.params_MLP["hidden_channels"])):
        hidden_channels = _parameters.params_MLP["hidden_channels"][i]
        model = MLP(hidden_channels=hidden_channels, input_size=13)
        for params in model.parameters():
            if(len(params.shape)==1):
                nb_params = nb_params+params.shape[0]
            elif(len(params.shape)==2):
                nb_params = nb_params+params.shape[0]*params.shape[1]
        print(f"Model MLP_clin_graph with hidden_channels={hidden_channels} et input_size=13 has {nb_params} parameters.")

    nb_params = 0
    for i in range(len(_parameters.params_MLP["hidden_channels"])):
        hidden_channels = _parameters.params_MLP["hidden_channels"][i]
        model = MLP(hidden_channels=hidden_channels, input_size=15)
        for params in model.parameters():
            if(len(params.shape)==1):
                nb_params = nb_params+params.shape[0]
            elif(len(params.shape)==2):
                nb_params = nb_params+params.shape[0]*params.shape[1]
        print(f"Model MLP_img with hidden_channels={hidden_channels} et input_size=15 has {nb_params} parameters.")


    nb_params = 0
    for i in range(len(_parameters.params_MIL["hidden_channels"])):
        hidden_channels = _parameters.params_MIL["hidden_channels"][i]
        prepNN = torch.nn.Sequential(
                    torch.nn.Linear(15, 32),        
                    torch.nn.ReLU())
        afterNN = torch.nn.Sequential(torch.nn.Linear(32, 1))
        model = MILModel(prepNN, afterNN, torch.max)
        for params in model.parameters():
            if(len(params.shape)==1):
                nb_params = nb_params+params.shape[0]
            elif(len(params.shape)==2):
                nb_params = nb_params+params.shape[0]*params.shape[1]
        print(f"Model MIL_img with input_size=15 has {nb_params} parameters.")


    nb_params = 0
    for i in range(len(_parameters.params_GNN["hidden_channels"])):
        hidden_channels = _parameters.params_GNN["hidden_channels"][i]
        model = GNN(hidden_channels=hidden_channels)
        for params in model.parameters():
            if(len(params.shape)==1):
                nb_params = nb_params+params.shape[0]
            elif(len(params.shape)==2):
                nb_params = nb_params+params.shape[0]*params.shape[1]
            #print(params.shape)
        print(f"Model GraphConv with hidden_channels={hidden_channels} has {nb_params} parameters.")

    nb_params = 0
    for i in range(len(_parameters.params_GNN["hidden_channels"])):
        hidden_channels = _parameters.params_GNN["hidden_channels"][i]
        model = GAT(hidden_channels=hidden_channels)
        for params in model.parameters():
            if(len(params.shape)==1):
                nb_params = nb_params+params.shape[0]
            elif(len(params.shape)==2):
                nb_params = nb_params+params.shape[0]*params.shape[1]
            elif(len(params.shape)==3):
                nb_params = nb_params+params.shape[0]*params.shape[1]*params.shape[2]
            #print(params.shape)
        print(f"Model GATv2 with hidden_channels={hidden_channels} et heads=2 has {nb_params} parameters.")

