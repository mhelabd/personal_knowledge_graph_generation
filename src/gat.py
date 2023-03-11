import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn.conv import MessagePassing

class GAT(MessagePassing):

    def __init__(self, in_channels, out_channels, heads = 2,
                 negative_slope = 0.2, dropout = 0., **kwargs):
        super(GAT, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.lin_l = None
        self.lin_r = None
        self.att_l = None
        self.att_r = None

        ############################################################################
        # TODO: Your code here! 
        # Define the layers needed for the message functions below.
        # self.lin_l is the linear transformation that you apply to embeddings 
        # BEFORE message passing.
        # 
        # Pay attention to dimensions of the linear layers, since we're using 
        # multi-head attention.
        # Our implementation is ~1 lines, but don't worry if you deviate from this.

        self.lin_l = nn.Linear(self.in_channels, self.out_channels * self.heads)
        # self.lin_r = nn.Linear(self.in_channels, self.out_channels * self.heads)
        
        ############################################################################

        self.lin_r = self.lin_l

        ############################################################################
        # TODO: Your code here! 
        # Define the attention parameters \overrightarrow{a_l/r}^T in the above intro.
        # You have to deal with multi-head scenarios.
        # Use nn.Parameter instead of nn.Linear
        # Our implementation is ~2 lines, but don't worry if you deviate from this.

        ############################################################################

        self.att_l = nn.Parameter(torch.Tensor(1, self.heads, self.out_channels))
        self.att_r = nn.Parameter(torch.Tensor(1, self.heads, self.out_channels))
        

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att_l)
        nn.init.xavier_uniform_(self.att_r)

    def forward(self, x, edge_index, size = None):
        
        H, C = self.heads, self.out_channels

        ############################################################################
        # TODO: Your code here! 
        # Implement message passing, as well as any pre- and post-processing (our update rule).
        # 1. First apply linear transformation to node embeddings, and split that 
        #    into multiple heads. We use the same representations for source and
        #    target nodes, but apply different linear weights (W_l and W_r)
        # 2. Calculate alpha vectors for central nodes (alpha_l) and neighbor nodes (alpha_r).
        # 3. Call propagate function to conduct the message passing. 
        #    3.1 Remember to pass alpha = (alpha_l, alpha_r) as a parameter.
        #    3.2 See there for more information: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
        # 4. Transform the output back to the shape of [N, H * C].
        # Our implementation is ~5 lines, but don't worry if you deviate from this.


        ############################################################################
        h = self.lin_l(x).reshape(-1, H, C) # [N, H, C]
        t = self.lin_r(x).reshape(-1, H, C) # [N, H, C]
        alpha_l = (h * self.att_l) # [N, H, C] = [N, H, C] * [1, H, C]
        alpha_r = (t * self.att_r) # [N, H, C] = [N, H, C] * [1, H, C]
        out = self.propagate(edge_index, alpha=(alpha_l, alpha_r), x=(h, t), size=size)
        out = out.reshape(-1, H*C)
        
        return out


    def message(self, x_j, alpha_j, alpha_i, index, ptr, size_i):

        ############################################################################
        # TODO: Your code here! 
        # Implement your message function. Putting the attention in message 
        # instead of in update is a little tricky.
        # 1. Calculate the final attention weights using alpha_i and alpha_j,
        #    and apply leaky Relu.
        # 2. Calculate softmax over the neighbor nodes for all the nodes. Use 
        #    torch_geometric.utils.softmax instead of the one in Pytorch.
        # 3. Apply dropout to attention weights (alpha).
        # 4. Multiply embeddings and attention weights. As a sanity check, the output
        #    should be of shape [E, H, C].
        # 5. ptr (LongTensor, optional): If given, computes the softmax based on
        #    sorted inputs in CSR representation. You can simply pass it to softmax.
        # Our implementation is ~4-5 lines, but don't worry if you deviate from this.


        ############################################################################
        alpha_ij = F.leaky_relu(alpha_i + alpha_j, negative_slope=0.2)   # [N, H]
        alpha_ij = torch_geometric.utils.softmax(alpha_ij, index, ptr, size_i)
        alpha_ij = F.dropout(alpha_ij, self.dropout, training=self.training)
        out = alpha_ij * x_j  # [E, H, C] = [N, H, C] *[N, H, C]

        return out


    def aggregate(self, inputs, index, dim_size = None):

        ############################################################################
        # TODO: Your code here! 
        # Implement your aggregate function here.
        # See here as how to use torch_scatter.scatter: https://pytorch-scatter.readthedocs.io/en/latest/_modules/torch_scatter/scatter.html
        # Pay attention to "reduce" parameter is different from that in GraphSage.
        # Our implementation is ~1 lines, but don't worry if you deviate from this.

        ############################################################################
        out = torch_scatter.scatter(inputs, index, dim=0, dim_size=dim_size, reduce='sum')
        return out