import torch.nn as nn

from layers.graph_transformer_edge_layer import GraphTransformerLayer
from layers.mlp_readout_layer import MLPReadout
from utils.metrics import SquaredErrorRelevanceArea


class GraphTransformerNet(nn.Module):

    def __init__(self, net_params):
        super().__init__()

        in_dim_node = net_params["in_dim_node"]
        in_dim_edge = net_params["in_dim_edge"]
        hidden_dim = net_params["hidden_dim"]
        num_heads = net_params["n_heads"]
        out_dim = net_params["out_dim"]
        in_feat_dropout = net_params["in_feat_dropout"]
        dropout = net_params["dropout"]
        n_layers = net_params["L"]

        self.readout = net_params["readout"]
        self.layer_norm = net_params["layer_norm"]
        self.batch_norm = net_params["batch_norm"]
        self.residual = net_params["residual"]
        self.edge_feat = net_params["edge_feat"]
        self.dropout = dropout
        self.device = net_params["device"]
        self.lap_pos_enc = net_params["lap_pos_enc"]
        self.wl_pos_enc = net_params["wl_pos_enc"]

        max_wl_role_index = 100

        if self.lap_pos_enc:
            pos_enc_dim = net_params["pos_enc_dim"]
            self.embedding_lap_pos_enc = nn.Linear(pos_enc_dim, hidden_dim)
        if self.wl_pos_enc:
            self.embedding_wl_pos_enc = nn.Embedding(max_wl_role_index, hidden_dim)

        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    hidden_dim, hidden_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual
                )
                for _ in range(n_layers - 1)
            ]
        )
        self.layers.append(
            GraphTransformerLayer(
                hidden_dim, out_dim, num_heads, dropout, self.layer_norm, self.batch_norm, self.residual
            )
        )
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        self.MLP_layer = MLPReadout(hidden_dim, 1)

        self.input_layer_node = nn.Linear(in_dim_node, hidden_dim)
        self.input_layer_edge = nn.Linear(in_dim_edge, hidden_dim)

        # Criteria for loss functions
        self.criterion_sera = SquaredErrorRelevanceArea()

    def forward(self, g, h, e, h_lap_pos_enc=None, h_wl_pos_enc=None):

        h = self.input_layer_node(h)
        h = self.in_feat_dropout(h)

        h = self.MLP_layer(h)
        h = self.in_feat_dropout(h)

        if self.lap_pos_enc:
            h_lap_pos_enc = self.embedding_lap_pos_enc(h_lap_pos_enc.float())
            h = h + h_lap_pos_enc
        if self.wl_pos_enc:
            h_wl_pos_enc = self.embedding_wl_pos_enc(h_wl_pos_enc)
            h = h + h_wl_pos_enc

        e = self.input_layer_edge(e)

        for conv in self.layers:
            h, e = conv(g, h, e)

        h_out = self.MLP_layer(h)

        return h_out

    def loss(self, scores, targets, rels):

        res = self.criterion_sera.loss(scores, targets, rels)

        return res
