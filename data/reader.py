import hashlib
import pickle
import time

import dgl
import numpy as np
import torch
from scipy import sparse as sp


class datasetDGLs(torch.utils.data.Dataset):

    def __init__(self, data, split):

        self.split = split
        self.is_test = split.lower() in ["test", "val"]

        self.dataset = data
        self.node_targets = []
        self.graph_lists = []
        self.n_samples = len(self.dataset)
        self._prepare()

    def _prepare(self):

        for g, t in self.dataset:

            try:
                g.edata["feat"] = g.edata["feat"].type(torch.float32)
            except KeyError:
                pass

            self.graph_lists.append(g)
            self.node_targets.append(t)

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):
        """
        Get the idx^th sample.
        Parameters
        ---------
        idx : int
            The sample index.
        Returns
        -------
        (dgl.DGLGraph, int)
            DGLGraph with node feature stored in `feat` field
            And its target.
        """
        return self.graph_lists[idx], self.node_targets[idx]


def self_loop(g):
    """
    Utility function only, to be used only when necessary as per user self_loop flag
    : Overwriting the function dgl.transform.add_self_loop() to not miss ndata['feat'] and edata['feat']


    This function is called inside a function in customDataset class.
    """
    new_g = dgl.DGLGraph()
    new_g.add_nodes(g.number_of_nodes())
    new_g.ndata["feat"] = g.ndata["feat"]

    src, dst = g.all_edges(order="eid")
    src = dgl.backend.zerocopy_to_numpy(src)
    dst = dgl.backend.zerocopy_to_numpy(dst)
    non_self_edges_idx = src != dst
    nodes = np.arange(g.number_of_nodes())
    new_g.add_edges(src[non_self_edges_idx], dst[non_self_edges_idx])
    new_g.add_edges(nodes, nodes)

    # This new edata is not used since this function gets called only for GCN, GAT
    # However, we need this for the generic requirement of ndata and edata
    new_g.edata["feat"] = torch.zeros(new_g.number_of_edges())
    return new_g


def laplacian_positional_encoding(g, pos_enc_dim):
    """
    Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = g.adjacency_matrix(scipy_fmt="csr").astype(float)

    # A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with scipy
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which="SR", tol=1e-2)  # for 40 PEs

    EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    g.ndata["lap_pos_enc"] = torch.from_numpy(EigVec[:, 1 : pos_enc_dim + 1]).real.float()

    return g


def wl_positional_encoding(g):
    """
    WL-based absolute positional embedding
    adapted from

    "Graph-Bert: Only Attention is Needed for Learning Graph Representations"
    Zhang, Jiawei and Zhang, Haopeng and Xia, Congying and Sun, Li, 2020
    https://github.com/jwzhanggy/Graph-Bert
    """
    max_iter = 2
    node_color_dict = {}
    node_neighbor_dict = {}

    edge_list = torch.nonzero(g.adj().to_dense() != 0, as_tuple=False).numpy()
    node_list = g.nodes().numpy()

    # setting init
    for node in node_list:
        node_color_dict[node] = 1
        node_neighbor_dict[node] = {}

    for pair in edge_list:
        u1, u2 = pair
        if u1 not in node_neighbor_dict:
            node_neighbor_dict[u1] = {}
        if u2 not in node_neighbor_dict:
            node_neighbor_dict[u2] = {}
        node_neighbor_dict[u1][u2] = 1
        node_neighbor_dict[u2][u1] = 1

    # WL recursion
    iteration_count = 1
    exit_flag = False
    while not exit_flag:
        new_color_dict = {}
        for node in node_list:
            neighbors = node_neighbor_dict[node]
            neighbor_color_list = [node_color_dict[neb] for neb in neighbors]
            color_string_list = [str(node_color_dict[node])] + sorted([str(color) for color in neighbor_color_list])
            color_string = "_".join(color_string_list)
            hash_object = hashlib.md5(color_string.encode())
            hashing = hash_object.hexdigest()
            new_color_dict[node] = hashing
        color_index_dict = {k: v + 1 for v, k in enumerate(sorted(set(new_color_dict.values())))}
        for node in new_color_dict:
            new_color_dict[node] = color_index_dict[new_color_dict[node]]
        if node_color_dict == new_color_dict or iteration_count == max_iter:
            exit_flag = True
        else:
            node_color_dict = new_color_dict
        iteration_count += 1

    g.ndata["wl_pos_enc"] = torch.LongTensor(list(node_color_dict.values()))
    return g


class customDataset(torch.utils.data.Dataset):

    def __init__(self, name):

        start = time.time()
        self.name = name

        with open("data/custom/" + name + ".pkl", "rb") as f:
            print(f"\n[I] Loading dataset {name.upper()}")
            f = pickle.load(f)

            train = f[0]
            val = f[1]
            test = f[2]

            self.train = datasetDGLs(train, "train")
            self.val = datasetDGLs(val, "val")
            self.test = datasetDGLs(test, "test")

        print("\n[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time() - start))

    # form a mini batch from a given list of samples = [(graph, target) pairs]
    def collate(self, samples):
        graphs, targets = map(list, zip(*samples))
        targets = torch.cat(targets)
        batched_graph = dgl.batch(graphs)

        return batched_graph, targets

    def _add_self_loops(self):

        # function for adding self loops
        # this function will be called only if self_loop flag is True

        self.train.graph_lists = [self_loop(g) for g in self.train.graph_lists]
        self.val.graph_lists = [self_loop(g) for g in self.val.graph_lists]
        self.test.graph_lists = [self_loop(g) for g in self.test.graph_lists]

    def _add_laplacian_positional_encodings(self, pos_enc_dim):

        # Graph positional encoding v/ Laplacian eigenvectors
        self.train.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.train.graph_lists]
        self.val.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.val.graph_lists]
        self.test.graph_lists = [laplacian_positional_encoding(g, pos_enc_dim) for g in self.test.graph_lists]

    def _add_wl_positional_encodings(self):

        # WL positional encoding from Graph-Bert, Zhang et al 2020.
        self.train.graph_lists = [wl_positional_encoding(g) for g in self.train.graph_lists]
        self.val.graph_lists = [wl_positional_encoding(g) for g in self.val.graph_lists]
        self.test.graph_lists = [wl_positional_encoding(g) for g in self.test.graph_lists]
