import torch.nn as nn
import dgl


class HeteroGraphConv(nn.Module):
    r"""A generic module for computing convolution on heterogeneous graphs.

    The heterograph convolution applies sub-modules on their associating
    relation graphs, which reads the features from source nodes and writes the
    updated ones to destination nodes. If multiple relations have the same
    destination node types, their results are aggregated by the specified method.

    If the relation graph has no edge, the corresponding module will not be called.

    Parameters
    ----------
    mods : dict[str, nn.Module]
        Modules associated with every edge types. The forward function of each
        module must have a `DGLHeteroGraph` object as the first argument, and
        its second argument is either a tensor object representing the node
        features or a pair of tensor object representing the source and destination
        node features.

    Attributes
    ----------
    mods : dict[str, nn.Module]
        Modules associated with every edge types.
    """

    def __init__(self, mods: dict):
        super(HeteroGraphConv, self).__init__()
        self.mods = nn.ModuleDict(mods)

    def forward(self, graph: dgl.DGLHeteroGraph, input_src: dict, input_dst: dict,
                node_transformation_weight: nn.ParameterDict, relations_attention_weight: nn.ParameterDict):
        """
        call the forward function with each module.

        Parameters
        ----------
        graph: DGLHeteroGraph, The Heterogeneous Graph.
        input_src: dict[tuple, Tensor], Input source node features {ntype: features, }.
        input_dst: dict[tuple, Tensor], Input destination node features {ntype: features, }.
        node_transformation_weight: nn.ParameterDict, weights {ntype, (inp_dim, hidden_dim)}
        relations_attention_weight: nn.ParameterDict, weights {etype, (n_heads, 2 * hidden_dim)}

        Returns
        -------
        outputs, dict[tuple, Tensor]  Output representations for every relation -> {(stype, etype, dtype): features}.
        """

        # dictionary, {(srctype, etype, dsttype): representations}
        outputs = dict()

        for stype, etype, dtype in graph.canonical_etypes:
            rel_graph = graph[stype, etype, dtype]
            if rel_graph.number_of_edges() == 0:
                continue
            # for example, (author, writes, paper) relation, take author as src_nodes, take paper as dst_nodes
            dst_representation = self.mods[etype](rel_graph,
                                                  (input_src[stype], input_dst[dtype]),
                                                  node_transformation_weight[dtype],
                                                  node_transformation_weight[stype],
                                                  relations_attention_weight[etype])

            # dst_representation (dst_nodes, hid_dim)
            outputs[(stype, etype, dtype)] = dst_representation

        return outputs
