# import hydra
import torch
import torch.nn.functional as F
from einops.layers.torch import Rearrange

from nn.pooling import HomogeneousAggregator, HeterogeneousAggregator, AttentionBasedPooling
from nn.graph_constructor import GraphConstructor
import torch.nn as nn
from torch_geometric.nn.aggr import (
    AttentionalAggregation,
    GraphMultisetTransformer,
    MaxAggregation,
    MeanAggregation,
    SetTransformerAggregation,
)

class RelationalTransformer(nn.Module):
    def __init__(
        self,
        graph_constructor,
        args,
    ):
        super().__init__()
        assert args.use_cls_token == (args.pooling_method == "cls_token")
        self.pooling_method = args.pooling_method
        self.pooling_layer_idx = args.pooling_layer_idx
        self.rev_edge_features = args.rev_edge_features
        self.nodes_per_layer = args.layer_layout
        graph_constructor["d_node"] = args.d_node
        graph_constructor["d_edge"] = args.d_edge
        graph_constructor["layer_layout"] = args.layer_layout
        graph_constructor["rev_edge_features"] = args.rev_edge_features
        self.construct_graph = GraphConstructor(
            **graph_constructor
        )

        self.use_cls_token = args.use_cls_token
        if args.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(args.d_node))

        self.layers = nn.ModuleList(
            [
                torch.jit.script(
                    RTLayer(
                        args.d_node,
                        args.d_edge,
                        args.d_attn_hid,
                        args.d_node_hid,
                        args.d_edge_hid,
                        args.n_heads,
                        args.dropout,
                        node_update_type=args.node_update_type,
                        disable_edge_updates=(
                            (args.disable_edge_updates or (i == args.n_layers - 1))
                            and args.pooling_method != "mean_edge"
                            and args.pooling_layer_idx != "all"
                        ),
                        modulate_v=args.modulate_v,
                        use_ln=args.use_ln,
                        tfixit_init=args.tfixit_init,
                        n_layers=args.n_layers,
                    )
                )
                for i in range(args.n_layers)
            ]
        )

        if args.pooling_method != "cls_token":
            # self.pool = HeterogeneousAggregator(
            #     args.layer_layout[1] // 2,
            #     args.layer_layout[1] // 2,
            #     args.layer_layout[1] // 4,
            #     args.pooling_method,
            #     args.pooling_layer_idx,
            #     args.d_node,
            #     1
            # )
            self.pool = AttentionBasedPooling(args.d_node, args.d_node)

        self.num_graph_features = (
            args.layer_layout[-1] * args.d_node
            if args.pooling_method == "cat" and args.pooling_layer_idx == "last"
            else args.d_edge if args.pooling_method in ("mean_edge", "max_edge") else args.d_node
        )
        self.proj_out = nn.Sequential(
            nn.Linear(self.num_graph_features, args.d_out_hid),
            nn.GELU(),
            # nn.Linear(args.d_out_hid, args.d_out_hid),
            # nn.ReLU(),
            nn.Linear(args.d_out_hid, args.d_out),
        )

        self.final_features = (None,None,None,None)

    def forward(self, inputs):
        attn_weights = None
        node_features, edge_features, mask = self.construct_graph(inputs)
        if self.use_cls_token:
            node_features = torch.cat(
                [
                    # repeat(self.cls_token, "d -> b 1 d", b=node_features.size(0)),
                    self.cls_token.unsqueeze(0).expand(node_features.size(0), 1, -1),
                    node_features,
                ],
                dim=1,
            )
            edge_features = F.pad(edge_features, (0, 0, 1, 0, 1, 0), value=0)
        for layer in self.layers:
            node_features, edge_features, attn_weights = layer(node_features, edge_features, mask)

        if self.pooling_method == "cls_token":
            graph_features = node_features[:, 0]
        else:
            node_features_pooled = self.pool(node_features)
            edge_features_pooled = self.pool(edge_features.sum(1))
            graph_features = torch.cat([node_features_pooled, edge_features_pooled], dim=-1)
            # graph_features = self.pool(node_features, edge_features)
        self.final_features = (graph_features, node_features, edge_features, attn_weights)
        return graph_features


class RTLayer(nn.Module):
    def __init__(
        self,
        d_node,
        d_edge,
        d_attn_hid,
        d_node_hid,
        d_edge_hid,
        n_heads,
        dropout,
        node_update_type="rt",
        disable_edge_updates=False,
        modulate_v=True,
        use_ln=True,
        tfixit_init=False,
        n_layers=None,
    ):
        super().__init__()
        self.node_update_type = node_update_type
        self.disable_edge_updates = disable_edge_updates
        self.use_ln = use_ln
        self.n_layers = n_layers

        self.self_attn = torch.jit.script(
            RTAttention(
                d_node,
                d_edge,
                d_attn_hid,
                n_heads,
                modulate_v=modulate_v,
                use_ln=use_ln,
            )
        )
        # self.self_attn = RTAttention(d_hid, d_hid, d_hid, n_heads)
        self.lin0 = Linear(d_node, d_node)
        self.dropout0 = nn.Dropout(dropout)
        if use_ln:
            self.node_ln0 = nn.LayerNorm(d_node)
            self.node_ln1 = nn.LayerNorm(d_node)
        else:
            self.node_ln0 = nn.Identity()
            self.node_ln1 = nn.Identity()

        act_fn = nn.GELU

        self.node_mlp = nn.Sequential(
            Linear(d_node, d_node_hid, bias=False),
            act_fn(),
            Linear(d_node_hid, d_node),
            nn.Dropout(dropout),
        )

        if not self.disable_edge_updates:
            self.edge_updates = EdgeLayer(
                d_node=d_node,
                d_edge=d_edge,
                d_edge_hid=d_edge_hid,
                dropout=dropout,
                act_fn=act_fn,
                use_ln=use_ln,
            )
        else:
            self.edge_updates = NoEdgeLayer()

        if tfixit_init:
            self.fixit_init()

    def fixit_init(self):
        temp_state_dict = self.state_dict()
        n_layers = self.n_layers
        for name, param in self.named_parameters():
            if "weight" in name:
                if name.split(".")[0] in ["node_mlp", "edge_mlp0", "edge_mlp1"]:
                    temp_state_dict[name] = (0.67 * (n_layers) ** (-1.0 / 4.0)) * param
                elif name.split(".")[0] in ["self_attn"]:
                    temp_state_dict[name] = (0.67 * (n_layers) ** (-1.0 / 4.0)) * (
                        param * (2**0.5)
                    )

        self.load_state_dict(temp_state_dict)

    def node_updates(self, node_features, edge_features, mask):
        out = self.self_attn(node_features, edge_features, mask)
        attn_out, attn_weights = out
        node_features = self.node_ln0(
            node_features
            + self.dropout0(
                self.lin0(attn_out)
            )
        )
        node_features = self.node_ln1(node_features + self.node_mlp(node_features))

        return node_features, attn_weights

    def forward(self, node_features, edge_features, mask):
        node_features, attn_weights = self.node_updates(node_features, edge_features, mask)
        edge_features = self.edge_updates(node_features, edge_features, mask)

        return node_features, edge_features, attn_weights


class EdgeLayer(nn.Module):
    def __init__(
        self,
        *,
        d_node,
        d_edge,
        d_edge_hid,
        dropout,
        act_fn,
        use_ln=True,
    ) -> None:
        super().__init__()
        self.edge_mlp0 = EdgeMLP(
            d_edge=d_edge,
            d_node=d_node,
            d_edge_hid=d_edge_hid,
            act_fn=act_fn,
            dropout=dropout,
        )
        self.edge_mlp1 = nn.Sequential(
            Linear(d_edge, d_edge_hid, bias=False),
            act_fn(),
            Linear(d_edge_hid, d_edge),
            nn.Dropout(dropout),
        )
        if use_ln:
            self.eln0 = nn.LayerNorm(d_edge)
            self.eln1 = nn.LayerNorm(d_edge)
        else:
            self.eln0 = nn.Identity()
            self.eln1 = nn.Identity()

    def forward(self, node_features, edge_features, mask):
        edge_features = self.eln0(
            edge_features + self.edge_mlp0(node_features, edge_features)
        )
        edge_features = self.eln1(edge_features + self.edge_mlp1(edge_features))
        return edge_features


class NoEdgeLayer(nn.Module):
    def forward(self, node_features, edge_features, mask):
        return edge_features


class EdgeMLP(nn.Module):
    def __init__(self, *, d_node, d_edge, d_edge_hid, act_fn, dropout):
        super().__init__()
        self.reverse_edge = Rearrange("b n m d -> b m n d")
        self.lin0_e = Linear(2 * d_edge, d_edge_hid)
        self.lin0_s = Linear(d_node, d_edge_hid)
        self.lin0_t = Linear(d_node, d_edge_hid)
        self.act = act_fn()
        self.lin1 = Linear(d_edge_hid, d_edge)
        self.drop = nn.Dropout(dropout)

    def forward(self, node_features, edge_features):
        source_nodes = (
            self.lin0_s(node_features)
            .unsqueeze(-2)
            .expand(-1, -1, node_features.size(-2), -1)
        )
        target_nodes = (
            self.lin0_t(node_features)
            .unsqueeze(-3)
            .expand(-1, node_features.size(-2), -1, -1)
        )

        # reversed_edge_features = self.reverse_edge(edge_features)
        edge_features = self.lin0_e(
            torch.cat([edge_features, self.reverse_edge(edge_features)], dim=-1)
        )
        edge_features = edge_features + source_nodes + target_nodes
        edge_features = self.act(edge_features)
        edge_features = self.lin1(edge_features)
        edge_features = self.drop(edge_features)

        return edge_features


class RTAttention(nn.Module):
    def __init__(self, d_node, d_edge, d_hid, n_heads, modulate_v=None, use_ln=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_node = d_node
        self.d_edge = d_edge
        self.d_hid = d_hid
        self.use_ln = use_ln
        self.modulate_v = modulate_v
        self.scale = 1 / (d_hid**0.5)
        self.split_head_node = Rearrange("b n (h d) -> b h n d", h=n_heads)
        self.split_head_edge = Rearrange("b n m (h d) -> b h n m d", h=n_heads)
        self.cat_head_node = Rearrange("... h n d -> ... n (h d)", h=n_heads)

        self.qkv_node = Linear(d_node, 3 * d_hid, bias=False)
        self.edge_factor = 4 if modulate_v else 3
        self.qkv_edge = Linear(d_edge, self.edge_factor * d_hid, bias=False)
        self.proj_out = Linear(d_hid, d_node)

    def forward(self, node_features, edge_features, mask):
        qkv_node = self.qkv_node(node_features)
        # qkv_node = rearrange(qkv_node, "b n (h d) -> b h n d", h=self.n_heads)
        qkv_node = self.split_head_node(qkv_node)
        q_node, k_node, v_node = torch.chunk(qkv_node, 3, dim=-1)

        qkv_edge = self.qkv_edge(edge_features)
        # qkv_edge = rearrange(qkv_edge, "b n m (h d) -> b h n m d", h=self.n_heads)
        qkv_edge = self.split_head_edge(qkv_edge)
        qkv_edge = torch.chunk(qkv_edge, self.edge_factor, dim=-1)
        # q_edge, k_edge, v_edge, q_edge_b, k_edge_b, v_edge_b = torch.chunk(
        #     qkv_edge, 6, dim=-1
        # )
        # qkv_edge = [item.masked_fill(mask.unsqueeze(1) == 0, 0) for item in qkv_edge]

        q = q_node.unsqueeze(-2) + qkv_edge[0]  # + q_edge_b
        k = k_node.unsqueeze(-3) + qkv_edge[1]  # + k_edge_b
        if self.modulate_v:
            v = v_node.unsqueeze(-3) * qkv_edge[3] + qkv_edge[2]
        else:
            v = v_node.unsqueeze(-3) + qkv_edge[2]
        dots = self.scale * torch.einsum("b h i j d, b h i j d -> b h i j", q, k)
        # dots.masked_fill_(mask.unsqueeze(1).squeeze(-1) == 0, -1e-9)

        attn = F.softmax(dots, dim=-1)
        out = torch.einsum("b h i j, b h i j d -> b h i d", attn, v)
        out = self.cat_head_node(out)
        return self.proj_out(out), attn


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)  # , gain=1 / math.sqrt(2))
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m