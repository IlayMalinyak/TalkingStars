import torch
from torch.linalg import svd

def calculate_node_importance_svd(node_features, edge_features, top_k_ratio=0.5):
    """
    Calculate node importance using SVD and prune less important nodes.
    
    Args:
        node_features: Tensor of shape [batch_size, num_nodes, feature_dim]
        edge_features: Tensor of shape [batch_size, num_nodes, num_nodes, edge_dim]
        top_k_ratio: Ratio of nodes to keep (0-1)
        
    Returns:
        pruned_node_features: Tensor with only important nodes
        pruned_edge_features: Updated edge features
        node_mask: Boolean mask of nodes to keep
        importance_scores: Importance score for each node
    """
    batch_size, num_nodes, feature_dim = node_features.shape
    device = node_features.device
    
    # Create importance scores for each node in the batch
    importance_scores = torch.zeros(batch_size, num_nodes, device=device)
    node_masks = torch.zeros(batch_size, num_nodes, dtype=torch.bool, device=device)
    
    for b in range(batch_size):
        # print(f"Processing batch {b + 1}/{batch_size}")
        # For node importance, we can use the node features directly
        U, S, Vh = svd(node_features[b])
        
        # Node importance based on contribution to singular values
        # Calculate how much each node contributes to the principal components
        contribution = torch.zeros(num_nodes, device=device)
        
        # We use the left singular vectors weighted by singular values
        # Higher values mean the node contributes more to important dimensions
        weighted_U = U * S.unsqueeze(0)
        contribution = torch.norm(weighted_U, dim=1)
        
        # Alternative: use adjacency matrix for structural importance
        adj_matrix = (edge_features[b].sum(dim=-1) != 0).float()
        U_adj, S_adj, Vh_adj = svd(adj_matrix)
        
        # Combine feature-based and structural importance
        structural_importance = torch.norm(U_adj * S_adj.unsqueeze(0), dim=1)
        combined_importance = contribution + structural_importance
        
        # Store importance scores
        importance_scores[b] = combined_importance
        
        # Determine number of nodes to keep
        k = max(1, int(num_nodes * top_k_ratio))
        
        # Select top-k important nodes
        _, top_indices = torch.topk(combined_importance, k)
        node_masks[b, top_indices] = True
    
    # Create pruned node features
    pruned_node_features = node_features.clone()
    pruned_node_features[~node_masks.unsqueeze(-1).expand_as(node_features)] = 0
    
    # Update edge features to reflect pruned nodes
    pruned_edge_features = edge_features.clone()
    
    # Zero out edges connected to pruned nodes
    for b in range(batch_size):
        for i in range(num_nodes):
            if not node_masks[b, i]:
                pruned_edge_features[b, i, :, :] = 0
                pruned_edge_features[b, :, i, :] = 0
    
    return pruned_node_features, pruned_edge_features, node_masks, importance_scores

def svd_graph_reduction(node_features, edge_features, reduction_ratio=0.25):
    """
    Reduce graph size using SVD-based node importance.
    
    Args:
        node_features: Original node features [batch_size, num_nodes, feature_dim]
        edge_features: Original edge features [batch_size, num_nodes, num_nodes, edge_dim]
        reduction_ratio: Ratio of nodes to keep (0-1)
        
    Returns:
        Reduced node features, edge features, and mask for remaining nodes
    """
    batch_size, num_nodes, feature_dim = node_features.shape
    edge_dim = edge_features.shape[-1]
    device = node_features.device
    
    # Calculate node importance and get mask for nodes to keep
    pruned_nodes, pruned_edges, node_mask, _ = calculate_node_importance_svd(
        node_features, edge_features, top_k_ratio=reduction_ratio
    )
    
    # Count how many nodes to keep for each batch item
    nodes_to_keep = node_mask.sum(dim=1)
    max_nodes = nodes_to_keep.max().item()
    
    # Create index arrays for advanced indexing
    batch_indices = torch.arange(batch_size, device=device).view(-1, 1, 1)
    
    # Create new tensors with reduced size
    reduced_nodes = torch.zeros(batch_size, max_nodes, feature_dim, device=device)
    reduced_edges = torch.zeros(batch_size, max_nodes, max_nodes, edge_dim, device=device)
    
    # For each batch, gather the kept nodes and their edges
    for b in range(batch_size):
        kept_indices = torch.where(node_mask[b])[0]
        num_kept = kept_indices.size(0)
        
        # Copy node features using advanced indexing
        reduced_nodes[b, :num_kept] = node_features[b, kept_indices]
        
        # Create a meshgrid for edge indices
        rows, cols = torch.meshgrid(kept_indices, kept_indices, indexing='ij')
        
        # Copy all edges at once using advanced indexing
        reduced_edges[b, :num_kept, :num_kept] = edge_features[b, rows, cols]
    
    # Create the mask in one operation
    reduced_mask = reduced_edges.sum(dim=-1, keepdim=True) != 0
    
    return reduced_nodes, reduced_edges, reduced_mask

def fast_node_importance(node_features, edge_features, importance_type="combined"):
    """
    Calculate node importance using efficient heuristics instead of SVD.
    
    Args:
        node_features: Tensor of shape [batch_size, num_nodes, feature_dim]
        edge_features: Tensor of shape [batch_size, num_nodes, num_nodes, edge_dim]
        importance_type: Type of importance to calculate ("feature", "structure", "combined")
        
    Returns:
        Tensor of shape [batch_size, num_nodes] containing importance scores
    """
    batch_size, num_nodes, feature_dim = node_features.shape
    device = node_features.device
    
    # 1. Feature-based importance: L2 norm of feature vectors
    # Higher magnitude = more information content
    feature_importance = torch.norm(node_features, dim=2)
    
    # 2. Structure-based importance: node degree and edge weights
    # Sum of absolute edge weights (both incoming and outgoing)
    edge_mask = (edge_features.sum(dim=-1) != 0).float()
    in_degree = edge_mask.sum(dim=1)  # Sum across source nodes
    out_degree = edge_mask.sum(dim=2)  # Sum across target nodes
    
    # Calculate edge weight magnitude
    edge_weights = torch.norm(edge_features, dim=3)
    weighted_in = torch.sum(edge_weights, dim=1)  # Sum incoming edge weights
    weighted_out = torch.sum(edge_weights, dim=2)  # Sum outgoing edge weights
    
    # Combine degrees and weights 
    structural_importance = in_degree + out_degree + weighted_in + weighted_out
    
    # Return importance based on specified type
    if importance_type == "feature":
        return feature_importance
    elif importance_type == "structure":
        return structural_importance
    else:
        # Normalize each importance metric to [0,1] range
        norm_feature = feature_importance / (torch.max(feature_importance, dim=1, keepdim=True)[0] + 1e-8)
        norm_structure = structural_importance / (torch.max(structural_importance, dim=1, keepdim=True)[0] + 1e-8)
        # Combine both signals
        return norm_feature + norm_structure

def efficient_graph_reduction(node_features, edge_features, top_k_ratio=0.25, 
                              importance_type="combined", return_indices=False):
    """
    Efficiently reduce graph size based on node importance.
    
    Args:
        node_features: Tensor of shape [batch_size, num_nodes, feature_dim]
        edge_features: Tensor of shape [batch_size, num_nodes, num_nodes, edge_dim]
        top_k_ratio: Ratio of nodes to keep (0-1)
        importance_type: Method to calculate importance ("feature", "structure", "combined")
        return_indices: Whether to return indices of kept nodes
        
    Returns:
        Reduced node features, edge features, and mask
    """
    batch_size, num_nodes, feature_dim = node_features.shape
    edge_dim = edge_features.shape[-1]
    device = node_features.device
    
    # Calculate importance efficiently
    importance = fast_node_importance(node_features, edge_features, importance_type)
    
    # Determine k - how many nodes to keep
    k = max(1, int(num_nodes * top_k_ratio))
    
    # Select top-k important nodes for each batch item
    _, top_indices = torch.topk(importance, k, dim=1)
    
    # Create storage for reduced graph
    reduced_nodes = torch.zeros(batch_size, k, feature_dim, device=device)
    reduced_edges = torch.zeros(batch_size, k, k, edge_dim, device=device)
    
    # Extract the subgraph efficiently using advanced indexing
    for b in range(batch_size):
        # Get indices for this batch item
        idx = top_indices[b]
        
        # Extract node features - direct indexing
        reduced_nodes[b] = node_features[b, idx]
        
        # Extract edge features - needs double indexing
        # This is the slowest part - using einsum for efficiency
        for i, src_idx in enumerate(idx):
            for j, tgt_idx in enumerate(idx):
                reduced_edges[b, i, j] = edge_features[b, src_idx, tgt_idx]
    
    # Create mask from reduced edges
    mask = (reduced_edges.sum(dim=-1, keepdim=True) != 0)
    
    if return_indices:
        return reduced_nodes, reduced_edges, mask, top_indices
    return reduced_nodes, reduced_edges, mask

def hybrid_graph_reduction(node_features, edge_features, top_k_ratio=0.25, cluster_factor=None):
    """
    Hybrid approach: Important node selection + optional clustering.
    First reduces through importance, then optionally clusters similar nodes.
    
    Args:
        node_features: Original node features
        edge_features: Original edge features
        top_k_ratio: Ratio of nodes to keep via importance
        cluster_factor: Optional secondary reduction via clustering
        
    Returns:
        Reduced node and edge features
    """
    # First pass: Keep important nodes
    reduced_nodes, reduced_edges, mask = efficient_graph_reduction(
        node_features, edge_features, top_k_ratio
    )
    
    # Optional second pass: Cluster similar nodes
    if cluster_factor is not None and cluster_factor < 1.0:
        # Simple greedy clustering based on feature similarity
        batch_size, num_nodes, feature_dim = reduced_nodes.shape
        target_nodes = max(1, int(num_nodes * cluster_factor))
        
        # Skip if we would keep all nodes
        if target_nodes >= num_nodes:
            return reduced_nodes, reduced_edges, mask
            
        clustered_nodes = torch.zeros(batch_size, target_nodes, feature_dim, device=reduced_nodes.device)
        clustered_edges = torch.zeros(batch_size, target_nodes, target_nodes, reduced_edges.shape[-1], 
                                     device=reduced_edges.device)
        clustered_mask = torch.zeros(batch_size, target_nodes, target_nodes, 1, 
                                    dtype=mask.dtype, device=mask.device)
        
        for b in range(batch_size):
            # Simple greedy clustering
            # Calculate pairwise distances
            pairwise_dist = torch.cdist(reduced_nodes[b], reduced_nodes[b])
            
            # Initialize clusters with first node
            cluster_assignments = torch.full((num_nodes,), -1, dtype=torch.long, device=reduced_nodes.device)
            cluster_centers = []
            
            # Add most distant nodes as centers
            first_center = 0
            cluster_centers.append(first_center)
            cluster_assignments[first_center] = 0
            
            # Greedy algorithm to find diverse centers
            while len(cluster_centers) < target_nodes:
                # Find node with maximum distance to nearest center
                min_dists = float('inf') * torch.ones(num_nodes, device=reduced_nodes.device)
                for c in cluster_centers:
                    min_dists = torch.minimum(min_dists, pairwise_dist[:, c])
                
                # Skip already assigned nodes
                min_dists[cluster_assignments >= 0] = -1
                
                # Get next center
                if (min_dists > 0).any():
                    next_center = torch.argmax(min_dists).item()
                    cluster_centers.append(next_center)
                    cluster_assignments[next_center] = len(cluster_centers) - 1
                else:
                    # No more nodes to assign as centers
                    break
            
            # Assign remaining nodes to nearest center
            for i in range(num_nodes):
                if cluster_assignments[i] < 0:  # Not a center
                    dists = pairwise_dist[i, cluster_centers]
                    nearest = torch.argmin(dists).item()
                    cluster_assignments[i] = nearest
            
            # Average nodes in each cluster
            for c in range(len(cluster_centers)):
                cluster_mask = (cluster_assignments == c)
                if cluster_mask.sum() > 0:
                    # Average node features
                    clustered_nodes[b, c] = reduced_nodes[b, cluster_mask].mean(dim=0)
                    
                    # Sum edges between clusters
                    for c2 in range(len(cluster_centers)):
                        cluster_mask2 = (cluster_assignments == c2)
                        # Extract all edges between clusters c and c2
                        c_to_c2_edges = reduced_edges[b, cluster_mask][:, cluster_mask2]
                        if c_to_c2_edges.numel() > 0:
                            clustered_edges[b, c, c2] = c_to_c2_edges.mean(dim=(0, 1))
                            if clustered_edges[b, c, c2].sum() > 0:
                                clustered_mask[b, c, c2] = True
        
        return clustered_nodes, clustered_edges, clustered_mask
    
    return reduced_nodes, reduced_edges, mask

# Apply the reduction within your GraphConstructor.forward method
def apply_reduction_to_graph(node_features, edge_features, mask, reduction_ratio=0.25):
    """
    Wrapper function to apply the reduction strategy.
    """
    return efficient_graph_reduction(node_features, edge_features, reduction_ratio)