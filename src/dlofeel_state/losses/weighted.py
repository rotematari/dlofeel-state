import torch
from torch import nn

from dlofeel_state.losses.der import compute_edges, compute_edge_lengths, compute_curvature_binormal
import numpy as np
from scipy.stats import norm

#import mse 
from torch.nn.functional import mse_loss

## loss = w1*mse(nodes)+ w2*mse(edges//lengths) + w3*mse(lengths) + w4*mse(kb)

class WeightedLoss(nn.Module):
    def __init__(
        self,
        w_nodes=1.0,
        w_edges=1.0,
        w_lengths=1.0,
        w_kb=0.01,
        eps=1e-8,
    ):
        super().__init__()
        self.w_nodes = w_nodes
        self.w_edges = w_edges
        self.w_lengths = w_lengths
        self.w_kb = w_kb
        self.eps = eps
        self.mse_loss = nn.MSELoss()

    def forward(self, preds, tgts, return_components=False):

        pred_edges = compute_edges(preds)
        tgt_edges = compute_edges(tgts)

        pred_lengths = compute_edge_lengths(pred_edges)
        tgt_lengths = compute_edge_lengths(tgt_edges)

        pred_dirs = pred_edges / pred_lengths.unsqueeze(2).clamp_min(self.eps)
        tgt_dirs = tgt_edges / tgt_lengths.unsqueeze(2).clamp_min(self.eps)

        pred_kb = compute_curvature_binormal(pred_edges, pred_lengths)
        tgt_kb = compute_curvature_binormal(tgt_edges, tgt_lengths)

        
        nodes_mse = self.mse_loss(preds, tgts)
        # nodes_mse = self.normal_weight_distance(preds, tgts)
        edges_mse = self.mse_loss(pred_dirs, tgt_dirs)
        lengths_mse = self.mse_loss(pred_lengths, tgt_lengths)
        kb_mse = self.mse_loss(pred_kb, tgt_kb)


        total = self.w_nodes*nodes_mse + self.w_lengths*lengths_mse  + self.w_edges*edges_mse +self.w_kb*kb_mse

        if return_components:
            # Pre-weight per-term MSEs, for logging an ablation breakdown.
            return total, {
                "nodes_mse": nodes_mse,
                "edges_mse": edges_mse,
                "lengths_mse": lengths_mse,
                "kb_mse": kb_mse,
            }
        return total


    def normal_weight_distance(self, preds, tgts):
        
        # normal disterbuted weight for nodes
        mu = 0      # Mean
        sigma = 0.5   # Standard Deviation
        num_pts = preds.shape[1]
        x = np.linspace(-1, 1, num_pts)

        # Get the PDF values
        pdf_values = norm.pdf(x, loc=mu, scale=sigma)

        # print(pdf_values)
        # Normalize the PDF values to sum to 1
        pdf_values /= np.sum(pdf_values)
        
        pdf_values_tensor = torch.tensor(pdf_values, dtype=torch.float32).to(preds.device)
        pdf_values_tensor = pdf_values_tensor.view(1, -1, 1)
        weighted_mse = torch.sum((preds - tgts).pow(2)*pdf_values_tensor,dim = [1,2]).mean()
        
        return weighted_mse

def compute_weighted_loss(
    preds,
    tgts,
    w_nodes=1.0,
    w_edges=1.0,
    w_lengths=1.0,
    w_kb=1.0,
    eps=1e-8,
):
    return WeightedLoss(
        w_nodes=w_nodes,
        w_edges=w_edges,
        w_lengths=w_lengths,
        w_kb=w_kb,
        eps=eps,
    )(preds, tgts)
    

if __name__ == "__main__":
    


    # Define parameters
    mu = 0      # Mean
    sigma = 0.5   # Standard Deviation
    x = np.linspace(-1, 1, 32)

    # Get the PDF values
    pdf_values = norm.pdf(x, loc=mu, scale=sigma)

    print(pdf_values)
    # Normalize the PDF values to sum to 1
    pdf_values /= np.mean(pdf_values)
    print("Normalized PDF values:", pdf_values)
    # sum of pdf values
    print("Sum of PDF values:", np.sum(pdf_values))