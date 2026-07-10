import torch
import torch.nn as nn
import torch.nn.functional as F
class DualStreamAdditiveAttention(nn.Module):
    """
    Efficient Additive Attention for 3D Hand Reconstruction.
    Input: tensor in shape [B, N, C]  (B=batch, N=num_joints, C=channels)
    Output: tensor in shape [B, N, C]
    """

    def __init__(self, in_dims=256, num_heads=1):
        super().__init__()
        self.in_dims = in_dims
        self.num_heads = num_heads
        self.head_dims = in_dims // num_heads

        assert in_dims % num_heads == 0, f"in_dims {in_dims} must be divisible by num_heads {num_heads}"

        self.to_query = nn.Linear(in_dims, in_dims)
        self.to_key = nn.Linear(in_dims, in_dims)
        if self.head_dims == 256:
            self.num_tokens = 21
        elif self.head_dims == 128:
            self.num_tokens = 84
        else:
            self.num_tokens = 336

        self.w_g = nn.Parameter(torch.randn(num_heads, self.head_dims, 1))
        self.w_g2 = nn.Parameter(torch.randn(num_heads, self.num_tokens, 1))
        self.scale_factor = self.head_dims ** -0.5

        self.proj = nn.Linear(in_dims, in_dims)
        self.final = nn.Linear(in_dims, in_dims)

        self.norm1 = nn.LayerNorm(in_dims)
        self.norm2 = nn.LayerNorm(in_dims)

    def forward(self, x):
        """
        Args:
            x: [B, N, C]
        Returns:
            [B, N, C]
        """
        B, N, C = x.shape

        query = self.to_query(x)
        key = self.to_key(x)

        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        query = query.view(B, N, self.num_heads, self.head_dims).transpose(1, 2)  # [B,H,N,D]
        key = key.view(B, N, self.num_heads, self.head_dims).transpose(1, 2)  # [B,H,N,D]

        query_weight = torch.matmul(query, self.w_g) * self.scale_factor
        query_weight2 = torch.matmul(query.transpose(3, 2)) * self.scale_factor
        A = torch.nn.functional.normalize(query_weight, dim=2)
        A2 = F.softmax(query_weight2,dim=3).transpose(3,2)


        G = torch.sum(A * query, dim=2)
        G2 = torch.sum(A2 * query, dim=3)

        G = G.unsqueeze(2)  # [B,H,1,D]
        G2 = G2.unsqueeze(3)  # [B,H,1,D]
        out = self.proj(((1+G2)*G * key).transpose(1, 2).reshape(B, N, C)) + query.transpose(1, 2).reshape(B, N, C)

        out = self.final(out)

        return out
