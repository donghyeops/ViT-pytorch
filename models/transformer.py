import torch
import torch.nn as nn


class MsaBlock(nn.Module):
    def __init__(self, input_dim=512, n_head=4):
        super().__init__()
        self.D = input_dim
        self.D_h = input_dim // n_head
        self.N_HEAD = n_head
        assert input_dim % n_head == 0
        self.linear_bottom = nn.Linear(self.D, self.N_HEAD * self.D_h * 3)
        self.linear_top = nn.Linear(self.N_HEAD * self.D_h, self.D)

    def forward(self, x):
        x = self.self_attention(x)
        x = self.linear_top(x)
        return x

    def self_attention(self, x):
        BS, N, _ = x.shape
        qkv = self.linear_bottom(x)
        q, k, v = torch.split(qkv, self.N_HEAD * self.D_h, dim=-1)  # 3 * (BS, N, N_HEAD * D_h)
        q = q.view(BS, N, self.N_HEAD, self.D_h)
        k = k.view(BS, N, self.N_HEAD, self.D_h)
        v = v.view(BS, N, self.N_HEAD, self.D_h)

        A = torch.softmax(q.matmul(k.permute(0, 1, 3, 2)) / (self.D_h ** 0.5), dim=-1)  # (BS, N, N_HEAD, N_HEAD)
        out = A.matmul(v)  # (N, BS, N_HEAD, D_h)
        return out.view(BS, N, -1)  # (N, BS, N_HEAD * D_h)


class LayerNormResidual(nn.Module):
    def __init__(self, dim, preprocess):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.preprocess = preprocess

    def forward(self, x):
        x_in = self.preprocess(x)
        x_in = self.layer_norm(x_in)
        return x_in + x


class MsaMLP(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.linear(x)


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, L=4, D=512):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(L):
            self.blocks.append(
                nn.Sequential(
                    LayerNormResidual(dim=D, preprocess=MsaBlock(input_dim=D, n_head=L)),
                    LayerNormResidual(dim=D, preprocess=MsaMLP(dim=D))
                )
            )

    def forward(self, x):
        for layer in self.blocks:
            x = layer(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, patch_size=16, vector_size=512, image_size=224, mha_length=4):
        super().__init__()
        self.P = patch_size
        self.D = vector_size
        self.L = mha_length
        assert image_size % patch_size == 0, f'Image_Size({image_size}) must be multiple of PATCH_SIZE({patch_size})'
        self.N = (image_size // patch_size) ** 2
        self.class_embed = nn.Parameter(torch.FloatTensor(self.D), requires_grad=True)
        self.pose_embed = nn.Parameter(torch.FloatTensor(self.N + 1, self.D), requires_grad=True)
        self.linear_projector = nn.Sequential(
            nn.Linear(3 * self.P * self.P, self.D)
        )
        self.msa = MultiHeadedSelfAttention(L=self.L, D=self.D)
        self.out_ln = nn.LayerNorm(self.D)

    def forward(self, x):
        bs, c, h, w = x.shape
        x = self.slice_patch(x)  # (BS, N, C, P, P)
        x = torch.flatten(x, start_dim=2)  # (BS, N, C*P*P)
        patch_embed = self.linear_projector(x)  # (BS, N, D)
        class_embed = torch.repeat_interleave(self.class_embed.unsqueeze(0), repeats=bs, dim=0)  # (BS, D)
        class_embed.unsqueeze_(1)  # (BS, 1, D)
        input_embed = torch.cat([class_embed, patch_embed], dim=1) + self.pose_embed.unsqueeze(0)
        out = self.msa(input_embed)
        out = self.out_ln(out[:, 0])
        return out

    def slice_patch(self, x):
        patches = x.unfold(1, 3, 3).unfold(2, self.P, self.P).unfold(3, self.P, self.P)  # (BS, 1, n, n, C, P, P)
        patches = torch.flatten(patches, 1, 3)  # (BS, N, C, P, P)
        return patches


if __name__ == '__main__':
    model = VisionTransformer(patch_size=16, vector_size=512)
    input_data = torch.FloatTensor(1, 3, 224, 224)
    print('input:', input_data.shape)

    output = model(input_data)
    print('output:', output.shape)
