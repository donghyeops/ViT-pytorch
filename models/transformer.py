import torch
import torch.nn as nn


class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, dim=512, n_head=4):
        super().__init__()
        self.D = dim
        self.D_h = dim // n_head
        self.N_HEAD = n_head
        assert dim % n_head == 0
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


class MLP(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.GELU()
        )

    def forward(self, x):
        return self.layers(x)


class VisionTransformer(nn.Module):
    """
    Paper: https://arxiv.org/pdf/2010.11929.pdf
    """
    def __init__(self, patch_size=16, vector_size=512, image_size=224, mha_length=4):
        super().__init__()
        self.P = patch_size
        self.D = vector_size
        self.L = mha_length
        assert image_size % patch_size == 0, f'Image_Size({image_size}) must be multiple of PATCH_SIZE({patch_size})'
        self.N = (image_size // patch_size) ** 2
        self.class_embed = nn.Parameter(torch.FloatTensor(self.D), requires_grad=True)
        self.pose_embed = nn.Parameter(torch.FloatTensor(self.N + 1, self.D), requires_grad=True)
        self.linear_projector = nn.Linear(3 * self.P * self.P, self.D)

        self.MSA_modules = nn.ModuleList(
            [MultiHeadedSelfAttention(dim=self.D, n_head=4) for _ in range(self.L)]
        )
        self.LN_1_modules = nn.ModuleList(
            [nn.LayerNorm(self.D) for _ in range(self.L)]
        )
        self.LN_2_modules = nn.ModuleList(
            [nn.LayerNorm(self.D) for _ in range(self.L)]
        )
        self.MLP_modules = nn.ModuleList(
            [MLP(dim=self.D) for _ in range(self.L)]
        )
        self.LN_3 = nn.LayerNorm(self.D)

    def forward(self, x):
        """
         - B: batch_size
         - C: channel
         - H: height
         - W: width
         - P: patch_size (height & width of patch)
         - N: the number of patches
        """
        # 이미지 패치 임베딩
        x = self.slice_patch(x)  # (B, N, C, P, P)
        x = x.flatten(start_dim=2)  # (B, N, C*P*P)
        patch_embed = self.linear_projector(x)  # (B, N, D)

        # 클래스 토큰 임베딩
        B = x.shape[0]
        class_embed = torch.repeat_interleave(self.class_embed.unsqueeze(0), repeats=B, dim=0)  # (B, D)
        class_embed.unsqueeze_(1)  # (B, 1, D)

        # z_0 계산
        z_0 = torch.cat([class_embed, patch_embed], dim=1) + self.pose_embed.unsqueeze(0)

        # multi-headed self attention
        z = z_0
        for i in range(self.L):
            z_prime = self.MSA_modules[i](self.LN_1_modules[i](z)) + z
            z = self.MLP_modules[i](self.LN_2_modules[i](z_prime)) + z_prime
        y = self.LN_3(z[:, 0])
        return y

    def slice_patch(self, x):
        """
        입력된 이미지들을 P 크기의 패치로 자름
        :param x: (B, C, H, W)
        :return: (B, N, C, P, P)
        """
        # Tensor.unfold: https://pytorch.org/docs/stable/generated/torch.Tensor.unfold.html
        # x = x.unfold(1, 3, 3).unfold(2, self.P, self.P).unfold(3, self.P, self.P)
        # x = torch.flatten(x, 1, 3)  # (BS, N, C, P, P)
        # or
        x = x.unfold(dimension=2, size=self.P, step=self.P)  # (B, C, H, W) -> (B, C, C/H, W, P)
        x = x.unfold(dimension=3, size=self.P, step=self.P)  # (B, C, C/H, W, P) -> (B, C, H/P, W/P, P, P)
        x = x.flatten(start_dim=2, end_dim=3)  # (B, C, H/P, W/P, P, P) -> (B, C, N='H/P*W/P', P, P)
        x = x.permute(0, 2, 1, 3, 4)  # (B, C, N, P, P) -> (B, N, C, P, P)
        return x


if __name__ == '__main__':
    model = VisionTransformer(patch_size=16, vector_size=512)
    input_data = torch.FloatTensor(2, 3, 224, 224)
    print('input:', input_data.shape)
    output = model(input_data)
    print('output:', output.shape)
