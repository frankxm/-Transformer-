from torch.nn import Module
import torch

from .feedForward import FeedForward
from .multiHeadAttention import MultiHeadAttention

class Encoder(Module):
    def __init__(self,
                 d_model: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool = False,
                 dropout: float = 0.1):
        super(Encoder, self).__init__()

        self.MHA = MultiHeadAttention(d_model=d_model, q=q, v=v, h=h, mask=mask, device=device, dropout=dropout)
        self.feedforward = FeedForward(d_model=d_model, d_hidden=d_hidden)
        self.dropout = torch.nn.Dropout(p=dropout)
        # Layer Normalization 是对每一层的输出进行归一化，使其保持在适当的范围内，从而加速训练和提高稳定性。
        self.layerNormal_1 = torch.nn.LayerNorm(d_model)
        self.layerNormal_2 = torch.nn.LayerNorm(d_model)

    def forward(self, x, stage):

        residual = x
        x, score = self.MHA(x, stage)
        # 对多头注意力的输出应用 dropout。这样可以在训练时防止过拟合。
        x = self.dropout(x)
        # 残差连接（Residual Connection）：将原始输入 x 加到多头注意力的输出 x 上，然后进行归一化（Layer Normalization）。
        # 残差连接有助于避免梯度消失问题，使得训练时梯度能更容易地流动。通过 x + residual 保持信息流动的路径。
        x = self.layerNormal_1(x + residual)

        residual = x
        x = self.feedforward(x)
        x = self.dropout(x)
        x = self.layerNormal_2(x + residual)

        return x, score
