from torch.nn import Module
import torch
from torch.nn import ModuleList
# 明确指出 encoder 是与 transformer 位于同一包下。
# 无论从哪里运行（transformer.py 或 main.py），都不会出问题。
from .encoder import Encoder
import math
import torch.nn.functional as F


class Transformer(Module):
    def __init__(self,
                 d_model: int,
                 d_input: int,
                 d_channel: int,
                 d_output: int,
                 d_hidden: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 device: str,
                 dropout: float = 0.1,
                 pe: bool = False,
                 mask: bool = False):
        super(Transformer, self).__init__()

        self.encoder_list_1 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  mask=mask,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])

        self.encoder_list_2 = ModuleList([Encoder(d_model=d_model,
                                                  d_hidden=d_hidden,
                                                  q=q,
                                                  v=v,
                                                  h=h,
                                                  dropout=dropout,
                                                  device=device) for _ in range(N)])

        self.embedding_channel = torch.nn.Linear(d_channel, d_model)
        self.embedding_input = torch.nn.Linear(d_input, d_model)

        self.gate = torch.nn.Linear(d_model * d_input + d_model * d_channel, 2)
        self.output_linear = torch.nn.Linear(d_model * d_input + d_model * d_channel, d_output)

        self.pe = pe
        self._d_input = d_input
        self._d_model = d_model

    def forward(self, x, stage='train'):
        """
        前向传播
        :param x: 输入
        :param stage: 用于描述此时是训练集的训练过程还是测试集的测试过程  测试过程中均不在加mask机制
        :return: 输出，gate之后的二维向量，step-wise encoder中的score矩阵，channel-wise encoder中的score矩阵，step-wise embedding后的三维矩阵，channel-wise embedding后的三维矩阵，gate
        """
        # step-wise
        # score矩阵为 input， 默认加mask 和 pe
        encoding_1 = self.embedding_channel(x)
        input_to_gather = encoding_1
        # 位置编码的作用：在处理序列数据时，模型的自注意力机制无法区分输入的顺序，位置编码为每个输入的位置提供了一个唯一的标识，帮助模型学习到顺序信息。
        if self.pe:
            pe = torch.ones_like(encoding_1[0])
            # 生成位置索引  position 是一个张量，包含从 0 到 d_input-1 的位置索引
            position = torch.arange(0, self._d_input).unsqueeze(-1)
            # 计算正弦和余弦的缩放因子  temp 是一个通过 log 和 exp 转换的缩放因子，用来生成正弦和余弦波形的频率
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            # 计算位置编码
            temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
            # 偶数位置使用sin
            pe[:, 0::2] = torch.sin(temp)
            # 奇数位置使用cos
            pe[:, 1::2] = torch.cos(temp)
            # 将位置编码加到编码上
            encoding_1 = encoding_1 + pe

        for encoder in self.encoder_list_1:
            encoding_1, score_input = encoder(encoding_1, stage)

        # channel-wise
        # score矩阵为channel 默认不加mask和pe
        # x.transpose(-1, -2) 是一个转置操作，目的是将输入数据的 channel 和 input 维度进行交换。换句话说，如果原始 x 的形状是 [batch_size, d_input, d_channel]，转置后会变成 [batch_size, d_channel, d_input]。
        # self.embedding_input 将对 每个通道 的输入特征进行映射，将每个特征从 d_input 维度映射到 d_model 维度。
        encoding_2 = self.embedding_input(x.transpose(-1, -2))
        channel_to_gather = encoding_2

        for encoder in self.encoder_list_2:
            encoding_2, score_channel = encoder(encoding_2, stage)

        # 三维变二维  通常是 [batch_size, seq_len, d_model]）将它们 展平（reshape），变成二维张量（[batch_size, d_model * seq_len]），以便后面通过线性层处理。
        encoding_1 = encoding_1.reshape(encoding_1.shape[0], -1)
        encoding_2 = encoding_2.reshape(encoding_2.shape[0], -1)

        # gate 将 encoding_1 和 encoding_2 在最后一个维度上拼接。假设 encoding_1 和 encoding_2 都是形状为 [batch_size, d_model * seq_len] 的张量，拼接后会得到形状为 [batch_size, d_model * seq_len * 2] 的张量。也就是说，它们的特征维度加倍了。
        # 通过 self.gate 这一线性层，我们将拼接后的张量输入到一个全连接层（Linear）。它的作用是计算一个二维向量（形状为 [batch_size, 2]），每个元素表示 encoding_1 和 encoding_2 各自的权重。
        # F.softmax()：对 self.gate 的输出应用 softmax 操作，使得输出的两个数值表示对 encoding_1 和 encoding_2 的概率分布（即加权系数，范围为 [0, 1]）
        gate = F.softmax(self.gate(torch.cat([encoding_1, encoding_2], dim=-1)), dim=-1)
        # 将加权后的 encoding_1 和 encoding_2 在最后一个维度上拼接起来，得到一个新的编码 encoding。最终，encoding 的形状是 [batch_size, d_model * seq_len * 2]，代表加权组合后的特征。
        encoding = torch.cat([encoding_1 * gate[:, 0:1], encoding_2 * gate[:, 1:2]], dim=-1)

        # 输出
        output = self.output_linear(encoding)
        # tensor([[ 0.3892,  0.4223, -0.2670,  ...,  0.0346,  0.0584, -0.1309],
        #         [ 0.1937,  0.0838, -0.2206,  ...,  0.1389,  0.1232, -0.0893],
        #         [ 0.3945,  0.2716, -0.5189,  ...,  0.3526, -0.0309,  0.5403],
        #         ...,
        #         [-0.0351,  0.3012, -0.5427,  ..., -0.1491,  0.1623, -0.1757],
        #         [ 0.2976, -0.0415, -0.3020,  ..., -0.3094,  0.2548,  0.0324],
        #         [ 0.2194,  0.0476, -0.4809,  ..., -0.3580, -0.1419, -0.1354]],
        return output, encoding, score_input, score_channel, input_to_gather, channel_to_gather, gate
