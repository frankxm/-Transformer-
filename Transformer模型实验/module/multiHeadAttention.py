from torch.nn import Module
import torch
import math
import torch.nn.functional as F

# h为什么要分多个头？
# 每个头可以独立地学习不同的上下文信息。比如，一个头可能专注于捕捉单词间的语法关系，另一个头则可能专注于捕捉语义上的联系。
# 通过多头并行化，Transformer 能够在不同的“子空间”中并行地学习信息。

# 查询（Query，Q）：我们希望关注的信息 希望获取的内容（通常是当前词或token的嵌入）。
# 键（Key，K）：可以用于与查询对比的信息，帮助决定如何权重地关注其他词的表示。
# 值（Value，V）：实际从中获取的信息，最终的加权信息，最终影响注意力的输出
# 多头注意力机制
# 假设我们有 h 个注意力头，每个头的维度是 d_model / h。每个头都会计算自己的查询（Q）、键（K）和值（V）。这些头的输出会被拼接（concatenate），然后通过一个线性层映射到输出的维度。
class MultiHeadAttention(Module):
    def __init__(self,
                 d_model: int,
                 q: int,
                 v: int,
                 h: int,
                 device: str,
                 mask: bool=False,
                 dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        # d_model（模型的总维度）被分配到多个头（h）中。每个头都有一个查询、键和值，维度分别是 q 和 v。因此，Q、K 和 V 的维度为 q * h 和 v * h，这样可以并行计算多个头的注意力。通常，q = v 以保证每个头的输出维度一致。
        self.W_q = torch.nn.Linear(d_model, q * h)
        self.W_k = torch.nn.Linear(d_model, q * h)
        self.W_v = torch.nn.Linear(d_model, v * h)

        self.W_o = torch.nn.Linear(v * h, d_model)

        self.device = device
        self._h = h
        self._q = q

        self.mask = mask
        self.dropout = torch.nn.Dropout(p=dropout)
        self.score = None

    def forward(self, x, stage):
        # x 是输入的特征，形状为 [batch_size, seq_len, d_model]，其中 d_model 是每个词的表示维度。
        # d_model 是模型的维度大小，通常是一个超参数，表示词嵌入的维度。
        # self.W_q, self.W_k, self.W_v 是三个线性变换层，用来生成查询（Q），键（K），值（V）：
        # x 是输入数据，Q、K 和 V 都是从 x 通过线性变换得到的。W_q、W_k 和 W_v 是对应的权重矩阵，通过这些矩阵将输入 x 转换为查询（Q）、键（K）和值（V）。所以，Q、K 和 V 都来源于同一个输入 x，这就是典型的 自注意力机制。
        # Q、K 和 V 分成多个头（h个头）。这通过 chunk(self._h, dim=-1) 来实现，即将 Q、K、V 沿着最后一个维度（d_model）拆分成 h 个部分。
        Q = torch.cat(self.W_q(x).chunk(self._h, dim=-1), dim=0)
        K = torch.cat(self.W_k(x).chunk(self._h, dim=-1), dim=0)
        V = torch.cat(self.W_v(x).chunk(self._h, dim=-1), dim=0)

        # 其中，Q 和 K 用于计算注意力分数（score），V 用于加权求和得到最终的注意力输出。
        # 其中 Q 和 K 是通过内积来计算的，相当于每个词对其他词的注意力权重。为了避免数值过大，结果被除以 sqrt(q)。
        score = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self._q)
        self.score = score
        # 训练时使用一个 遮挡矩阵。遮挡的作用是为了防止在自注意力计算时，某个位置（词汇）依赖于后面的位置，保证序列中的信息传递顺序
        if self.mask and stage == 'train':
            mask = torch.ones_like(score[0])
            mask = torch.tril(mask, diagonal=0)
            score = torch.where(mask > 0, score, torch.Tensor([-2**32+1]).expand_as(score[0]).to(self.device))

        score = F.softmax(score, dim=-1)

        attention = torch.matmul(score, V)

        attention_heads = torch.cat(attention.chunk(self._h, dim=0), dim=-1)

        self_attention = self.W_o(attention_heads)

        return self_attention, self.score