import torch
from torch import nn


def get_tokens_and_segments(tokens_a, tokens_b=None):
    tokens = ["<cls>"] + tokens_a + ["<sep>"]
    segments = [0] * (len(tokens_a) + 2)

    if tokens_b is not None:
        tokens += tokens_b + ["<sep>"]
        segments += [1] * (len(tokens_b) + 1)

    return tokens, segments


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()

        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)

        self.relu = nn.ReLU()

        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(X + self.dropout(Y))


class EncoderBlock(nn.Module):
    def __init__(self, hidden_size, ffn_num_hiddens, num_heads, dropout):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.addnorm1 = AddNorm(hidden_size, dropout)

        self.ffn = PositionWiseFFN(hidden_size, ffn_num_hiddens, hidden_size)

        self.addnorm2 = AddNorm(hidden_size, dropout)

    def forward(self, X, valid_lens=None):
        key_padding_mask = None

        if valid_lens is not None:
            batch_size, seq_len, _ = X.shape

            positions = torch.arange(seq_len, device=X.device)

            key_padding_mask = positions.unsqueeze(0) >= valid_lens.unsqueeze(1)

        Y, _ = self.attention(
            X, X, X, key_padding_mask=key_padding_mask, need_weights=False
        )

        X = self.addnorm1(X, Y)

        Y = self.ffn(X)

        return self.addnorm2(X, Y)


class BERTEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        ffn_num_hiddens,
        num_heads,
        num_layers,
        dropout,
        max_len=1000,
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        self.segment_embedding = nn.Embedding(2, hidden_size)
        # 位置参数，每个batch共享，毕竟只是存位置信息
        # 用来让模型自己学位置之间的大体关系，比如句首，句中，句尾
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, hidden_size))

        self.blks = nn.ModuleList(
            [
                EncoderBlock(hidden_size, ffn_num_hiddens, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, tokens, segments, valid_lens=None):
        X = self.token_embedding(tokens)

        segment_X = self.segment_embedding(segments)

        X = X + segment_X + self.pos_embedding[:, : X.shape[1], :]

        for blk in self.blks:
            X = blk(X, valid_lens)

        return X


class MaskLM(nn.Module):
    def __init__(self, vocab_size, hidden_size, mlm_num_hiddens=None):
        super().__init__()

        if mlm_num_hiddens is None:
            mlm_num_hiddens = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlm_num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(mlm_num_hiddens),
            nn.Linear(mlm_num_hiddens, vocab_size),
        )

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]

        pred_positions = pred_positions.reshape(-1)

        batch_size = X.shape[0]

        batch_idx = torch.arange(batch_size, device=X.device)

        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)

        masked_X = X[batch_idx, pred_positions]

        masked_X = masked_X.reshape(batch_size, num_pred_positions, -1)

        mlm_Y_hat = self.mlp(masked_X)

        return mlm_Y_hat


class NextSentencePred(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()

        self.output = nn.Linear(num_inputs, 2)

    def forward(self, X):
        X = torch.flatten(X, start_dim=1)

        return self.output(X)


class BERTModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size=768,
        ffn_num_hiddens=3072,
        num_heads=12,
        num_layers=12,
        dropout=0.1,
        max_len=512,
        mlm_num_hiddens=None,
    ):
        super().__init__()

        self.encoder = BERTEncoder(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            ffn_num_hiddens=ffn_num_hiddens,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            max_len=max_len,
        )

        self.hidden = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh())

        self.mlm = MaskLM(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            mlm_num_hiddens=mlm_num_hiddens,
        )

        self.nsp = NextSentencePred(hidden_size)

    def forward(self, tokens, segments, valid_lens=None, pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)

        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None

        cls_X = encoded_X[:, 0, :]

        nsp_Y_hat = self.nsp(self.hidden(cls_X))

        return (encoded_X, mlm_Y_hat, nsp_Y_hat)


if __name__ == "__main__":
    vocab_size = 10000

    net = BERTModel(
        vocab_size=vocab_size,
        hidden_size=128,
        ffn_num_hiddens=256,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        max_len=64,
    )

    tokens = torch.randint(0, vocab_size, (2, 8))

    segments = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1]])

    valid_lens = torch.tensor([8, 7])

    pred_positions = torch.tensor([[1, 3, 5], [2, 4, 6]])

    encoded_X, mlm_Y_hat, nsp_Y_hat = net(tokens, segments, valid_lens, pred_positions)

    print("encoded_X:", encoded_X.shape)
    print("mlm_Y_hat:", mlm_Y_hat.shape)
    print("nsp_Y_hat:", nsp_Y_hat.shape)
