import torch
import torch.nn as nn
import math


class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        x = x.long()
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, seq_len: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.positional_encoding = torch.zeros(seq_len, d_model)
        positions = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # seq_len x 1
        # div_term = 1 x d_model/2
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000)/self.d_model))
        self.positional_encoding[:, 0::2] = torch.sin(positions * div_term)
        self.positional_encoding[:, 1::2] = torch.cos(positions * div_term)
        self.positional_encoding = self.positional_encoding.unsqueeze(0)
        self.register_buffer("pe", self.positional_encoding)


    def forward(self, x):
        x =  x + (self.positional_encoding[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, epsilon=10**-6):
        super().__init__()
        self.epsilon = epsilon
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x_mean = x.mean(dim=-1, keepdim=True)
        x_std = x.std(dim=-1, keepdim=True)
        x = self.gamma * (x-x_mean)/(x_std + self.epsilon) + self.beta
        return x

class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_layer_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_layer_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_layer_2(self.dropout(torch.relu(self.linear_layer_1(x))))


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head: int, d_model: int, d_q:int, d_k:int, d_v:int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.d_q = d_q
        self.d_k = d_k
        self.d_v = d_v
        self.num_head = num_head
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)
        assert d_q == d_k, "d_q and d_k must be equal"
        assert d_k % num_head == 0, "d_k must be divisible by num_head"
        assert d_v % num_head == 0, "d_v must be divisible by num_head"
        self.d_k_per_head = d_k // num_head
        self.d_q_per_head = d_q // num_head
        self.d_v_per_head = d_v // num_head
        self.W_q = nn.Parameter(torch.rand(d_model, d_q))
        self.W_k = nn.Parameter(torch.rand(d_model, d_k))
        self.W_v = nn.Parameter(torch.rand(d_model, d_v))
        self.W_o = nn.Parameter(torch.rand(d_v, d_model))


    @staticmethod
    def attention(query, key, value, mask, dropout):
        # Batch_len x  num_head x seq_len x d_k_per_head
        d_k = query.shape[-1]
        attention_scores = query @ key.transpose(-2, -1) / math.sqrt(d_k) # Batch_len x  num_head x seq_len x seq_len
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1)
        # Batch_len x  num_head x seq_len x seq_len
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        weighted_values = attention_scores @ value # Batch_len x  num_head x seq_len x d_v_per_head
        return weighted_values, attention_scores


    def forward(self, q, k, v, mask):
        query = q @ self.W_q # (batch, seq_len, d_model) -> (batch, seq_len, d_q)
        key = k @ self.W_k # (batch, seq_len, d_model) -> (batch, seq_len, d_k)
        value = v @ self.W_v # (batch, seq_len, d_model) -> (batch, seq_len, d_v)

        query = query.view(query.shape[0], query.shape[1], self.num_head, self.d_q_per_head).transpose(1, 2)
        # (batch, seq_len, d_q) -> (batch, seq_len, num_head, d_q_per_head) -> (batch, num_head, seq_len, d_q_per_head)
        key = key.view(key.shape[0], key.shape[1], self.num_head, self.d_k_per_head).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_head, self.d_v_per_head).transpose(1, 2)
        # weighted_values : # Batch_len x  num_head x seq_len x d_v_per_head
        # attention_scores: # Batch_len x  num_head x seq_len x seq_len
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout_layer)
        # Batch_len x  num_head x seq_len x d_v_per_head -> Batch_len x  seq_len x num_head  x d_v_per_head ->
        # Batch_len x  seq_len x d_v
        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], self.num_head * self.d_v_per_head)
        x  = x @ self.W_o
        return x


class ResidualConnection(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForwardLayer, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers:nn.ModuleList) -> None:
        super().__init__()
        self.layers =  layers
        self.norm = LayerNormalization()


    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention,
                 feedforward_block: FeedForwardLayer, dropout:float)-> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feedforward_block = feedforward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])


    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feedforward_block)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, encoder_output, src_mask, x, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.projection(x), dim = -1)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding,
                 src_pos_embed: PositionalEncoding, tgt_embed: InputEmbedding,
                 tgt_pos_embed: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.src_pos_embed = src_pos_embed
        self.tgt_embed = tgt_embed
        self.tgt_pos_embed = tgt_pos_embed
        self.projection = projection_layer


    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos_embed(src)
        src = self.encoder(src, src_mask)
        return src


    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos_embed(tgt)
        tgt = self.decoder(encoder_output, src_mask, tgt, tgt_mask)
        return tgt

    def project(self, x):
        return self.projection(x)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int,
                      d_model: int = 512, d_q: int= 512, d_k: int = 512, d_v: int = 512, h: int = 8, N: int = 6,
                      d_ff: int=2048, dropout: float = 0.1) -> Transformer:
    # embedding layer
    src_embedding = InputEmbedding(src_vocab_size, d_model)
    tgt_embedding = InputEmbedding(tgt_vocab_size, d_model)

    # create positional embedding
    src_pos_embedding = PositionalEncoding(src_seq_len, d_model, dropout)
    tgt_pos_embedding = PositionalEncoding(tgt_seq_len, d_model, dropout)

    # create encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttention(h, d_model, d_q, d_k, d_v, dropout)
        encoder_feed_forward_block = FeedForwardLayer(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, encoder_feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttention(h, d_model, d_q, d_k, d_v, dropout)
        decoder_cross_attention_block = MultiHeadAttention(h, d_model, d_q, d_k, d_v, dropout)
        decoder_feed_forward_block = FeedForwardLayer(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block,
                                     decoder_feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_embedding, src_pos_embedding, tgt_embedding, tgt_pos_embedding,
                              projection_layer)

    # Initializing the transformer
    for param in transformer.parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
    return transformer
