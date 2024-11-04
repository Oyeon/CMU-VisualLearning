# Credit to the CS-231n course at Stanford, from which this assignment is adapted
import numpy as np
import copy
import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class AttentionLayer(nn.Module):

    def __init__(self, embed_dim, dropout=0.1):
       
        super().__init__()
        self.embed_dim = embed_dim
        # Initialize the following layers and parameters to perform attention
        # This class assumes that the input dimension for query, key and value is embed_dim
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
            
    def forward(self, query, key, value, attn_mask=None):
        N, S, D = query.shape
        N, T, D = value.shape
        assert key.shape == value.shape
       
        # Compute attention 
    
        # Project query, key, and value
        query = self.query_proj(query)  # Shape: (N, S, D)
        key = self.key_proj(key)        # Shape: (N, T, D)
        value = self.value_proj(value)  # Shape: (N, T, D)

        # Compute dot-product attention with scaling
        scaling = float(self.embed_dim) ** -0.5
        dot_product = torch.bmm(query, key.transpose(1, 2)) * scaling  # Shape: (N, S, T)

        if attn_mask is not None:
            # Convert attn_mask (multiplicative) to an additive mask
            # If mask[i,j] = 0, we set additive_mask[i,j] to a large negative value
            additive_mask = (1.0 - attn_mask) * -1e9  # Shape: (N, S, T)
            dot_product += additive_mask
        
        # Apply softmax, dropout, and use value
        attn_weights = F.softmax(dot_product, dim=-1)  # Shape: (N, S, T)
        attn_weights = self.dropout(attn_weights)
        y = torch.bmm(attn_weights, value)  # Shape: (N, S, D)
        return y  

class MultiHeadAttentionLayer(AttentionLayer):

    def __init__(self, embed_dim, num_heads, dropout=0.1):
       
        super().__init__(embed_dim, dropout)
        self.num_heads = num_heads

        # Initialize layers and parameters for multi-head attention
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        H = self.num_heads
        N, S, D = query.shape
        N, T, D = value.shape
        assert key.shape == value.shape

        # Project query, key, and value and split into multiple heads
        query = self.query_proj(query).view(N, S, H, self.head_dim).transpose(1, 2)  # (N, H, S, D/H)
        key = self.key_proj(key).view(N, T, H, self.head_dim).transpose(1, 2)        # (N, H, T, D/H)
        value = self.value_proj(value).view(N, T, H, self.head_dim).transpose(1, 2)  # (N, H, T, D/H)

        # Compute scaled dot-product attention for each head
        scaling = float(self.head_dim) ** -0.5
        dot_product = torch.matmul(query, key.transpose(-2, -1)) * scaling  # (N, H, S, T)

        if attn_mask is not None:
            # Convert attn_mask (multiplicative) to an additive mask
            additive_mask = (1.0 - attn_mask.unsqueeze(1)) * -1e9  # (N, 1, S, T)
            dot_product += additive_mask
        
        # Apply softmax, dropout, and compute attention output
        attn_weights = F.softmax(dot_product, dim=-1)  # (N, H, S, T)
        attn_weights = self.dropout(attn_weights)
        y = torch.matmul(attn_weights, value)  # (N, H, S, D/H)

        # Concatenate heads and project to output dimension
        y = y.transpose(1, 2).contiguous().view(N, S, D)  # (N, S, D)
        output = self.out_proj(y)  # (N, S, D)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        # Use torch.nn.Embedding to create the encoding. Initialize dropout layer.
        self.encoding = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Initialize positional encodings
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.encoding.weight = nn.Parameter(pe, requires_grad=False)
      
    def forward(self, x):
        N, S, D = x.shape
        # Add the encoding to x
        positions = torch.arange(0, S, device=x.device).unsqueeze(0).expand(N, S)  # (N, S)
        output = x + self.encoding(positions)
        output = self.dropout(output)
   
        return output

class SelfAttentionBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        # Initialize layers using MultiHeadAttentionLayer for self_attn
        self.self_attn = MultiHeadAttentionLayer(input_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(input_dim)
       
    def forward(self, seq, mask):
        # Self-attention on the sequence using the mask
        attn_output = self.self_attn(seq, seq, seq, attn_mask=mask)  # (N, S, D)
        attn_output = self.dropout(attn_output)
        # Residual connection and layer normalization
        out = self.layernorm(seq + attn_output)
        return out

class CrossAttentionBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dropout=0.1):
        super().__init__()
        # Initialize layers using MultiHeadAttentionLayer for cross_attn
        self.cross_attn = MultiHeadAttentionLayer(input_dim, num_heads, dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)
       
    def forward(self, seq, cond):
        # Cross-attention on the sequence using conditioning features
        attn_output = self.cross_attn(seq, cond, cond)  # (N, S, D)
        attn_output = self.dropout(attn_output)
        # Residual connection and layer normalization
        out = self.norm(seq + attn_output)
        return out

class FeedForwardBlock(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1 ):
        super().__init__()
        # Initialize MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, input_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, seq):
        # MLP on the sequence with dropout
        mlp_output = self.mlp(seq)
        mlp_output = self.dropout(mlp_output)
        # Residual connection and layer normalization
        out = self.norm(seq + mlp_output)
        return out

class DecoderLayer(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1 ):
        super().__init__()
        self.self_atn_block = SelfAttentionBlock(input_dim, num_heads, dropout)
        self.cross_atn_block = CrossAttentionBlock(input_dim, num_heads, dropout)
        self.feedforward_block = FeedForwardBlock(input_dim, num_heads, dim_feedforward, dropout)

    def forward(self, seq, cond, mask):
        out = self.self_atn_block(seq, mask)
        out = self.cross_atn_block(out, cond)
        return self.feedforward_block(out)
       
class TransformerDecoder(nn.Module):
    def __init__(self, word_to_idx, idx_to_word, input_dim, embed_dim, num_heads=4,
                 num_layers=2, max_length=50, device='cuda'):
        """
        Construct a new TransformerDecoder instance.
        Inputs:
        - word_to_idx: A dictionary giving the vocabulary. It contains V entries.
          and maps each string to a unique integer in the range [0, V).
        - input_dim: Dimension of input image feature vectors.
        - embed_dim: Embedding dimension of the transformer.
        - num_heads: Number of attention heads.
        - num_layers: Number of transformer layers.
        - max_length: Max possible sequence length.
        """
        super().__init__()

        vocab_size = len(word_to_idx)
        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self.idx_to_word = idx_to_word
        
        self.layers = nn.ModuleList([DecoderLayer(embed_dim, num_heads) for _ in range(num_layers)])
        
        self.caption_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=self._null)
        self.positional_encoding = PositionalEncoding(embed_dim, max_len=max_length)
        self.feature_embedding = nn.Linear(input_dim, embed_dim)
        self.score_projection = nn.Linear(embed_dim, vocab_size) 

        self.apply(self._init_weights)
        self.device = device 
        self.to(device)

    def get_data_embeddings(self, features, captions):
        # Get caption embeddings and add positional encoding
        captions_embed = self.caption_embedding(captions)  # (N, T, D)
        captions_embed = self.positional_encoding(captions_embed)  # (N, T, D)

        # Unsqueeze feature embedding along dimension 1
        feature_embedding = self.feature_embedding(features).unsqueeze(1)  # (N, 1, D)
        return feature_embedding, captions_embed

    def get_causal_mask(self, _len):
        # Create a causal mask of shape (_len, _len)
        mask = torch.tril(torch.ones(_len, _len, device=self.device))
        return mask
                                          
    def forward(self, features, captions):
        """
        Given image features and caption tokens, return a distribution over the
        possible tokens for each timestep. Note that since the entire sequence
        of captions is provided all at once, we mask out future timesteps.
        Inputs:
         - features: image features, of shape (N, D)
         - captions: ground truth captions, of shape (N, T)
        Returns:
         - scores: score for each token at each timestep, of shape (N, T, V)
        """
        features_embed, captions_embed = self.get_data_embeddings(features, captions)
        mask = self.get_causal_mask(captions_embed.shape[1]).to(captions_embed.device)
        mask = mask.unsqueeze(0)  # (1, T, T)
        
        output = captions_embed
        for layer in self.layers:
            output = layer(output, features_embed, mask=mask)

        scores = self.score_projection(output)
        return scores

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def sample(self, features, max_length=30):
        """
        Given image features, use greedy decoding to predict the image caption.
        Inputs:
         - features: image features, of shape (N, D)
         - max_length: maximum possible caption length
        Returns:
         - captions: captions for each example, of shape (N, max_length)
        """
        with torch.no_grad():
            features = torch.Tensor(features).to(self.device)
            N = features.shape[0]

            # Create an empty captions tensor (where all tokens are NULL).
            captions = self._null * np.ones((N, max_length), dtype=np.int32)

            # Create a partial caption, with only the start token.
            partial_caption = self._start * np.ones(N, dtype=np.int32)
            partial_caption = torch.LongTensor(partial_caption).to(self.device)
            # [N] -> [N, 1]
            partial_caption = partial_caption.unsqueeze(1)

            for t in range(max_length):

                # Predict the next token (ignoring all other time steps).
                output_logits = self.forward(features, partial_caption)
                output_logits = output_logits[:, -1, :]

                # Choose the most likely word ID from the vocabulary.
                # [N, V] -> [N]
                word = torch.argmax(output_logits, axis=1)

                # Update our overall caption and our current partial caption.
                captions[:, t] = word.cpu().numpy()
                word = word.unsqueeze(1)
                partial_caption = torch.cat([partial_caption, word], dim=1)

            return captions
