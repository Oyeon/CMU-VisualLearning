import sys
import torch.nn as nn
import torch

sys.path.append("../transformer_captioning") 
from transformer import (
    AttentionLayer,
    MultiHeadAttentionLayer,
    PositionalEncoding,
    SelfAttentionBlock,
    CrossAttentionBlock,
    FeedForwardBlock
)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff=2048, dropout=0.1):
        super().__init__()
        self.self_attention = SelfAttentionBlock(d_model, num_heads, dropout=dropout)
        self.feed_forward = FeedForwardBlock(d_model, num_heads, d_ff, dropout=dropout)

    def forward(self, seq, mask):
        x = self.self_attention(seq, mask)
        x = self.feed_forward(x)
        return x

class ViT(nn.Module):
    """
        - A ViT takes an image as input, divides it into patches, and then feeds the patches through a transformer to output a sequence of patch embeddings. 
        - To perform classification with a ViT we patchify the image, embed each patch using an embedding layer and add a learnable [CLS] token to the beginning of the sequence.
        - The output embedding corresponding to the [CLS] token is then fed through a linear layer to obtain the logits for each class.
    """

    def __init__(self, patch_dim, d_model, d_ff, num_heads, num_layers, num_patches, num_classes, device='cuda'):
        """
            Construct a new ViT instance.
            Inputs:
            - patch_dim: the dimension of each patch
            - d_model: the dimension of the input to the transformer blocks
            - d_ff: the dimension of the intermediate layer in the feed forward block 
            - num_heads: the number of heads in the multi-head attention layer
            - num_layers: the number of transformer blocks
            - num_patches: the number of patches in the image
        """
        super().__init__()

        self.patch_dim = patch_dim
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_patches = num_patches
        self.num_classes = num_classes
        self.device = device

        # Linear layer that takes as input a patch and outputs a d_model dimensional vector
        self.patch_embedding = nn.Linear(patch_dim * patch_dim * 3, d_model)
        # Positional encoding from the transformer captioning solution
        self.positional_encoding = PositionalEncoding(d_model, max_len=num_patches + 1)
        # Takes the [CLS] token embedding and outputs logits for each class
        self.fc = nn.Linear(d_model, num_classes)
        # Learnable [CLS] token embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)])

        self.apply(self._init_weights)
        self.device = device 
        self.to(device)

    def patchify(self, images):
        """
            Given a batch of images, divide each image into patches and flatten each patch into a vector.
            Inputs:
                - images: a FloatTensor of shape (N, 3, H, W) giving a minibatch of images
            Returns:
                - patches: a FloatTensor of shape (N, num_patches, patch_dim x patch_dim x 3) giving a minibatch of patches    
        """
        # Break images into a grid of patches
        N, C, H, W = images.shape
        patch_size = self.patch_dim
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(N, -1, C * patch_size * patch_size)
        return patches

    def forward(self, images):
        patches = self.patchify(images)
        patches_embedded = self.patch_embedding(patches)
        
        # Append a [CLS] token to the beginning of the sequence of patch embeddings
        N = patches_embedded.size(0)
        cls_tokens = self.cls_token.expand(N, -1, -1)
        output = torch.cat((cls_tokens, patches_embedded), dim=1)

        output = self.positional_encoding(output)
        mask = None  # No masking needed for ViT

        for layer in self.layers:
            output = layer(output, mask)

        # Take the [CLS] token embedding and feed it through a linear layer to obtain logits
        cls_output = output[:, 0, :]
        logits = self.fc(cls_output)

        return logits

    def _init_weights(self, module):
        """
        Initialize the weights of the network.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
