import torch
from architectures.Transformer import Transformer
import torch.nn as nn
from data.utils import patchify

class ClassificationViT(nn.Module):
    """Vision transformer for classfication
    Args:
        n_classes: number of classes
        embedding_dim: dimension of embedding
        patch_size: image patch size
        num_patches: number of image patches
    Returns:
        Logits of classfication
    """
    def __init__(self, n_classes, embedding_dim=256, patch_size=4, num_patches=8):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embedding_dim = embedding_dim

        self.transformer = Transformer(embedding_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim) * 0.02)
        self.position_encoding = nn.Parameter(
            torch.randn(1, num_patches * num_patches + 1, embedding_dim) * 0.02
        )
        self.patch_projection = nn.Linear(patch_size * patch_size * 3, embedding_dim)

        # A Layernorm and a Linear layer are applied on ViT encoder embeddings
        self.output_head = nn.Sequential(
            nn.LayerNorm(embedding_dim), nn.Linear(embedding_dim, n_classes)
        )

    def forward(self, images):
        """
        (1) Splitting images into fixed-size patches;
        (2) Linearly embed each image patch, prepend CLS token;
        (3) Add position embeddings;
        (4) Feed the resulting sequence of vectors to Transformer encoder.
        (5) Extract the embeddings corresponding to the CLS token.
        (6) Apply output head to the embeddings to obtain the logits
        """
        patch_images = patchify(images, patch_size=self.patch_size)
        linear_embed = self.patch_projection(patch_images)
        expand_cls = torch.broadcast_to(self.cls_token,
                                        (linear_embed.size(0), 1, linear_embed.size(2)))
        linear_embed_cls = torch.cat((expand_cls, linear_embed), 1)
        embeddings_plus_position = linear_embed_cls + self.position_encoding
        transformer_output = self.transformer(embeddings_plus_position)
        cls_embedding = transformer_output[:, 0, :]
        return  self.output_head(cls_embedding)