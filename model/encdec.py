import torch
import torch.nn as nn
from model.layers import LayerNorm, clones


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers: # layer is EncoderLayer(d_model, c(attn), c(ff), dropout)
            x = layer(x, mask) # calls EncoderLayer.forward(src*, src_mask) *with embeddings and positional encoding
        return self.norm(x) # calls LayerNorm.forward()


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)  # self.layer is a DecoderLayer object
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        ## this loop calls a DecoderLayer object directly N times (again executing DecoderLayer.forward() through __call__)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder # Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        self.decoder = decoder # Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)
        self.src_embed = src_embed # nn.Sequential(Embeddings(d_model, src_vocab), c(position))
        self.tgt_embed = tgt_embed # nn.Sequential(Embeddings(d_model, tgt_vocab), c(position))
        self.generator = generator # Generator(d_model, tgt_vocab)

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        ## self.src_embed calls Embeddings.forward(src) (output then channelled to PositionalEncoding.forward())
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
