import math
import copy

import torch
import torch.nn as nn

from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        self.alpha = config.alpha
        self.beta = config.beta

        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, resi, l):
        theta = math.log(self.beta / l + 1)
        
        mixed_query_layer = theta * self.query(hidden_states) + (1 - theta) * hidden_states
        mixed_key_layer = theta * self.key(hidden_states) + (1 - theta) * hidden_states
        
        origin_with_identity_mapping = theta * self.value(hidden_states) + (1 - theta) * hidden_states
        initial_with_identity_mapping = theta * self.value(resi) + (1 - theta) * resi
        mixed_value_layer = (1 - self.alpha) * origin_with_identity_mapping + self.alpha * initial_with_identity_mapping
        

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = torch.nn.functional.gelu
        self.dropout = Dropout(config.transformer["dropout_rate"]) 

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        img_size = _pair(img_size)


        patch_size = _pair(config.patches["size"])
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Attention(config)

        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)

    def forward(self, x, resi, l):
        h = x
        x = self.attention_norm(x)
        x, attn = self.attn(x, resi, l)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, attn


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.num_layer = config.transformer["num_layers"]
        self.layer = nn.ModuleList()

        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(self.num_layer):
            layer = Block(config)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, h0):
        attn_list = []
        fe_list = []
        resi = h0
        for idx, layer_block in enumerate(self.layer):
            fraction = (idx + 1) % 8
            hidden_states, attn = layer_block(hidden_states, resi, idx + 1)
            if fraction == 0: resi = hidden_states
            attn_list.append(attn)
            fe_list.append(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded, fe_list, attn_list


class Transformer(nn.Module):
    def __init__(self, config, img_size):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config)

    def forward(self, input_ids):
        embedding_output = self.embeddings(input_ids)
        encoded, fe_list, attn_list = self.encoder(embedding_output, embedding_output)
        return encoded, fe_list, attn_list


class ViTRI(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False):
        super(ViTRI, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Transformer(config, img_size)
        self.head = Linear(config.hidden_size, num_classes)

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        x, fe_list, attn_list = self.transformer(x)
        logits = self.head(x[:, 0])

        return logits, x, fe_list, attn_list