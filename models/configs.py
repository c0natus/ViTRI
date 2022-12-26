# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ml_collections


def get_b12_config():
    """Returns the ViT-B/12 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 384
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = config.hidden_size * 3
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0
    config.transformer.dropout_rate = 0.2
    config.classifier = 'token'
    config.alpha = 0.1
    config.beta = 0.5
    config.representation_size = None
    return config


def get_b16_config():
    """Returns the ViT-B/16 configuration. (S)"""
    config = get_b12_config()
    config.transformer.num_layers = 16
    return config


def get_b24_config():
    """Returns the ViT-B/16 configuration."""
    config = get_b12_config()
    config.transformer.num_layers = 24
    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration. (L)"""
    config = get_b12_config()
    config.transformer.num_layers = 32
    return config

def get_b40_config():
    """Returns the ViT-B/40 configuration. (L)"""
    config = get_b12_config()
    config.transformer.num_layers = 40
    return config


CONFIGS = {
    'ViT-B_12': get_b12_config(),
    'ViT-B_16': get_b16_config(),
    'ViT-B_24': get_b24_config(),
    'ViT-B_32': get_b32_config(),
    'ViT-B_40': get_b40_config(),
}