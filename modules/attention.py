import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads # 12
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads) # 768/12 = 64
    self.all_head_size = self.num_attention_heads * self.attention_head_size # 12*64 = 768

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size) #768 x 768
    self.key = nn.Linear(config.hidden_size, self.all_head_size) # 768 x 768
    self.value = nn.Linear(config.hidden_size, self.all_head_size) # 768 x 768
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob) # 768
    if hasattr(torch.nn.functional, CausalSelfAttention)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj
  # query: b h t d; key: b h t d; kT = b h d t; q.KT = (b h t t) x (b h t d) = (b h t d)-> (b t h d)

  def attention(self, key, query, value, attention_mask):

    attn_scores = torch.matmul(query, (key.transpose(-2,-1))/(self.attention_head_size**0.5))
    mask = torch.tril(torch.ones_like(attn_scores))
    attn_scores = attn_scores.masked_fill(mask==0, float('-inf'))

    if attention_mask is not None:
      attn_scores = attn_scores + attention_mask
      
    attn_probs = torch.softmax(attn_scores, dim = -1)
    attn_probs = self.dropout(attn_probs)

    y = torch.matmul(attn_probs, value)
    y = rearrange(y, 'b h t d -> b t (h d)')

    return y


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
