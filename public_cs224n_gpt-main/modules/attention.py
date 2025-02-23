import torch

from einops import rearrange
from torch import nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    # each k,q,v is of size bs x num_attention_heads x seq_len x attention_head_size
    ### YOUR CODE HERE
    # compute the for product
    key_T = key.transpose(-2,-1)
    dot_product_scores = torch.matmul(query, key_T) # bs x num_attention_heads x seq_len x seq_len

    # apply scaling
    dot_product_scores = dot_product_scores / torch.sqrt(torch.tensor(self.attention_head_size, dtype=key.dtype, device=key.device))
    # mask out the padding tokens
    dot_product_scores += attention_mask
    # mask out the future tokens
    future_mask = torch.triu(torch.ones(dot_product_scores.shape[-1], dot_product_scores.shape[-1]), diagonal=1).to(device=key.device)
    dot_product_scores = dot_product_scores.masked_fill(future_mask==1, float('-inf'))

    # attention weights
    weights = F.softmax(dot_product_scores, dim=-1)
    weights = self.dropout(weights)

    # calculate weighted values
    weighted_values = torch.matmul(weights, value)
    output = rearrange(weighted_values, 'b h t d -> b t (h d)')
    return output


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
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    print(attn_value)
    return attn_value

#
# if __name__ == "__main__":
#   bs = 2
#   heads = 4
#   seq_len = 10
#   head_size = 15
#   hs = head_size * heads
#   config = GPT2Config(hidden_size=hs,
#                       num_attention_heads=heads,
#                       attention_probs_dropout_prob=0.5)
#   test_attention = CausalSelfAttention(config)
#   # hidden_states: [bs, seq_len, hidden_state]
#   hidden_s = torch.rand(bs, seq_len, hs)
#   # attention_mask: [bs, 1, 1, seq_len]
#   mask = torch.tensor([[[[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]]],
#                        [[[1, 1, 1, 1, 1, 1, 1, 1, 0, 0]]]])
#   test_attention(hidden_s, mask)
#   print('hi')