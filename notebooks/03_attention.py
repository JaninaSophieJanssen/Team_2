import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    return mo, nn, plt, sns, torch


@app.cell
def _(mo):
    mo.md("""
    ## Computing attention

    In this notebook we gonna look at how to compute the attention score and build a transformer block.

    Example sentence: **"be water my friend"**
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    A first simple approach could be to take the dot product between the word embeddings themselves. However this way we have no possibility to learn how the words should pay attention to each other.
    """)
    return


@app.cell
def _(torch):
    # Example sentence: "be water my friend"
    # We randomly initialize word embeddings for the four words in our example sentence.
    torch.manual_seed(42)
    inputs = torch.tensor([
        [0.27, 0.84, 0.45, 0.12],  # be
        [0.63, 0.19, 0.78, 0.91],  # water
        [0.34, 0.56, 0.22, 0.88],  # my
        [0.71, 0.05, 0.93, 0.37],  # friend
    ])
    inputs
    return (inputs,)


@app.cell
def _(inputs):
    # Compute attention scores for all pairs
    attn_scores = inputs @ inputs.T
    attn_scores
    return (attn_scores,)


@app.cell
def _(attn_scores, plt, sns):
    def visualize_attention(matrix, title, cmap="YlOrRd", annot=True, fmt='.3f'):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(matrix.detach().numpy(),
                    annot=annot,
                    fmt=fmt,
                    cmap=cmap,
                    square=True,
                    cbar_kws={'label': 'Value'},
                    xticklabels=['be', 'water', 'my', 'friend'],
                    yticklabels=['be', 'water', 'my', 'friend'],
                    ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Keys')
        ax.set_ylabel('Queries')
        plt.tight_layout()
        return fig

    fig_scores = visualize_attention(attn_scores, "Attention Scores (X @ X^T)")
    fig_scores
    return (visualize_attention,)


@app.cell
def _(mo):
    mo.md("""
    Now we normalize the attention scores using the softmax function so that they sum to 1 and we can use them as weights to compute a weighted sum of our vectors.
    """)
    return


@app.cell
def _(attn_scores, torch):
    attn_weights = torch.softmax(attn_scores, dim=-1)
    attn_weights
    return (attn_weights,)


@app.cell
def _(attn_weights, visualize_attention):
    fig_weights = visualize_attention(attn_weights, "Attention Weights (Normalized)", cmap="Blues")
    fig_weights
    return


@app.cell
def _(mo):
    mo.md("""
    Using these weights, we can compute a weighted sum of our input vectorsto get the final output vectors.
    """)
    return


@app.cell
def _(attn_weights, inputs):
    # Context vectors as weighted sum
    context_vecs = attn_weights @ inputs
    context_vecs
    return


@app.cell
def _(mo):
    mo.md("""
    ## Self-Attention with Query, Key, Value Projections

    Now, we introduce learnable parameters. So, instead of using embeddings directly, we project them into separate query (Q), key (K), and value (V) spaces using weight matrices.
    """)
    return


@app.cell
def _(inputs, nn, torch):
    # Set dimensions
    d_in = inputs.shape[1]  # 4
    d_out = 3  # output embedding dimension

    # We initialize random weight matrices for query, key, and value projections
    torch.manual_seed(123)
    W_query = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    return W_key, W_query, W_value, d_in, d_out


@app.cell
def _(mo):
    mo.md(r"""
    Using these matrices, we can compute our query, key, and value vectors.
    """)
    return


@app.cell
def _(W_key, W_query, W_value, inputs):
    # Compute query, key, value vectors
    queries = inputs @ W_query
    keys = inputs @ W_key
    values = inputs @ W_value

    print("Queries:\n", queries)
    print("Keys:\n", keys)
    print("Values:\n", values)
    return keys, queries, values


@app.cell
def _(mo):
    mo.md("""
    Step 1: Compute the attention scores using the projected queries and keys.
    """)
    return


@app.cell
def _(keys, queries):
    # Compute attention scores with learned projections
    attn_scores_qkv = queries @ keys.T
    attn_scores_qkv
    return (attn_scores_qkv,)


@app.cell
def _(attn_scores_qkv, visualize_attention):
    fig_qkv_scores = visualize_attention(attn_scores_qkv, "Attention Scores (Q @ K^T)", cmap="Oranges")
    fig_qkv_scores
    return


@app.cell
def _(mo):
    mo.md("""
    Step 2: Now we softmax these scores to obtain attention weights that sum to 1. Also, we divide the scores by the square root of the dimension of the key vectors (d_k) to stabilize gradients during training.
    """)
    return


@app.cell
def _(attn_scores_qkv, keys, torch):
    # Scaled dot-product attention (divide by sqrt(d_k))
    d_k = keys.shape[-1]
    attn_weights_qkv = torch.softmax(attn_scores_qkv / d_k**0.5, dim=-1)
    attn_weights_qkv
    return (attn_weights_qkv,)


@app.cell
def _(attn_weights_qkv, visualize_attention):
    fig_qkv_weights = visualize_attention(attn_weights_qkv, "Attention Weights (Scaled & Normalized)", cmap="Greens")
    fig_qkv_weights
    return


@app.cell
def _(mo):
    mo.md("""
    Step 3: Using these weights, we compute a weighted sum of the value vectors to get the final output vectors.
    """)
    return


@app.cell
def _(attn_weights_qkv, values):
    # Compute context vectors
    context_vecs_qkv = attn_weights_qkv @ values
    context_vecs_qkv
    return


@app.cell
def _(mo):
    mo.md("""
    ## Self-Attention as Torch Layer
    We can write this whole process as a compact PyTorch layer.
    """)
    return


@app.cell
def _(d_in, d_out, nn, torch):
    class SelfAttention_v2(nn.Module):
        def __init__(self, d_in, d_out, qkv_bias=False):
            super().__init__()
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        def forward(self, x):
            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)

            attn_scores = queries @ keys.T
            attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
            context_vec = attn_weights @ values
            return context_vec

    torch.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    sa_v2
    return (sa_v2,)


@app.cell
def _(inputs, sa_v2):
    output_v2 = sa_v2(inputs)
    output_v2
    return


@app.cell
def _(mo):
    mo.md("""
    ## Masking Self-Attention

    We mask attention so that future tokens are not attended to previous positions (autoregressive).
    """)
    return


@app.cell
def _(attn_scores_qkv, torch):
    # Create causal mask
    seq_length = attn_scores_qkv.shape[0]
    mask_simple = torch.tril(torch.ones(seq_length, seq_length))
    mask_simple
    return mask_simple, seq_length


@app.cell
def _(mask_simple, plt, sns):
    # Visualize mask
    fig_mask, ax_mask = plt.subplots(figsize=(5, 4))
    sns.heatmap(mask_simple.numpy(),
                annot=True,
                fmt='.0f',
                cmap="Greys",
                square=True,
                cbar=False,
                xticklabels=['be', 'water', 'my', 'friend'],
                yticklabels=['be', 'water', 'my', 'friend'],
                ax=ax_mask)
    ax_mask.set_title("Causal Mask (1=attend, 0=mask)")
    plt.tight_layout()
    fig_mask
    return


@app.cell
def _(attn_scores_qkv, seq_length, torch):
    # Apply mask to attention scores
    mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
    masked_scores = attn_scores_qkv.masked_fill(mask.bool(), -torch.inf)
    masked_scores
    return (masked_scores,)


@app.cell
def _(keys, masked_scores, torch):
    # Compute causal attention weights
    attn_weights_causal = torch.softmax(masked_scores / keys.shape[-1]**0.5, dim=-1)
    attn_weights_causal
    return (attn_weights_causal,)


@app.cell
def _(attn_weights_causal, visualize_attention):
    fig_causal = visualize_attention(attn_weights_causal, "Causal Attention Weights", cmap="Purples")
    fig_causal
    return


@app.cell
def _(mo):
    mo.md("""
    ## Mask Attention Class with Dropout
    Again we can write this as a compact PyTorch layer and also add dropout for regularization.
    """)
    return


@app.cell
def _(nn, torch):
    class MaskedAttention(nn.Module):
        def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
            super().__init__()
            self.d_out = d_out
            self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
            self.dropout = nn.Dropout(dropout)
            self.register_buffer(
                'mask',
                torch.triu(torch.ones(context_length, context_length), diagonal=1)
            )

        def forward(self, x):
            b, num_tokens, d_in = x.shape
            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)

            attn_scores = queries @ keys.transpose(1, 2)
            attn_scores.masked_fill_(
                self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
            )
            attn_weights = torch.softmax(
                attn_scores / keys.shape[-1]**0.5, dim=-1
            )
            attn_weights = self.dropout(attn_weights)
            context_vec = attn_weights @ values
            return context_vec

    return (MaskedAttention,)


@app.cell
def _(MaskedAttention, d_in, d_out, seq_length, torch):
    torch.manual_seed(123)
    ca = MaskedAttention(d_in, d_out, seq_length, dropout=0.0)
    ca
    return (ca,)


@app.cell
def _(ca, inputs, torch):
    # Create batch input (batch_size=2)
    batch = torch.stack((inputs, inputs), dim=0)
    context_vecs_masked = ca(batch)
    context_vecs_masked
    return (batch,)


@app.cell
def _(mo):
    mo.md("""
    ## Multi-Head Attention

    Lastly, we can extend this to multi-head attention by computing multiple sets of query, key, and value projections in parallel and concatenating their outputs and projecting them back to the original dimension.
    """)
    return


@app.cell
def _(nn, torch):
    class MultiHeadAttention(nn.Module):
        def __init__(self, d_in, d_out_per_head, context_length, dropout, num_heads, qkv_bias=False):
            super().__init__()

            self.num_heads = num_heads
            self.head_dim = d_out_per_head
            self.d_out = num_heads * d_out_per_head  # Calculate total d_out

            self.W_query = nn.Linear(d_in, self.d_out, bias=qkv_bias)
            self.W_key = nn.Linear(d_in, self.d_out, bias=qkv_bias)
            self.W_value = nn.Linear(d_in, self.d_out, bias=qkv_bias)
            self.out_proj = nn.Linear(self.d_out, self.d_out)
            self.dropout = nn.Dropout(dropout)
            self.register_buffer(
                "mask",
                torch.triu(torch.ones(context_length, context_length), diagonal=1)
            )

        def forward(self, x):
            b, num_tokens, d_in = x.shape

            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)

            # Split into multiple heads
            keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
            values = values.view(b, num_tokens, self.num_heads, self.head_dim)
            queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

            keys = keys.transpose(1, 2)
            queries = queries.transpose(1, 2)
            values = values.transpose(1, 2)

            # Compute attention for all heads
            attn_scores = queries @ keys.transpose(2, 3)
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_bool, -torch.inf)

            attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)

            context_vec = (attn_weights @ values).transpose(1, 2)
            context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
            context_vec = self.out_proj(context_vec)

            return context_vec, attn_weights

    return (MultiHeadAttention,)


@app.cell
def _(MultiHeadAttention, d_in, seq_length, torch):
    torch.manual_seed(123)
    d_out_per_head = 4  # Each head outputs 4 dimensions
    mha = MultiHeadAttention(d_in, d_out_per_head, seq_length, dropout=0.0, num_heads=2)
    # Total d_out will be: 2 heads * 4 dims = 8
    mha
    return d_out_per_head, mha


@app.cell
def _(batch, mha):
    context_vecs_mha, attn_weights_mha = mha(batch)
    context_vecs_mha, attn_weights_mha.shape
    return (attn_weights_mha,)


@app.cell
def _(attn_weights_mha, plt, sns):
    # Visualize attention patterns for both heads
    fig_mha, axes = plt.subplots(1, 2, figsize=(12, 5))

    for head_idx in range(2):
        ax = axes[head_idx]
        head_weights = attn_weights_mha[0, head_idx].detach().numpy()
        sns.heatmap(head_weights,
                    annot=True,
                    fmt='.3f',
                    cmap="viridis",
                    square=True,
                    xticklabels=['be', 'water', 'my', 'friend'],
                    yticklabels=['be', 'water', 'my', 'friend'],
                    ax=ax,
                    cbar_kws={'label': 'Attention Weight'})
        ax.set_title(f"Head {head_idx + 1} Attention Pattern")
        ax.set_xlabel('Keys')
        ax.set_ylabel('Queries')

    plt.tight_layout()
    fig_mha
    return


@app.cell
def _(mo):
    mo.md("""
    ## Building a Complete Transformer Decoder

    Now let's build a full transformer decoder by combining multi-head attention with feed-forward networks, layer normalization, and residual connections. We use GELU as the activation function in the feed-forward network.
    """)
    return


@app.cell
def _(nn, torch):
    class LayerNorm(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.eps = 1e-5
            self.scale = nn.Parameter(torch.ones(emb_dim))
            self.shift = nn.Parameter(torch.zeros(emb_dim))

        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            norm_x = (x - mean) / torch.sqrt(var + self.eps)
            return self.scale * norm_x + self.shift

    class FeedForward(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(emb_dim, 4 * emb_dim),
                nn.GELU(),
                nn.Linear(4 * emb_dim, emb_dim),
            )

        def forward(self, x):
            return self.layers(x)
    return FeedForward, LayerNorm


@app.cell
def _(FeedForward, LayerNorm, MultiHeadAttention, nn):
    class TransformerBlock(nn.Module):
        def __init__(self, emb_dim, context_length, n_heads, drop_rate, qkv_bias=False):
            super().__init__()
            self.att = MultiHeadAttention(
                d_in=emb_dim,
                d_out_per_head=emb_dim // n_heads,
                context_length=context_length,
                num_heads=n_heads,
                dropout=drop_rate,
                qkv_bias=qkv_bias)
            self.ff = FeedForward(emb_dim)
            self.norm1 = LayerNorm(emb_dim)
            self.norm2 = LayerNorm(emb_dim)
            self.drop_shortcut = nn.Dropout(drop_rate)

        def forward(self, x):
            # Shortcut connection for attention block
            shortcut = x
            x = self.norm1(x)
            x, _ = self.att(x)  # MultiHeadAttention returns (context_vec, attn_weights)
            x = self.drop_shortcut(x)
            x = x + shortcut  # Add the original input back

            # Shortcut connection for feed forward block
            shortcut = x
            x = self.norm2(x)
            x = self.ff(x)
            x = self.drop_shortcut(x)
            x = x + shortcut  # Add the original input back

            return x
    return (TransformerBlock,)


@app.cell
def _(LayerNorm, TransformerBlock, nn, torch):
    class TransformerDecoder(nn.Module):
        def __init__(self, vocab_size, emb_dim, context_length, n_heads, n_layers, drop_rate, qkv_bias=False):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, emb_dim)
            self.pos_emb = nn.Embedding(context_length, emb_dim)
            self.drop_emb = nn.Dropout(drop_rate)

            self.trf_blocks = nn.Sequential(
                *[TransformerBlock(emb_dim, context_length, n_heads, drop_rate, qkv_bias)
                  for _ in range(n_layers)])

            self.final_norm = LayerNorm(emb_dim)
            self.out_head = nn.Linear(emb_dim, vocab_size, bias=False)

        def forward(self, in_idx):
            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
            x = self.drop_emb(x)
            x = self.trf_blocks(x)
            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits
    return (TransformerDecoder,)


@app.cell
def _(TransformerDecoder, torch):
    # Model hyperparameters (GPT-2 124M style)
    vocab_size = 50257      # Vocabulary size
    emb_dim = 768           # Embedding dimension
    context_length = 256    # Context length (shorter for demo)
    n_heads = 12            # Number of attention heads
    n_layers = 12           # Number of layers
    drop_rate = 0.1         # Dropout rate
    qkv_bias = False        # Query-Key-Value bias

    torch.manual_seed(123)
    model = TransformerDecoder(
        vocab_size=vocab_size,
        emb_dim=emb_dim,
        context_length=context_length,
        n_heads=n_heads,
        n_layers=n_layers,
        drop_rate=drop_rate,
        qkv_bias=qkv_bias
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    model
    return model, vocab_size


@app.cell
def _(model, torch, vocab_size):
    # Test the model with a simple batch
    batch_size = 2
    seq_len = 10

    # Create random token indices
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Forward pass
    logits = model(input_ids)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected: [batch_size={batch_size}, seq_len={seq_len}, vocab_size={vocab_size}]")

    logits.shape
    return batch_size, input_ids, logits, seq_len


if __name__ == "__main__":
    app.run()
