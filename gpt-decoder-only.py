import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math
    import random
    import string
    from torch.utils.data import Dataset, DataLoader
    return DataLoader, Dataset, F, math, nn, random, string, torch


@app.cell
def _(random, string):
    def generate_kv_sample(
        num_pairs=5,
        key_length=1,
        value_range=(0, 99),
        query_key=None,
    ):
        """
        Generates a single key-value recall training example.
        Returns a string suitable for decoder-only LM training.
        """
        keys = set()
        while len(keys) < num_pairs:
            key = ''.join(random.choices(string.ascii_uppercase, k=key_length))
            keys.add(key)
        keys = list(keys)

        values = {k: random.randint(*value_range) for k in keys}

        if query_key is None:
            query_key = random.choice(keys)

        context = []
        for k in keys:
            context.append(f"{k} is {values[k]}.")
        context_str = " ".join(context)

        query = f" What is {query_key}? {query_key} is {values[query_key]}."
        return context_str + query


    def generate_dataset(
        num_samples=1000,
        num_pairs_range=(3, 10),
        key_length=1,
        value_range=(0, 99),
    ):
        """
        Generates a list of text samples.
        """
        dataset = []
        for _ in range(num_samples):
            num_pairs = random.randint(*num_pairs_range)
            sample = generate_kv_sample(
                num_pairs=num_pairs,
                key_length=key_length,
                value_range=value_range,
            )
            dataset.append(sample)
        return dataset

    return (generate_dataset,)


@app.cell
def _(string):
    class SimpleTokenizer:
        """Character-level tokenizer"""
        def __init__(self):
            # Build vocabulary from expected characters
            chars = list(string.ascii_uppercase + string.digits + " .?")
            self.char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
            self.idx_to_char = {idx: ch for ch, idx in self.char_to_idx.items()}
            self.vocab_size = len(self.char_to_idx)

        def encode(self, text):
            """Convert text to token indices"""
            return [self.char_to_idx.get(ch, 0) for ch in text]

        def decode(self, indices):
            """Convert token indices back to text"""
            return ''.join([self.idx_to_char.get(idx, '?') for idx in indices])


    return (SimpleTokenizer,)


@app.cell
def _(Dataset, torch):
    class KVDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length=128):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.data = []

            for text in texts:
                tokens = tokenizer.encode(text)
                if len(tokens) <= max_length:
                    self.data.append(tokens)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            tokens = self.data[idx]
            # Pad to max_length
            padded = tokens + [0] * (self.max_length - len(tokens))
            padded = padded[:self.max_length]

            # Input and target (shifted by 1)
            x = torch.tensor(padded[:-1], dtype=torch.long)
            y = torch.tensor(padded[1:], dtype=torch.long)

            return x, y
    return (KVDataset,)


@app.cell
def _(F, math, nn, torch):

    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model, num_heads, dropout=0.1):
            super().__init__()
            assert d_model % num_heads == 0

            self.d_model = d_model
            self.num_heads = num_heads
            self.head_dim = d_model // num_heads

            self.qkv_proj = nn.Linear(d_model, 3 * d_model)
            self.out_proj = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            batch_size, seq_len, d_model = x.shape

            # Project and split into Q, K, V
            qkv = self.qkv_proj(x)
            qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq, head_dim)
            q, k, v = qkv[0], qkv[1], qkv[2]

            # Attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # Apply causal mask
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)

            # Apply attention to values
            out = torch.matmul(attn_weights, v)

            # Reshape and project
            out = out.transpose(1, 2).contiguous()
            out = out.reshape(batch_size, seq_len, d_model)
            out = self.out_proj(out)

            return out


    class FeedForward(nn.Module):
        def __init__(self, d_model, d_ff, dropout=0.1):
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = F.gelu(self.linear1(x))
            x = self.dropout(x)
            x = self.linear2(x)
            return x


    class TransformerBlock(nn.Module):
        def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
            super().__init__()
            self.attention = MultiHeadAttention(d_model, num_heads, dropout)
            self.feed_forward = FeedForward(d_model, d_ff, dropout)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask):
            # Self-attention with residual
            attn_out = self.attention(self.ln1(x), mask)
            x = x + self.dropout(attn_out)

            # Feed-forward with residual
            ff_out = self.feed_forward(self.ln2(x))
            x = x + self.dropout(ff_out)

            return x

    return (TransformerBlock,)


@app.cell
def _(F, TransformerBlock, nn, torch):
    class SimpleGPT(nn.Module):
        def __init__(
            self,
            vocab_size,
            d_model=128,
            num_heads=4,
            num_layers=4,
            d_ff=512,
            max_seq_len=128,
            dropout=0.1
        ):
            super().__init__()

            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Embedding(max_seq_len, d_model)

            self.blocks = nn.ModuleList([
                TransformerBlock(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])

            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size)

            self.dropout = nn.Dropout(dropout)
            self.max_seq_len = max_seq_len

            # Initialize weights
            self.apply(self._init_weights)

        def _init_weights(self, module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def forward(self, x):
            batch_size, seq_len = x.shape

            # Create causal mask
            mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

            # Token + position embeddings
            positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
            positions = positions.unsqueeze(0).expand(batch_size, seq_len)

            x = self.token_embedding(x) + self.position_embedding(positions)
            x = self.dropout(x)

            # Apply transformer blocks
            for block in self.blocks:
                x = block(x, mask)

            x = self.ln_f(x)
            logits = self.head(x)

            return logits

        def generate(self, idx, max_new_tokens, temperature=1.0):
            """Generate new tokens autoregressively"""
            for _ in range(max_new_tokens):
                # Crop context if needed
                idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len:]

                # Get predictions
                logits = self(idx_cond)
                logits = logits[:, -1, :] / temperature

                # Sample
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)

                # Append
                idx = torch.cat((idx, idx_next), dim=1)

            return idx

    return (SimpleGPT,)


@app.cell
def _(F, torch):
    def train_epoch(model, dataloader, optimizer, device):
        model.train()
        total_loss = 0

        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # Forward pass
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)


    def evaluate(model, dataloader, device):
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                total_loss += loss.item()

        return total_loss / len(dataloader)


    return evaluate, train_epoch


@app.cell
def _(
    DataLoader,
    KVDataset,
    SimpleGPT,
    SimpleTokenizer,
    generate_dataset,
    torch,
):
    # Hyperparameters
    num_train_samples = 5000
    num_val_samples = 500
    batch_size = 32
    num_epochs = 20
    learning_rate = 3e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")

    # Generate data
    print("Generating training data...")
    train_texts = generate_dataset(num_samples=num_train_samples)
    val_texts = generate_dataset(num_samples=num_val_samples)

    # Show some examples
    print("\nExample training samples:")
    for i in range(3):
        print(f"{i+1}. {train_texts[i]}")

    # Create tokenizer
    tokenizer = SimpleTokenizer()
    print(f"\nVocabulary size: {tokenizer.vocab_size}")

    # Create datasets
    train_dataset = KVDataset(train_texts, tokenizer, max_length=128)
    val_dataset = KVDataset(val_texts, tokenizer, max_length=128)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    model = SimpleGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        num_heads=4,
        num_layers=4,
        d_ff=512,
        max_seq_len=128,
        dropout=0.1
    ).to(device)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return (
        device,
        model,
        num_epochs,
        optimizer,
        tokenizer,
        train_loader,
        val_loader,
    )


@app.cell
def _(
    device,
    evaluate,
    model,
    num_epochs,
    optimizer,
    train_epoch,
    train_loader,
    val_loader,
):
    # Training loop
    print("\nTraining...")
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    return


@app.cell
def _(device, generate_dataset, model, tokenizer, torch):
    # Test generation
    print("\n" + "="*80)
    print("Testing generation:")
    print("="*80)

    model.eval()
    test_samples = generate_dataset(num_samples=3, num_pairs_range=(5, 7))

    for i, test_text in enumerate(test_samples):
        # Split at "What is"
        context_end = test_text.index("What is")
        context = test_text[:context_end]
        full_text = test_text

        print(f"\nTest {i+1}:")
        print(f"Context: {context}")
        print(f"Full text: {full_text}")

        # Encode context
        context_tokens = tokenizer.encode(context)
        context_tensor = torch.tensor([context_tokens], dtype=torch.long).to(device)

        # Generate
        with torch.no_grad():
            generated = model.generate(context_tensor, max_new_tokens=30, temperature=0.5)

        generated_text = tokenizer.decode(generated[0].tolist())
        print(f"Generated: {generated_text}")
        print("-" * 80)
    return


@app.cell
def _(model, torch):
    # Save model
    torch.save(model.state_dict(), './simple_gpt_model.pt')
    print("\nModel saved to simple_gpt_model.pt")
    return


if __name__ == "__main__":
    app.run()
