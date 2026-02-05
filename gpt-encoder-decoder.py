import marimo

__generated_with = "0.17.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import math
    import random
    import numpy as np
    return DataLoader, Dataset, math, nn, optim, random, torch


@app.cell
def _(Dataset, math, nn, random, torch):
    class SqrtDataset(Dataset):
        """Dataset for generating number -> square root pairs"""

        def __init__(self, num_samples=10000, max_num=99999):
            self.num_samples = num_samples
            self.max_num = max_num

            # Special tokens
            self.PAD_TOKEN = '<PAD>'
            self.SOS_TOKEN = '<SOS>'
            self.EOS_TOKEN = '<EOS>'

            # Build vocabulary (digits + special tokens)
            self.vocab = [self.PAD_TOKEN, self.SOS_TOKEN, self.EOS_TOKEN] + [str(i) for i in range(10)]
            self.token2idx = {token: idx for idx, token in enumerate(self.vocab)}
            self.idx2token = {idx: token for token, idx in self.token2idx.items()}
            self.vocab_size = len(self.vocab)

            # Generate dataset
            self.data = self._generate_data()

        def _generate_data(self):
            data = []
            for _ in range(self.num_samples):
                num = random.randint(1, self.max_num)
                sqrt_num = round(math.sqrt(num))

                # Convert to sequences of tokens
                input_seq = list(str(num))
                output_seq = list(str(sqrt_num))

                data.append((input_seq, output_seq))
            return data

        def encode_sequence(self, seq, add_sos=False, add_eos=False):
            """Convert sequence of digit strings to indices"""
            encoded = []
            if add_sos:
                encoded.append(self.token2idx[self.SOS_TOKEN])
            encoded.extend([self.token2idx[token] for token in seq])
            if add_eos:
                encoded.append(self.token2idx[self.EOS_TOKEN])
            return encoded

        def decode_sequence(self, indices):
            """Convert indices back to string"""
            tokens = [self.idx2token[idx] for idx in indices
                     if idx not in [self.token2idx[self.PAD_TOKEN],
                                   self.token2idx[self.SOS_TOKEN],
                                   self.token2idx[self.EOS_TOKEN]]]
            return ''.join(tokens)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            input_seq, output_seq = self.data[idx]

            # Encode sequences
            encoder_input = self.encode_sequence(input_seq)
            # Decoder input has SOS at start
            decoder_input = self.encode_sequence(output_seq, add_sos=True)
            # Decoder target has EOS at end
            decoder_target = self.encode_sequence(output_seq, add_eos=True)

            return {
                'encoder_input': torch.tensor(encoder_input, dtype=torch.long),
                'decoder_input': torch.tensor(decoder_input, dtype=torch.long),
                'decoder_target': torch.tensor(decoder_target, dtype=torch.long),
            }


    def collate_fn(batch):
        """Custom collate function to pad sequences"""
        encoder_inputs = [item['encoder_input'] for item in batch]
        decoder_inputs = [item['decoder_input'] for item in batch]
        decoder_targets = [item['decoder_target'] for item in batch]

        # Pad sequences
        encoder_inputs = nn.utils.rnn.pad_sequence(encoder_inputs, batch_first=True, padding_value=0)
        decoder_inputs = nn.utils.rnn.pad_sequence(decoder_inputs, batch_first=True, padding_value=0)
        decoder_targets = nn.utils.rnn.pad_sequence(decoder_targets, batch_first=True, padding_value=0)

        return {
            'encoder_input': encoder_inputs,
            'decoder_input': decoder_inputs,
            'decoder_target': decoder_targets,
        }
    return SqrtDataset, collate_fn


@app.cell
def _(math, nn, torch):
    class PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding"""

        def __init__(self, d_model, max_len=100, dropout=0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            # Create positional encoding matrix
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

            pe = pe.unsqueeze(0)  # [1, max_len, d_model]
            self.register_buffer('pe', pe)

        def forward(self, x):
            """
            Args:
                x: [batch_size, seq_len, d_model]
            """
            x = x + self.pe[:, :x.size(1), :]
            return self.dropout(x)


    class MultiHeadAttention(nn.Module):
        """Multi-head attention mechanism"""

        def __init__(self, d_model, num_heads, dropout=0.1):
            super().__init__()
            assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

            self.d_model = d_model
            self.num_heads = num_heads
            self.d_k = d_model // num_heads

            # Linear projections
            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_o = nn.Linear(d_model, d_model)

            self.dropout = nn.Dropout(dropout)

        def scaled_dot_product_attention(self, Q, K, V, mask=None):
            """
            Args:
                Q: [batch_size, num_heads, seq_len, d_k]
                K: [batch_size, num_heads, seq_len, d_k]
                V: [batch_size, num_heads, seq_len, d_k]
                mask: [batch_size, 1, seq_len, seq_len] or [batch_size, 1, 1, seq_len]
            Returns:
                output: [batch_size, num_heads, seq_len, d_k]
                attention: [batch_size, num_heads, seq_len, seq_len]
            """
            # Calculate attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

            # Apply mask (if provided)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            # Apply softmax
            attention = torch.softmax(scores, dim=-1)
            attention = self.dropout(attention)

            # Apply attention to values
            output = torch.matmul(attention, V)

            return output, attention

        def forward(self, query, key, value, mask=None):
            """
            Args:
                query: [batch_size, seq_len, d_model]
                key: [batch_size, seq_len, d_model]
                value: [batch_size, seq_len, d_model]
                mask: [batch_size, seq_len, seq_len] or [batch_size, 1, seq_len]
            Returns:
                output: [batch_size, seq_len, d_model]
            """
            batch_size = query.size(0)

            # Linear projections and split into multiple heads
            Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
            V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

            # Adjust mask dimensions for multi-head attention
            if mask is not None:
                if mask.dim() == 2:  # [batch_size, seq_len]
                    mask = mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
                elif mask.dim() == 3:  # [batch_size, seq_len, seq_len]
                    mask = mask.unsqueeze(1)  # [batch_size, 1, seq_len, seq_len]

            # Apply attention
            output, attention = self.scaled_dot_product_attention(Q, K, V, mask)

            # Concatenate heads and apply final linear projection
            output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
            output = self.W_o(output)

            return output


    class PositionWiseFeedForward(nn.Module):
        """Position-wise feed-forward network"""

        def __init__(self, d_model, d_ff, dropout=0.1):
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)
            self.relu = nn.ReLU()

        def forward(self, x):
            """
            Args:
                x: [batch_size, seq_len, d_model]
            Returns:
                output: [batch_size, seq_len, d_model]
            """
            return self.linear2(self.dropout(self.relu(self.linear1(x))))


    class EncoderLayer(nn.Module):
        """Single encoder layer"""

        def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
            super().__init__()

            # Multi-head self-attention
            self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

            # Feed-forward network
            self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            """
            Args:
                x: [batch_size, seq_len, d_model]
                mask: [batch_size, seq_len]
            Returns:
                output: [batch_size, seq_len, d_model]
            """
            # Self-attention with residual connection and layer norm
            attn_output = self.self_attention(x, x, x, mask)
            x = self.norm1(x + self.dropout1(attn_output))

            # Feed-forward with residual connection and layer norm
            ff_output = self.feed_forward(x)
            x = self.norm2(x + self.dropout2(ff_output))

            return x


    class DecoderLayer(nn.Module):
        """Single decoder layer"""

        def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
            super().__init__()

            # Masked multi-head self-attention
            self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

            # Multi-head cross-attention
            self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)

            # Feed-forward network
            self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
            self.norm3 = nn.LayerNorm(d_model)
            self.dropout3 = nn.Dropout(dropout)

        def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
            """
            Args:
                x: [batch_size, tgt_len, d_model]
                encoder_output: [batch_size, src_len, d_model]
                src_mask: [batch_size, src_len] - padding mask for source
                tgt_mask: [batch_size, tgt_len, tgt_len] - causal mask for target
            Returns:
                output: [batch_size, tgt_len, d_model]
            """
            # Masked self-attention with residual connection and layer norm
            self_attn_output = self.self_attention(x, x, x, tgt_mask)
            x = self.norm1(x + self.dropout1(self_attn_output))

            # Cross-attention with residual connection and layer norm
            cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
            x = self.norm2(x + self.dropout2(cross_attn_output))

            # Feed-forward with residual connection and layer norm
            ff_output = self.feed_forward(x)
            x = self.norm3(x + self.dropout3(ff_output))

            return x


    class Encoder(nn.Module):
        """Transformer encoder"""

        def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1, max_len=100):
            super().__init__()

            self.d_model = d_model
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

            self.layers = nn.ModuleList([
                EncoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])

            self.dropout = nn.Dropout(dropout)

        def forward(self, x, mask=None):
            """
            Args:
                x: [batch_size, seq_len]
                mask: [batch_size, seq_len]
            Returns:
                output: [batch_size, seq_len, d_model]
            """
            # Embedding and positional encoding
            x = self.embedding(x) * math.sqrt(self.d_model)
            x = self.positional_encoding(x)

            # Pass through encoder layers
            for layer in self.layers:
                x = layer(x, mask)

            return x


    class Decoder(nn.Module):
        """Transformer decoder"""

        def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1, max_len=100):
            super().__init__()

            self.d_model = d_model
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.positional_encoding = PositionalEncoding(d_model, max_len, dropout)

            self.layers = nn.ModuleList([
                DecoderLayer(d_model, num_heads, d_ff, dropout)
                for _ in range(num_layers)
            ])

            self.dropout = nn.Dropout(dropout)

        def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
            """
            Args:
                x: [batch_size, tgt_len]
                encoder_output: [batch_size, src_len, d_model]
                src_mask: [batch_size, src_len]
                tgt_mask: [batch_size, tgt_len, tgt_len]
            Returns:
                output: [batch_size, tgt_len, d_model]
            """
            # Embedding and positional encoding
            x = self.embedding(x) * math.sqrt(self.d_model)
            x = self.positional_encoding(x)

            # Pass through decoder layers
            for layer in self.layers:
                x = layer(x, encoder_output, src_mask, tgt_mask)

            return x


    class Transformer(nn.Module):
        """Complete Transformer model (encoder-decoder)"""

        def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, num_heads=4,
                     d_ff=512, num_encoder_layers=3, num_decoder_layers=3,
                     dropout=0.1, max_len=100):
            super().__init__()

            self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff,
                                   num_encoder_layers, dropout, max_len)
            self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff,
                                   num_decoder_layers, dropout, max_len)

            self.fc_out = nn.Linear(d_model, tgt_vocab_size)

            # Initialize parameters
            self._init_parameters()

        def _init_parameters(self):
            """Initialize parameters with Xavier uniform"""
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def generate_square_subsequent_mask(self, sz, device):
            """Generate causal mask for decoder"""
            mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
            mask = mask == 0  # Convert to boolean (True where we can attend)
            return mask

        def create_padding_mask(self, seq):
            """Create padding mask (True where there's actual data, False for padding)"""
            return seq != 0  # Assuming 0 is padding token

        def forward(self, src, tgt):
            """
            Args:
                src: [batch_size, src_len]
                tgt: [batch_size, tgt_len]
            Returns:
                output: [batch_size, tgt_len, vocab_size]
            """
            # Create masks
            src_mask = self.create_padding_mask(src)  # [batch_size, src_len]
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)  # [tgt_len, tgt_len]
            tgt_mask = tgt_mask.unsqueeze(0).expand(tgt.size(0), -1, -1)  # [batch_size, tgt_len, tgt_len]

            # Combine causal mask with padding mask
            tgt_padding_mask = self.create_padding_mask(tgt)  # [batch_size, tgt_len]
            tgt_padding_mask = tgt_padding_mask.unsqueeze(1)  # [batch_size, 1, tgt_len]
            tgt_mask = tgt_mask & tgt_padding_mask  # Broadcast and combine

            # Encode
            encoder_output = self.encoder(src, src_mask)

            # Decode
            decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)

            # Project to vocabulary
            output = self.fc_out(decoder_output)

            return output

        def encode(self, src):
            """Encode source sequence"""
            src_mask = self.create_padding_mask(src)
            return self.encoder(src, src_mask), src_mask

        def decode_step(self, tgt, encoder_output, src_mask):
            """Decode one step (for inference)"""
            tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)
            tgt_mask = tgt_mask.unsqueeze(0).expand(tgt.size(0), -1, -1)

            tgt_padding_mask = self.create_padding_mask(tgt)
            tgt_padding_mask = tgt_padding_mask.unsqueeze(1)
            tgt_mask = tgt_mask & tgt_padding_mask

            decoder_output = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
            output = self.fc_out(decoder_output)

            return output

    return (Transformer,)


@app.cell
def _(math, random, torch):
    def train_epoch(model, dataloader, optimizer, criterion, device):
        model.train()
        total_loss = 0

        for batch in dataloader:
            src = batch['encoder_input'].to(device)
            tgt_input = batch['decoder_input'].to(device)
            tgt_output = batch['decoder_target'].to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(src, tgt_input)

            # Reshape for loss calculation
            output = output.reshape(-1, output.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            # Calculate loss (ignore padding)
            loss = criterion(output, tgt_output)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)


    def evaluate(model, dataloader, criterion, device):
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in dataloader:
                src = batch['encoder_input'].to(device)
                tgt_input = batch['decoder_input'].to(device)
                tgt_output = batch['decoder_target'].to(device)

                output = model(src, tgt_input)

                output = output.reshape(-1, output.shape[-1])
                tgt_output = tgt_output.reshape(-1)

                loss = criterion(output, tgt_output)
                total_loss += loss.item()

        return total_loss / len(dataloader)


    def greedy_decode(model, src, dataset, max_len=10, device='cpu'):
        """Greedy decoding for inference"""
        model.eval()

        with torch.no_grad():
            src = src.to(device)

            # Encode
            encoder_output, src_mask = model.encode(src)

            # Start with SOS token
            sos_idx = dataset.token2idx[dataset.SOS_TOKEN]
            eos_idx = dataset.token2idx[dataset.EOS_TOKEN]

            tgt = torch.tensor([[sos_idx]], dtype=torch.long, device=device)

            for _ in range(max_len):
                # Decode
                output = model.decode_step(tgt, encoder_output, src_mask)

                # Get next token
                next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)

                # Append to target
                tgt = torch.cat([tgt, next_token], dim=1)

                # Stop if EOS
                if next_token.item() == eos_idx:
                    break

            return tgt.squeeze(0).cpu().numpy()


    def test_model(model, dataset, device, num_tests=5):
        """Test the model with random examples"""
        print("\n" + "="*50)
        print("Testing the model:")
        print("="*50)

        correct = 0
        for _ in range(num_tests):
            # Generate random number
            num = random.randint(1, 99999)
            actual_sqrt = round(math.sqrt(num))

            # Prepare input
            input_seq = list(str(num))
            encoder_input = torch.tensor([dataset.encode_sequence(input_seq)], dtype=torch.long)

            # Decode
            prediction = greedy_decode(model, encoder_input, dataset, device=device)
            predicted_sqrt = dataset.decode_sequence(prediction)

            is_correct = str(actual_sqrt) == predicted_sqrt
            if is_correct:
                correct += 1

            print(f"Input: {num:5d} | Actual √: {actual_sqrt:3d} | Predicted: {predicted_sqrt:>3s} | {'✓' if is_correct else '✗'}")

        print(f"\nAccuracy: {correct}/{num_tests} ({100*correct/num_tests:.1f}%)")

    return evaluate, greedy_decode, test_model, train_epoch


@app.cell
def _(
    DataLoader,
    SqrtDataset,
    Transformer,
    collate_fn,
    evaluate,
    greedy_decode,
    math,
    nn,
    optim,
    test_model,
    torch,
    train_epoch,
):
    # Hyperparameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    D_MODEL = 128
    NUM_HEADS = 4
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    D_FF = 512
    DROPOUT = 0.1

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create dataset
    print("\nGenerating dataset...")
    train_dataset = SqrtDataset(num_samples=20000, max_num=99999)
    val_dataset = SqrtDataset(num_samples=2000, max_num=99999)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f"Vocabulary size: {train_dataset.vocab_size}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create model
    print("\nBuilding Transformer from scratch...")
    model = Transformer(
        src_vocab_size=train_dataset.vocab_size,
        tgt_vocab_size=train_dataset.vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dropout=DROPOUT
    ).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)

    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"Epoch {epoch+1:2d}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), './best_sqrt_transformer.pt')

    # Load best model and test
    print("\nLoading best model for testing...")
    model.load_state_dict(torch.load('./best_sqrt_transformer.pt'))
    test_model(model, train_dataset, device, num_tests=10)

    print("\n" + "="*50)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*50)

    # Test with specific examples
    print("\n" + "="*50)
    print("Testing with specific examples:")
    print("="*50)
    test_numbers = [1, 4, 9, 16, 25, 100, 1000, 10000, 12345, 99999]
    for num in test_numbers:
        actual_sqrt = round(math.sqrt(num))
        input_seq = list(str(num))
        encoder_input = torch.tensor([train_dataset.encode_sequence(input_seq)], dtype=torch.long)
        prediction = greedy_decode(model, encoder_input, train_dataset, device=device)
        predicted_sqrt = train_dataset.decode_sequence(prediction)
        is_correct = str(actual_sqrt) == predicted_sqrt
        print(f"√{num:5d} = {actual_sqrt:3d} | Predicted: {predicted_sqrt:>3s} | {'✓' if is_correct else '✗'}")


    return


if __name__ == "__main__":
    app.run()
