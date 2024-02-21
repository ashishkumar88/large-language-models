# Implementation of a vanilla Transformer model for sequence generation using PyTorch

import torch
import torch.nn as nn
import torch.optim as optim
import math
import spacy
import os

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

torch.manual_seed(42)  # For reproducibility
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

class MultiHeadAttention(nn.Module):
    """The multi-head attention module"""
    def __init__(self, d_model, num_heads):
        super().__init__() 
        
        # Ensure the dimension of the model is divisible by the number of heads.
        # This is necessary to equally divide the embedding dimension across heads.
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        
        self.d_model = d_model           # Total dimension of the model
        self.num_heads = num_heads       # Number of attention heads
        self.d_k = d_model // num_heads  # Dimnsion of each head. We assume d_v = d_k
               
        # Linear transformations for queries, keys, and values
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # Final linear layer to project the concatenated heads' outputs back to d_model dimensions
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        
        # 1. Calculate attention scores with scaling
        assert Q.size(-1) == K.size(-1) == self.d_k # Make sure the dimension of Q and K are the same
        attention_weights = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # weights = Q * K^T / sqrt(d_k)

        # 2. Apply mask (if provided) by setting masked positions to a large negative value
        if mask is not None:
            attention_weights = attention_weights.masked_fill(mask == 0, -1e9) # mask is 0 where we want to ignore

        # 3. Apply softmax to attention scores to get probabilities
        attention_probs = attention_weights.softmax(dim=-1) # softmax over the last dimension

        # 4. Return the weighted sum of values based on attention probabilities
        output = torch.matmul(attention_probs, V)
        
        return output
    
    def split_heads(self, x):
        # Reshape the input tensor to [batch_size, num_heads, seq_length, d_k]
        # to prepare for multi-head attention processing
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        # Inverse operation of split_heads: combine the head outputs back into the original tensor shape
        # [batch_size, seq_length, d_model]
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # 1. Linearly project the queries, keys, and values, and then split them into heads
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 2. Apply scaled dot-product attention for each head 
        output = self.scaled_dot_product_attention(Q, K, V, mask)

        # 3. Concatenate the heads' outputs and apply the final linear projection
        output = self.combine_heads(output) 
        output = self.W_o(output)
        
        return output

class PositionwiseFeedForward(nn.Module):
    """The Positionwise Feedforward Network (FFN) module"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()       
        self.linear1 = nn.Linear(d_model, d_ff)        
        self.linear2 = nn.Linear(d_ff, d_model)        
        self.dropout = nn.Dropout(dropout)        
        self.relu = nn.ReLU()

    def forward(self, x):
        inner = self.dropout(self.relu(self.linear1(x)))
        output = self.linear2(inner) 

        return output

class PositionalEncoding(nn.Module):    
    """
    Implements the positional encoding module using sinusoidal functions of different frequencies 
    for each dimension of the encoding.
    """
    def __init__(self, d_model, max_seq_length):
        super().__init__()        
        
        # Create a positional encoding (PE) matrix with dimensions [max_seq_length, d_model].
        # This matrix will contain the positional encodings for all possible positions up to max_seq_length.
        pe = torch.zeros(max_seq_length, d_model)
        
        # Generate a tensor of positions (0 to max_seq_length - 1) and reshape it to [max_seq_length, 1].
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Compute the division term used in the formulas for sin and cos functions.
        # This term is based on the dimension of the model and the position, ensuring that the wavelengths
        # form a geometric progression from 2π to 10000 * 2π. It uses only even indices for the dimensions.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply the sin function to even indices in the PE matrix. These values are determined by
        # multiplying the position by the division term, creating a pattern where each position has
        # a unique sinusoidal encoding.       
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply the cos function to odd indices in the PE matrix, complementing the sin-encoded positions.
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register 'pe' as a buffer within the module. Unlike parameters, buffers are not updated during training.
        # This is crucial because positional encodings are fixed and not subject to training updates.
        # The unsqueeze(0) adds a batch dimension for easier broadcasting with input tensors.
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Add positional encoding to the input tensor x.
        # x is expected to have dimensions [batch_size, seq_length, d_model].
        # The positional encoding 'pe' is sliced to match the seq_length of 'x', and then added to 'x'.
        # This operation leverages broadcasting to apply the same positional encoding across the batch.
        x = x + self.pe[:, :x.size(1)]
        return x

class EncoderLayer(nn.Module):
    """An encoder layer consists of a multi-head self-attention sublayer and a feed forward sublayer,
       with a dropout, residual connection, and layer normalization after each sub-layer.    
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):

        # Apply the self-attention sublayer and pass the output through a layer normalization and dropout
        attn_sublayer = lambda x: self.self_attn(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attn_sublayer(x)))
        output = self.layer_norm2(x + self.dropout(self.feed_forward(x)))
        
        return output

class DecoderLayer(nn.Module):
    """A decoder layer consists of a multi-head self-attention, cross-attention and a feed-forward sublayers,
       with a dropout, residual connection, and layer normalization after each sub-layer.    
    """
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)       

    def forward(self, x, enc_output, src_mask, tgt_mask):

        # Apply the self-attention sublayer and cross attention, pass the output through a layer normalization and dropout
        attn_sublayer_1 = lambda x: self.self_attn(x, x, x, tgt_mask)
        attn_sublayer_2 = lambda x: self.cross_attn(x, enc_output, enc_output, src_mask)

        x = self.layer_norm1(x + self.dropout(attn_sublayer_1(x)))
        x = self.layer_norm2(x + self.dropout(attn_sublayer_2(x)))
        output = self.layer_norm3(x + self.dropout(self.feed_forward(x)))

        return output

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, N, num_heads, d_ff, max_seq_length, dropout, pad_idx):
        super().__init__()

        # Embedding layers for source and target
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        # Encoder and Decoder stacks
        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(N)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(N)])

        # Output linear layer
        self.out = nn.Linear(d_model, tgt_vocab_size)
        
        self.dropout = nn.Dropout(dropout)

        # Initialization
        self.init_weights()
        self.pad_idx = pad_idx

    def init_weights(self):
        """Initialize parameters with Glorot / fan_avg"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def create_source_mask(self, src):
        """Create masks for both padding tokens and future tokens"""        
        # Source padding mask
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, src_len]
        # unsqueeze(1) adds a dimension for the heads of the multi-head attention
        # unsqueeze(2) adds a dimension for the attention scores 
        # This mask can be broadcasted across the src_len dimension of the attention scores, 
        # effectively masking out specific tokens across all heads and all positions in the sequence. 
        return src_mask    
    
    def create_target_mask(self, tgt):
        # Target padding mask
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(3)  # [batch_size, 1, tgt_len, 1]
        # unsqueeze(1) adds a dimension for the heads of the multi-head attention
        # unsqueeze(3) adds a dimension for the attention scores
        # The final shape allows the mask to be broadcast across the attention scores, ensuring positions only 
        # attend to allowed positions as dictated by the no-peak mask (the preceding positions) and the padding mask.
                
        # Target no-peak mask
        tgt_len = tgt.size(1)        
        tgt_nopeak_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).bool()
        
        # Combine masks
        tgt_mask = tgt_pad_mask & tgt_nopeak_mask  # [batch_size, 1, tgt_len, tgt_len]        
        return tgt_mask 
        
    def encode(self, src):
        """Encodes the source sequence using the Transformer encoder stack.
        """       
        src_mask = self.create_source_mask(src)
        src = self.dropout(self.positional_encoding(self.src_embedding(src)))
        
        # Pass through each layer in the encoder        
        for layer in self.encoder:
            src = layer(src, src_mask)
        return src, src_mask
        
    def decode(self, tgt, memory, src_mask):
        """Decodes the target sequence using the Transformer decoder stack, given the memory from the encoder.
        """
        tgt_mask = self.create_target_mask(tgt)
        tgt = self.dropout(self.positional_encoding(self.tgt_embedding(tgt)))
        
        # Pass through each layer in the decoder
        for layer in self.decoder:
            tgt = layer(tgt, memory, src_mask, tgt_mask)

        # Output layer
        output = self.out(tgt)
        return output

    def forward(self, src, tgt):
        memory, src_mask = self.encode(src)
        output = self.decode(tgt, memory, src_mask)        
    
        return output

# Load spacy models for tokenization
try:
    spacy_de = spacy.load('de_core_news_sm')
except IOError:
    os.system("python -m spacy download de_core_news_sm")
    spacy_de = spacy.load('de_core_news_sm')

try:
    spacy_en = spacy.load('en_core_web_sm')
except IOError:
    os.system("python -m spacy download en_core_web_sm")
    spacy_en = spacy.load('en_core_web_sm')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

def yield_tokens(data_iter, tokenizer, language):
    for data_sample in data_iter:
        yield tokenizer(data_sample[language])

tokenizer_de = get_tokenizer(tokenize_de)
tokenizer_en = get_tokenizer(tokenize_en)

# #### Build Vocabularies
train_data, _, _ = Multi30k(split=('train', 'valid', 'test'))
vocab_src = build_vocab_from_iterator(yield_tokens(train_data, tokenizer_de, 0), 
                                      specials=['<unk>', '<pad>', '<bos>', '<eos>'])
vocab_tgt = build_vocab_from_iterator(yield_tokens(train_data, tokenizer_en, 1), 
                                      specials=['<unk>', '<pad>', '<bos>', '<eos>'])

vocab_src.set_default_index(vocab_src['<unk>'])
vocab_tgt.set_default_index(vocab_tgt['<unk>'])

# #### Create the Transformer

# Define the hyperparameters of the model
src_vocab_size = len(vocab_src)  # Size of source vocabulary
tgt_vocab_size = len(vocab_tgt)  # Size of target vocabulary
d_model = 512  # Embedding dimension
N = 6          # Number of encoder and decoder layers
num_heads = 8  # Number of attention heads
d_ff = 2048    # Dimension of feed forward networks
max_seq_length = 5000 # Maximum sequence length
dropout = 0.1  # Dropout rate

# Assume pad_idx is the padding index in the target vocabulary
pad_idx = vocab_tgt['<pad>']

# Initialize the Transformer model
model = Transformer(src_vocab_size, tgt_vocab_size, d_model, N, num_heads, d_ff, max_seq_length, dropout, pad_idx)

# Move the model to the appropriate device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Hyperparameters for the training process
batch_size = 128
grad_clip = 1
optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Initialize the loss function with CrossEntropyLoss, ignoring the padding index
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# #### Data Processing
def data_process(raw_data_iter):
    data = []
    for raw_src, raw_tgt in raw_data_iter:
        src_tensor = torch.tensor([vocab_src[token] for token in tokenizer_de(raw_src)], dtype=torch.long)
        tgt_tensor = torch.tensor([vocab_tgt[token] for token in tokenizer_en(raw_tgt)], dtype=torch.long)
        data.append((src_tensor, tgt_tensor))
    return data

train_data, valid_data, test_data = Multi30k(split=('train', 'valid', 'test'))
train_data = data_process(train_data)
valid_data = data_process(valid_data)
#test_data = data_process(test_data)   
# The test set of Multi30k is corrupted
# See https://discuss.pytorch.org/t/unicodedecodeerror-when-running-test-iterator/192818/3

def generate_batch(data_batch):
    """Processes a batch of source-target pairs by adding start-of-sequence (BOS) and end-of-sequence (EOS) tokens
    to each sequence and padding all sequences to the same length.
    
    Parameters:
    - data_batch (Iterable[Tuple[Tensor, Tensor]]): A batch of source-target pairs, where each element is a tuple
      containing the source sequence tensor and the target sequence tensor.
    """
    src_batch, tgt_batch = [], []
    src_batch, tgt_batch = [], []
    
    # Iterate over each source-target pair in the provided batch
    for src_item, tgt_item in data_batch:
        # Prepend the start-of-sequence (BOS) token and append the end-of-sequence (EOS) token to the sequences        
        src_batch.append(torch.cat([torch.tensor([vocab_src['<bos>']]), src_item, 
                                    torch.tensor([vocab_src['<eos>']])], dim=0))
        tgt_batch.append(torch.cat([torch.tensor([vocab_tgt['<bos>']]), tgt_item, 
                                    torch.tensor([vocab_tgt['<eos>']])], dim=0))
        
    # Pad the sequences in the source batch to ensure they all have the same length.
    # 'batch_first=True' indicates that the batch dimension should come first in the resulting tensor.
    src_batch = pad_sequence(src_batch, padding_value=vocab_src['<pad>'], batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=vocab_tgt['<pad>'], batch_first=True)
    return src_batch, tgt_batch

# DataLoader for the training data, using the generate_batch function as the collate_fn.
# This allows custom processing of each batch (adding BOS/EOS tokens and padding) before being fed into the model.
train_iterator = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)

# Similarly, DataLoader for the validation data
valid_iterator = DataLoader(valid_data, batch_size=batch_size, shuffle=True, collate_fn=generate_batch)

def train(model, iterator, optimizer, criterion, grad_clip):
    """
    Trains the model for one epoch over the given dataset.
    This function iterates over the provided data iterator, performing the forward and backward passes for each batch.
    It employs teacher forcing by feeding the shifted target sequence (excluding the last token) as input to the decoder.
    
    Parameters:
    - model (torch.nn.Module): The model to be trained. 
    - iterator (Iterable): An iterable object that returns batches of data. 
    - optimizer (torch.optim.Optimizer): The optimizer to use for updating the model parameters.
    - criterion (Callable): The loss function used to compute the difference between the model's predictions and the actual targets.
    - grad_clip (float): The maximum norm of the gradients for gradient clipping. 

    Returns:
    - float: The average loss for the epoch, computed as the total loss over all batches divided by the number of batches in the iterator.
    """    
    # Set the model to training mode. 
    # This enables dropout, layer normalization etc., which behave differently during training.
    model.train()   
    
    epoch_loss = 0
    
    # Enumerate over the data iterator to get batches
    for i, batch in enumerate(iterator):         
        # Unpack the batch to get source (src) and target (tgt) sequences
        src, tgt = batch
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        
        # Forward pass through the model. 
        # For seq2seq models, the decoder input (tgt[:, :-1]) excludes the last token, implementing teacher forcing.
        output = model(src, tgt[:, :-1])
        
        # Reshape the output and target tensors to compute loss.
        # The output tensor is reshaped to a 2D tensor where rows correspond to each token in the batch and columns to vocabulary size.
                
        # tgt is of shape [batch_size, tgt_len]
        # output is of shape [batch_size, tgt_len, tgt_vocab_size]
        output = output.contiguous().view(-1, tgt_vocab_size)
        
        # The target tensor is reshaped to a 1D tensor, excluding the first token (BOS) from each sequence.
        tgt = tgt[:, 1:].contiguous().view(-1)
        
        # Compute loss, perform backpropagation, and update model parameters
        loss = criterion(output, tgt)          
        loss.backward() 
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  
        optimizer.step()        
        epoch_loss += loss.item()
        
    # Compute average loss per batch for the current epoch
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    """
    Evaluates the model's performance on a given dataset.
    This function is similar to the training loop, but without the backward pass and parameter updates. I
    """
    model.eval()
    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src, tgt = batch
            src, tgt = src.to(device), tgt.to(device)
            output = model(src, tgt[:, :-1])            
            output_dim = output.shape[-1]            
            output = output.contiguous().view(-1, output_dim)
            tgt = tgt[:, 1:].contiguous().view(-1)
            loss = criterion(output, tgt)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

# #### Training the Model
n_epochs = 40

for epoch in range(n_epochs):
    train_loss = train(model, train_iterator, optimizer, criterion, grad_clip)
    val_loss = evaluate(model, valid_iterator, criterion)

    print(f'\nEpoch: {epoch + 1}')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\tVal Loss: {val_loss:.3f}')

# #### Translating a Sample Sentence
def translate_sentence(model, sentence, vocab_src, vocab_tgt, max_length=50):
    """
    Translates a given source sentence into the target language using a trained Transformer model.
    The function preprocesses the input sentence by tokenizing and converting it to tensor format, then uses the model's
    encode and decode methods to generate the translated sentence. The translation process is performed token by token
    using greedy decoding, selecting the most likely next token at each step until an <eos> token is produced or the
    maximum length is reached.

    Parameters:
    - model (torch.nn.Module): The trained Transformer model. 
    - sentence (str): The source sentence to translate. 
    - vocab_src (dict): The source vocabulary mapping of tokens to indices. It should include special tokens such as
      '<bos>' (beginning of sentence) and '<eos>' (end of sentence).
    - vocab_tgt (dict): The target vocabulary mapping of indices to tokens. It should provide a method `lookup_token`
      to convert token indices back to the string representation.
    - max_length (int, optional): The maximum allowed length for the generated translation. The decoding process will
      stop when this length is reached if an <eos> token has not yet been generated.

    Returns:
    - str: The translated sentence as a string of text in the target language.
    """ 

    # Tokenize the input sentence and convert it to a tensor
    src_tensor = torch.tensor([vocab_src[token] for token in tokenizer_de(sentence)], dtype=torch.long).unsqueeze(0).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Encode the source tokens
    with torch.no_grad():
        memory, src_mask = model.encode(src_tensor)

    # Stores the previous token
    output_tokens = [vocab_tgt['<bos>']]   

    # Iterate over the maximum length
    for _ in range(max_length):         
        tgt_tensor = torch.tensor(output_tokens, dtype=torch.long).unsqueeze(0).to(device)   

        # Decode the sequence
        with torch.no_grad():            
            logits = model.decode(tgt_tensor, memory, src_mask)       

        # change logits to probs
        probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        next_token = probs.argmax(2)[:,-1].item()   

        # If the next token is the end of sentence token, stop the translation
        if next_token == vocab_tgt['<eos>']:             
               break               

        # Append the next token to the output tokens
        output_tokens.append(next_token)       

    # Convert the output token indices to text
    translated_sentence = ' '.join([vocab_tgt.lookup_token(token) for token in output_tokens[1:]])
    
    return translated_sentence

src_sentence = "Ein kleiner Junge spielt draußen mit einem Ball."  # German for "A little boy playing outside with a ball."
translated_sentence = translate_sentence(model, src_sentence, vocab_src, vocab_tgt)
print(f'Translated sentence: {translated_sentence}')

def translate_sentence_with_beam_search(model, sentence, vocab_src, vocab_tgt, max_length=50, beam_width=5):
    """
    Translates a given source sentence into the target language using a trained Transformer model and beam search.
    The function preprocesses the input sentence by tokenizing and converting it to tensor format, then uses the model's
    encode and decode methods to generate the translated sentence. The translation process is performed using beam search,
    which maintains a list of the most likely partial translations at each step and expands them by considering the next
    possible tokens. The beam width determines the number of partial translations to maintain at each step.

    Parameters:
    - model (torch.nn.Module): The trained Transformer model.
    - sentence (str): The source sentence to translate.
    - vocab_src (dict): The source vocabulary mapping of tokens to indices. It should include special tokens such as
      '<bos>' (beginning of sentence) and '<eos>' (end of sentence).
    - vocab_tgt (dict): The target vocabulary mapping of indices to tokens. It should provide a method `lookup_token`
        to convert token indices back to the string representation.
    - max_length (int, optional): The maximum allowed length for the generated translation. The decoding process will
        stop when this length is reached if an <eos> token has not yet been generated.
    - beam_width (int, optional): The number of partial translations to maintain at each step of the decoding process.

    Returns:
    - str: The translated sentence as a string of text in the target language.
    """
    
    # Tokenize the input sentence and convert it to a tensor
    src_tensor = torch.tensor([vocab_src[token] for token in tokenizer_de(sentence)], dtype=torch.long).unsqueeze(0).to(device)

    # Set the model to evaluation mode
    model.eval()

    # Encode the source tokens
    with torch.no_grad():
        memory, src_mask = model.encode(src_tensor)

    # Stores the previous token
    output_tokens = [vocab_tgt['<bos>']]

    # Stores the list of top beam_width partial translations
    partial_translations = [(output_tokens, 0)]

    # Iterate over the maximum length
    for _ in range(max_length):
        new_partial_translations = []

        # Iterate over the current partial translations
        for tokens, score in partial_translations:
            tgt_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

            # Decode the sequence
            with torch.no_grad():
                logits = model.decode(tgt_tensor, memory, src_mask)

            # change logits to probs
            probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Get the top beam_width next tokens
            top_tokens = torch.topk(probs, beam_width, dim=2)

            # Iterate over the top tokens
            for i in range(beam_width):
                next_token = top_tokens.indices[0, -1, i].item()
                new_score = score + top_tokens.values[0, -1, i].item()

                # If the next token is the end of sentence token, stop the translation
                if next_token == vocab_tgt['<eos>']:
                    return ' '.join([vocab_tgt.lookup_token(token) for token in tokens[1:]])

                # Append the next token to the output tokens
                new_tokens = tokens + [next_token]
                new_partial_translations.append((new_tokens, new_score))

        # Select the top beam_width partial translations
        partial_translations = sorted(new_partial_translations, key=lambda x: x[1], reverse=True)[:beam_width]

    # Convert the output token indices to text
    translated_sentence = ' '.join([vocab_tgt.lookup_token(token) for token in partial_translations[0][0][1:]])
     
    return translated_sentence

src_sentence = "Ein kleiner Junge spielt draußen mit einem Ball."  # German for "A little boy playing outside with a ball."
translated_sentence = translate_sentence_with_beam_search(model, src_sentence, vocab_src, vocab_tgt)
print(f'Translated sentence: {translated_sentence}')

ground_truth_english_sentences = [
    "A man in an orange hat starring at something.	", 
    "A Boston Terrier is running on lush green grass in front of a white fence.	", 
    "A girl in karate uniform breaking a stick with a front kick.	", 
    "Five people wearing winter jackets and helmets stand in the snow, with snowmobiles in the background.	", 
    "People are fixing the roof of a house.	", 
    "A man in light colored clothing photographs a group of men wearing dark suits and hats standing around a woman dressed in a strapless gown.	", 
    "A group of people standing in front of an igloo.	", 
    "A boy in a red uniform is attempting to avoid getting out at home plate, while the catcher in the blue uniform is attempting to catch him.	", 
    "A guy works on a building.	", 
    "A man in a vest is sitting in a chair and holding magazines.	", 
    "A mother and her young song enjoying a beautiful day outside.	", 
    "Men playing volleyball, with one player missing the ball but hands still in the air.	", 
    "A woman holding a bowl of food in a kitchen.	", 
    "Man sitting using tool at a table in his home.	", 
    "Three people sit in a cave.	", 
]

german_sentences_to_translate = [
    "Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt.", 
    "Ein Boston Terrier läuft über saftig-grünes Gras vor einem weißen Zaun.", 
    "Ein Mädchen in einem Karateanzug bricht ein Brett mit einem Tritt.", 
    "Fünf Leute in Winterjacken und mit Helmen stehen im Schnee mit Schneemobilen im Hintergrund.", 
    "Leute Reparieren das Dach eines Hauses.", 
    "Ein hell gekleideter Mann fotografiert eine Gruppe von Männern in dunklen Anzügen und mit Hüten, die um eine Frau in einem trägerlosen Kleid herum stehen.", 
    "Eine Gruppe von Menschen steht vor einem Iglu.", 
    "Ein Junge in einem roten Trikot versucht, die Home Base zu erreichen, während der Catcher im blauen Trikot versucht, ihn zu fangen.", 
    "Ein Typ arbeitet an einem Gebäude.", 
    "Ein Mann in einer Weste sitzt auf einem Stuhl und hält Magazine.", 
    "Eine Mutter und ihr kleiner Sohn genießen einen schönen Tag im Freien.", 
    "Männer, die Volleyball spielen, wobei ein Mann den Ball nicht trifft, während seine Hände immer noch in der Luft sind.", 
    "Eine Frau, die in einer Küche eine Schale mit Essen hält.", 
    "Ein sitzender Mann, der an einem Tisch in seinem Haus mit einem Werkzeug arbeitet.", 
    "Drei Leute sitzen in einer Höhle.", 
]

for i in range(len(german_sentences_to_translate)):
    src_sentence = german_sentences_to_translate[i]
    translated_sentence_with_greedy_search = translate_sentence(model, src_sentence, vocab_src, vocab_tgt)
    translated_sentence_with_beam_search = translate_sentence_with_beam_search(model, src_sentence, vocab_src, vocab_tgt)

    print("New sentence:")
    print(f"German sentence : {src_sentence}")
    print(f"Ground truth english sentence : {ground_truth_english_sentences[i]}")
    print(f"Translated using greedy search : {translated_sentence_with_greedy_search}")
    print(f"Translated using beam search : {translated_sentence_with_beam_search}")
    print("\n\n")


