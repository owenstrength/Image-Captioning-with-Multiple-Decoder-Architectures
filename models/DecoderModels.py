import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_dim = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = torch.zeros(1, 1, hidden_size)

    def forward(self, features, captions):
        cap_embedding = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), cap_embedding), dim=1)
        rnn_out, self.hidden = self.rnn(embeddings)
        outputs = self.linear(rnn_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        res = []
        for i in range(max_len):
            rnn_out, states = self.rnn(inputs, states)
            outputs = self.linear(rnn_out.squeeze(1))
            _, predicted_idx = outputs.max(1)
            res.append(predicted_idx.item())
            if predicted_idx == 1:
                break
            inputs = self.embed(predicted_idx)
            inputs = inputs.unsqueeze(1)
        return res

class DecoderGRU(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderGRU, self).__init__()
        
        self.hidden_dim = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = torch.zeros(1, 1, hidden_size)

    def forward(self, features, captions):
        cap_embedding = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), cap_embedding), dim=1)
        gru_out, self.hidden = self.gru(embeddings)
        outputs = self.linear(gru_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        res = []
        for i in range(max_len):
            gru_out, states = self.gru(inputs, states)
            outputs = self.linear(gru_out.squeeze(1))
            _, predicted_idx = outputs.max(1)
            res.append(predicted_idx.item())
            if predicted_idx == 1:
                break
            inputs = self.embed(predicted_idx)
            inputs = inputs.unsqueeze(1)
        return res

class DecoderLSTMAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderLSTMAttention, self).__init__()

        self.hidden_dim = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_score = nn.Linear(hidden_size, 1)  # For attention scores
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        cap_embedding = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), cap_embedding), dim=1)
        lstm_out, _ = self.lstm(embeddings)
        context_vectors = []
        for t in range(lstm_out.size(1)):
            attention_scores = self.attention_score(torch.tanh(self.attention(torch.cat((lstm_out[:, t:t+1], features.unsqueeze(1)), dim=2))))
            attention_weights = F.softmax(attention_scores, dim=1)
            context_vector = torch.sum(attention_weights * features.unsqueeze(1), dim=1)
            context_vectors.append(context_vector)

        context_vectors = torch.stack(context_vectors, dim=1)
        combined = lstm_out + context_vectors  # Residual connection
        outputs = self.linear(combined)
        return outputs

    def sample(self, inputs, features, states=None, max_len=20):
        res = []
        for _ in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            attention_scores = self.attention_score(torch.tanh(self.attention(torch.cat((lstm_out, features.unsqueeze(1)), dim=2))))
            attention_weights = F.softmax(attention_scores, dim=1)
            context_vector = torch.sum(attention_weights * features.unsqueeze(1), dim=1)
            combined = lstm_out.squeeze(1) + context_vector
            outputs = self.linear(combined)
            _, predicted_idx = outputs.max(1)
            res.append(predicted_idx.item())
            if predicted_idx == 1:  
                break
            inputs = self.embed(predicted_idx).unsqueeze(1)

        return res

class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderLSTM, self).__init__()
        
        self.hidden_dim = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))

    def forward(self, features, captions):
        cap_embedding = self.embed(captions[:, :-1])
        embeddings = torch.cat((features.unsqueeze(1), cap_embedding), dim=1)
        lstm_out, self.hidden = self.lstm(embeddings)
        outputs = self.linear(lstm_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        res = []
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            outputs = self.linear(lstm_out.squeeze(1))
            _, predicted_idx = outputs.max(1)
            res.append(predicted_idx.item())
            if predicted_idx == 1:
                break
            inputs = self.embed(predicted_idx)
            inputs = inputs.unsqueeze(1)
        return res