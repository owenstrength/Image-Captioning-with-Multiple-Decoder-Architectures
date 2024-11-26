import torch
import torch.nn as nn

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_dim = hidden_size
        self.embed = nn.Embedding(vocab_size, embed_size)
        # Test a GRU instead of LSTM
        self.lstm = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True)
        #self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.hidden = (torch.zeros(1, 1, hidden_size), torch.zeros(1, 1, hidden_size))

    def forward(self, features, captions):
        # Remove end token and embed captions
        cap_embedding = self.embed(captions[:, :-1])
        
        # Add image features as first token
        embeddings = torch.cat((features.unsqueeze(1), cap_embedding), dim=1)
        
        # LSTM forward pass
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
            
            if predicted_idx == 1:  # End token
                break
                
            inputs = self.embed(predicted_idx)
            inputs = inputs.unsqueeze(1)
            
        return res