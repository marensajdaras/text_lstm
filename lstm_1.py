import torch
import torch.nn as nn

class LSTMStep(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMStep, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Weights for input gate
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x_t, h_prev, c_prev):
        # Concatenate input and previous hidden state
        combined = torch.cat((x_t, h_prev), dim=1)

        # Compute input gate and candidate cell state
        i_t = self.sigmoid(self.W_i(combined))
        c_tilde = self.tanh(self.W_c(combined))

        return i_t, c_tilde

# Example usage
input_size = 4  # Example input feature size
hidden_size = 3  # Example hidden state size

lstm_step = LSTMStep(input_size, hidden_size)

# Example inputs (batch size = 1)
x_t = torch.randn(1, input_size)  # Current input
h_prev = torch.randn(1, hidden_size)  # Previous hidden state
c_prev = torch.randn(1, hidden_size)  # Previous cell state

# Compute input gate and candidate cell state
i_t, c_tilde = lstm_step(x_t, h_prev, c_prev)

print("Input gate (i_t):", i_t)
print("Candidate cell state (c~_t):", c_tilde)
