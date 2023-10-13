import torch.nn as nn
import torch.nn.functional as F


# Multilayer Perceptron with hidden layers
class MLP(nn.Module):

    def __init__(self, input_size: int, hidden_sizes: list, dropout: float):
        """
        Initialize fullyConnectedNet.
        Parameters
        ----------
        input_size – The number of expected features in the input x
        hidden_size – The numbers of features in the hidden layer h
        input -> (batch, in_features)
        :return
        output -> (batch, out_features)
        """
        super(MLP, self).__init__()

        self.input_size = input_size
        # list
        self.hidden_sizes = hidden_sizes

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc_list = nn.ModuleList()
        self.fc_list.append(nn.Linear(input_size, self.hidden_sizes[0]))
        for index in range(1, len(self.hidden_sizes)):
            self.fc_list.append(nn.Linear(self.hidden_sizes[index-1], self.hidden_sizes[index]))

    def forward(self, input_tensor):

        """
        :param input_tensor:
            2-D Tensor  (batch, input_size)
        :return:
            2-D Tensor (batch, hidden_sizes[-1])
            output_tensor
        """
        for fc in self.fc_list:
            input_tensor = self.relu(fc(input_tensor))
            input_tensor = self.dropout(input_tensor)

        return input_tensor
