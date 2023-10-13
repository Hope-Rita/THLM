import torch
import torch.nn as nn
import math


class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input):
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input):
        return self.act(input)


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, hidden_size, hidden_act, layer_norm_eps):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        if isinstance(hidden_act, bool):
            self.transform_act_fn = GELUActivation()
        else:
            self.transform_act_fn = hidden_act
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, hidden_size, hidden_act, layer_norm_eps, vocab_size):
        super().__init__()
        self.transform = BertPredictionHeadTransform(hidden_size, hidden_act, layer_norm_eps)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(vocab_size), requires_grad=True)

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`

    def forward(self, hidden_states, embeddings):
        hidden_states = self.transform(hidden_states)
        # hidden_states = torch.matmul(hidden_states, embeddings.t())
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertForMaskedLM(nn.Module):
    def __init__(self, hidden_size, hidden_act, layer_norm_eps, vocab_size):
        super(BertForMaskedLM, self).__init__()
        self.cls = BertLMPredictionHead(hidden_size, hidden_act, layer_norm_eps, vocab_size)

    def forward(self,outputs, embeddings):
        sequence_output = outputs
        prediction_scores = self.cls(sequence_output, embeddings)
        return prediction_scores