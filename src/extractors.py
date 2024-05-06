"""
Hidden state extractors for different models and for the 
last token in the sequence.
"""
import torch
from transformers import pipeline


def extract_hs_gpt2(s, pipe: pipeline):
    """

    Args:
        s (dict): A dictionary of elements from the tokenizer
        pipe (transformers.pipeline): A pipeline object

    Returns:
        torch.Tensor: The hidden state of the last token

    """
    tok = pipe.tokenizer
    tok.pad_token = tok.eos_token
    with torch.no_grad():
        s = tok(s, return_tensors="pt", padding="max_length", max_length=512).to(
            pipe.device
        )
        return pipe.model.transformer(**s)["last_hidden_state"][-1][-1, :]


def extract_hs_mistral_instruct(s, pipe: pipeline):
    """

    Args:
        s (dict): A dictionary of elements from the tokenizer
        pipe (transformers.pipeline): A pipeline object

    Returns:
        torch.Tensor: The hidden state of the last token

    """
    tok = pipe.tokenizer
    tok.pad_token = tok.eos_token
    with torch.no_grad():
        s = tok(s, return_tensors="pt", padding="max_length", max_length=512).to(
            pipe.device
        )
        return pipe.model.model(**s)["last_hidden_state"][:, -1]
