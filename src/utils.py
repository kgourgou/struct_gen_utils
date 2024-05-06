import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from gtda.homology import VietorisRipsPersistence


def output_logits(
    string,
    pipe,
):
    """
    Returns the logits of the next token.

    Args:
        string (str): the input string
        pipe (transformers.pipeline): A pipeline object

    Returns:
        logits (torch.Tensor): The logits of the next token,
            sorted by probability.
    """
    device = pipe.device
    tok = pipe.tokenizer
    tok.pad_token = tok.eos_token
    with torch.no_grad():
        s = tok(
            string, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        s = {k: v.to(device) for k, v in s.items()}
        logits = pipe.model(**s)["logits"][0, -1, :]
        probs = torch.nn.functional.softmax(logits, dim=-1)

    vocab = {tok.decode([i]): prob.cpu().numpy() for i, prob in enumerate(probs)}

    return sorted(vocab.items(), key=lambda x: x[1], reverse=True)


def generate_data(template: str, dataset, output_embeddings):
    """
    Generate the prompts, output tensor, and 0th homology of the output tensor
    for a given dataset and template.

    Args:
        template (str): The template string to be filled in with the x value
        dataset (list): The dataset to be used

    Returns:
        prompts (list): The list of prompts
        out_tensor (torch.Tensor): The output tensor of the LLM
        h0 (torch.Tensor): The 0th homology of the output tensor

    """
    prompts = []
    for elem in tqdm(dataset, desc="prepping prompts"):
        string = template.format(x=elem["text"])
        prompts.append(string)
    prompts = DataLoader(prompts, batch_size=8)

    out = []
    for elem in tqdm(prompts, desc="running through LLM"):
        output = output_embeddings(elem)
        out.append(output)

    out_tensor = torch.concat(out)
    VR = VietorisRipsPersistence(homology_dimensions=[0])

    pers_times = VR.fit_transform(out_tensor[None, :, :].cpu())
    h0 = pers_times[0, :, 1]

    return prompts, out_tensor, h0
