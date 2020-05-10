import torch
import transformers


def surprisal(model, sent_id, surprisal_idx):
    output = model(sent_id, labels=sent_id)
    log_probs = torch.log_softmax(output[1][0], dim=1)
    surprisals = []
    for i, t in enumerate(sent_id[0]):
        surprisals.append(-1 * log_probs[i][t])  # log_probs[i] is the probability distribution conditional on all words up to and INCLUDING word_i
    surprisals = torch.tensor(surprisals[surprisal_idx:])
    # print('Individual surprisals, sum:', surprisals, torch.sum(surprisals))
    return surprisals, torch.sum(surprisals)


def get_model_fn(name='gpt2'):
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelWithLMHead.from_pretrained(name)
    print('Loaded model and tokenizer for GPT-2')

    # Inputs is a single item: sentence, surprisal_index
    def model_fn(sent, idx):
        surprisal_idx = len(tokenizer.encode(sent[:idx], add_special_tokens=True,))  # Where to begin counting!
        sent_id = tokenizer.encode(sent, add_special_tokens=True, return_tensors='pt')
        _, total = surprisal(model, sent_id, surprisal_idx)  # Returns tensors
        return total.item()

    return model_fn