from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from configuration import *
from tokenizer import *


class MultiDataset(Dataset):

    def __init__(self, en_path, de_path):
        with open(en_path, 'r') as f:
            self.en = [line.strip() for line in f.readlines()]
        with open(de_path, 'r') as f:
            self.de = [line.strip() for line in f.readlines()]

    def __getitem__(self, idx):
        return self.en[idx], self.de[idx]

    def __len__(self):
        return len(self.en)


train_set, valid_set = (MultiDataset(train_en_path, train_de_path),
                        MultiDataset(valid_en_path, valid_de_path))

test_set_1, test_set_2, test_set_3 = (MultiDataset(test_en_path_1, test_de_path_1),
                                      MultiDataset(test_en_path_2, test_de_path_2),
                                      MultiDataset(test_en_path_3, test_de_path_3))

tokens = Tokenizer(train_set)
tokens()


def convert(x, isEn):
    if isEn:
        return ([tokens.vocab_en['<BOS>']]
                + [tokens.vocab_en[v.text] for v in tokens.spacy_en(x)]
                + [tokens.vocab_en['<EOS>']])
    else:
        return ([tokens.vocab_de['<BOS>']]
                + [tokens.vocab_de[v.text] for v in tokens.spacy_de(x)]
                + [tokens.vocab_de['<EOS>']])


def revert(seq_token):
    assert seq_token[0] == dec_begin_idx
    res = []
    for x in seq_token[1:]:
        if x == dec_end_idx:
            break
        if x == dec_pad_idx:
            continue
        res.append(tokens.vocab_de.get_itos()[x])
    return res


def collate_batch(b):
    x, y = [], []
    for enc_sequence, dec_sequence in b:
        x.append(torch.tensor(convert(enc_sequence, True)))
        y.append(torch.tensor(convert(dec_sequence, False)))
    x = pad_sequence(x, True, 1.0 * enc_pad_idx)
    y = pad_sequence(y, True, 1.0 * dec_pad_idx)
    return x, y


train_loader = DataLoader(train_set, batch, shuffle=True, num_workers=5, collate_fn=collate_batch)
valid_loader = DataLoader(valid_set, batch, shuffle=True, num_workers=5, collate_fn=collate_batch)

test_loader_1 = DataLoader(test_set_1, batch, shuffle=False, num_workers=5, collate_fn=collate_batch)
test_loader_2 = DataLoader(test_set_2, batch, shuffle=False, num_workers=5, collate_fn=collate_batch)
test_loader_3 = DataLoader(test_set_3, batch, shuffle=False, num_workers=5, collate_fn=collate_batch)

dec_begin_idx = tokens.vocab_de['<BOS>']
dec_end_idx = tokens.vocab_de['<EOS>']

enc_pad_idx = tokens.vocab_en['<PAD>']
dec_pad_idx = tokens.vocab_de['<PAD>']

enc_vocab_size = len(tokens.vocab_en)
dec_vocab_size = len(tokens.vocab_de)
