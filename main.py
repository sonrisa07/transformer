import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from rich.progress import track
from torchtext.data.metrics import bleu_score

from data_process import *
from configuration import *
from transformer import Transformer

model = Transformer(d_model, d_k, d_v, d_ffn, n_heads, n_encoder_layers, n_decoder_layers,
                    enc_vocab_size, dec_vocab_size, enc_seq_max_len, dec_seq_max_len, dropout, device).to(device)

# The following is mentioned in the paper
"""
criterion = nn.CrossEntropyLoss(ignore_index=dec_pad_idx, label_smoothing=label_smoothing)
optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.98), eps=eps, weight_decay=weight_decay)
lambda_lr = lambda epoch: d_model ** -0.5 * min((epoch + 1) ** -0.5, (epoch + 1) * (warmup_steps ** -1.5))
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)
"""

# The following is particular to this dataset
criterion = nn.CrossEntropyLoss(ignore_index=dec_pad_idx, label_smoothing=label_smoothing)
optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, verbose=True)


def train():
    model.train()
    avg_loss = 0.0
    for x, y in track(train_loader, description="training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x, y[:, :-1])  # batch * seq_len * dec_vocab_size
        p_preds = preds.contiguous().view(-1, preds.shape[-1])  # CrossEntropyLoss needs N * C (N * dec_vocab_size)
        loss = criterion(p_preds, y[:, 1:].contiguous().view(-1))  # dim: N
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        avg_loss += loss.detach().item()

    return avg_loss / len(train_loader)


def validate(dataloader):  # close teach forcing
    model.eval()  # simultaneously, close label smoothing
    avg_loss = 0.0
    avg_bleu = 0.0
    with (torch.no_grad()):
        for x, y in track(dataloader, description="validating"):
            x, y = x.to(device), y.to(device)
            begin_tokens = torch.tensor([[dec_begin_idx]
                                         for _ in range(y.shape[0])]).to(device)  # produce begin_idx for each sequence
            output_tokens = begin_tokens.to(device)
            for i in range(1, y.shape[-1]):  # auto regressive using model's output
                preds = model(x, output_tokens)  # batch * seq_len * dec_vocab_size
                _, pos = torch.max(preds, dim=-1)  # pos dim: batch * seq_len
                output_tokens = torch.concat((begin_tokens, pos), dim=-1).to(device)

                if i == y.shape[-1] - 1:
                    p_preds = preds.contiguous().view(-1, preds.shape[-1])
                    loss = criterion(p_preds, y[:, 1:].contiguous().view(-1))
                    avg_loss += loss.item()

            pred_sequences = []
            real_sequences = []
            for i in range(y.shape[0]):
                pred_sequences.append(revert(output_tokens[i].tolist()))
                real_sequences.append([revert(y[i].tolist())])  # the type must be like: [[]] !!!
            avg_bleu += bleu_score(pred_sequences, real_sequences)  # each item of references_corpus is like [[]]
    return avg_loss / len(dataloader), avg_bleu / len(dataloader)


def validate_forcing(dataloader):  # use teach forcing
    model.eval()  # simultaneously, close label smoothing
    avg_loss = 0.0
    avg_bleu = 0.0
    with (torch.no_grad()):
        for x, y in track(dataloader, description="validating"):
            x, y = x.to(device), y.to(device)
            preds = model(x, y[:, :-1])
            p_preds = preds.contiguous().view(-1, preds.shape[-1])
            loss = criterion(p_preds, y[:, 1:].contiguous().view(-1))
            _, pos = torch.max(preds, dim=-1)
            avg_loss += loss.item()
            pred_sequences = []
            real_sequences = []
            for i in range(y.shape[0]):
                pred_sequences.append(revert([dec_begin_idx] + pos[i].tolist()))
                real_sequences.append([revert(y[i].tolist())])  # the type must be like: [[]] !!!
            avg_bleu += bleu_score(pred_sequences, real_sequences)  # each item of references_corpus is like [[]]
    return avg_loss / len(dataloader), avg_bleu / len(dataloader)


def draw():
    loss_train, loss_valid, bleu = [], [], []
    first_line = True
    with open(draw_path, 'r') as file:
        for lin in file:
            if first_line:
                first_line = False
                continue
            e = lin.strip().split('\t')
            loss_train.append(float(e[0]))
            loss_valid.append(float(e[1]))
            bleu.append(float(e[2]))
    plt.figure()
    plt.plot(loss_train, 'r', label="train")
    plt.plot(loss_valid, 'c', label="loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid(True, which="both", axis="both")
    plt.savefig("./loss.png")

    plt.figure()
    plt.plot(bleu, 'b', label="bleu")
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.grid(True, which="both", axis="both")
    plt.savefig("./bleu.png")


def infer():
    model.load_state_dict(torch.load(save_path))
    loss, bleu = validate(test_loader_1)
    print("loss1:{}, bleu1:{}".format(loss, bleu))
    loss, bleu = validate(test_loader_2)
    print("loss2:{}, bleu2:{}".format(loss, bleu))
    loss, bleu = validate(test_loader_3)
    print("loss3:{}, bleu3:{}".format(loss, bleu))


if __name__ == '__main__':
    best_loss = 1e9
    with open(draw_path, 'w') as f:
        f.write("train_loss\tvalid_loss\tvalid_bleu\n")
        for step in range(1, train_steps + 1):
            print("epoch: {}".format(step))
            train_loss = train()
            print("train_loss: {}".format(train_loss))
            valid_loss, valid_bleu = validate_forcing(valid_loader)
            if step > warmup_steps:
                scheduler.step(valid_loss)
            print("valid_loss: {}, valid_bleu: {}".format(valid_loss, valid_bleu))
            f.write(str(train_loss) + '\t' + str(valid_loss) + '\t' + str(valid_bleu) + '\n')
            f.flush()
            if valid_loss < best_loss:
                best_loss = valid_loss
                torch.save(model.state_dict(), save_path)
    draw()
    infer()
