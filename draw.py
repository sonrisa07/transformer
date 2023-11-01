from matplotlib import pyplot as plt
from configuration import draw_path

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
    plt.grid(True, which='both', axis='both')
    plt.savefig("./loss.png")

    plt.figure()
    plt.plot(bleu, 'b', label="bleu")
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.grid(True, which='both', axis='both')
    plt.savefig("./bleu.png")

draw()
