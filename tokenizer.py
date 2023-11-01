import spacy
from torchtext.vocab import build_vocab_from_iterator


class Tokenizer:

    def __init__(self, dataset):
        self.en, self.de = zip(*dataset)
        self.spacy_en = spacy.load("en_core_web_sm")
        self.spacy_de = spacy.load("de_core_news_sm")
        self.vocab_en = None
        self.vocab_de = None

    def yield_tokens_en(self):
        for x in self.en:
            yield [token.text for token in self.spacy_en(x)]

    def yield_tokens_de(self):
        for x in self.de:
            yield [token.text for token in self.spacy_de(x)]

    def __call__(self):
        self.vocab_en = build_vocab_from_iterator(self.yield_tokens_en(),
                                                  specials=["<BOS>", "<EOS>", "<PAD>", "<UNK>"])
        self.vocab_de = build_vocab_from_iterator(self.yield_tokens_de(),
                                                  specials=["<BOS>", "<EOS>", "<PAD>", "<UNK>"])
        self.vocab_en.set_default_index(self.vocab_en["<UNK>"])
        self.vocab_de.set_default_index(self.vocab_de["<UNK>"])
