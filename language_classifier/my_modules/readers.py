"""BOTTERO Noélie"""
from typing import Iterable
from allennlp.data import Instance, Token
from allennlp.data.fields import LabelField, TextField
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import SingleIdTokenIndexer

class NGSequenceReader(DatasetReader):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.indexer = SingleIdTokenIndexer()

    # Transformation d'un corpus en instances
    def _read(self, file_path) -> Iterable[Instance]:
        with open(file_path) as lines:
            for line in lines:
                yield self.text_to_instance(line)

    # La fonction text_to_instance sépare les phrases en token de 1 caractères
    def text_to_instance(self, line: str) -> Instance:
        lang, txt = line.split(" ", 1)
        """ Caractères """
        # tokens = [Token(c) for c in line]

        """N-grams"""
        unigrams = []
        bigrams = []
        trigrams = []
        word = txt.split(" ")
        # Création de listes de n-grams
        prevtok = ''
        prevprevtok = ''
        for token in word:

            unigrams.append(f'{token}')
            bigrams.append(f'{prevtok} {token}')
            trigrams.append(f'{prevprevtok} {prevtok} {token}')
            prevprevtok = prevtok
            prevtok = token

        """Unigrammes"""
        tokens = [Token(c) for c in unigrams]
        """Bigrammes"""
        # tokens = [Token(c) for c in bigrams]
        """Trigrammes"""
        # tokens = [Token(c) for c in trigrams]

        # Cette fonction renvoie une liste d'instances quand reader.read() sera appelée
        return Instance({
            # Label = langue
            'labels': LabelField(lang),
            # text = séquence à analyser
            'text': TextField(tokens, {'tokens': self.indexer})
        })


if __name__ == "__main__":
    reader = NGSequenceReader(max_instances=100)
    insts = list(reader.read("../Data/corpus.txt"))
    print(insts[0])
