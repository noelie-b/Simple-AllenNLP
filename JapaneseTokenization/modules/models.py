#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Projet dans le cadre du cours de "Méthodes en apprentissage automatique" du master PluriTAL
# Auteur : BOTTERO Noélie

from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.models import Model
from allennlp.modules import Embedding, FeedForward, TimeDistributed
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import RnnSeq2SeqEncoder, GruSeq2SeqEncoder, LstmSeq2SeqEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import Activation
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure
from torch.nn.functional import cross_entropy, sigmoid

class SegModel(Model):
    def __init__(self, vocab: Vocabulary, **kwargs):
        super().__init__(vocab, **kwargs)
        self.num_classes = vocab.get_vocab_size("labels")

        self.text_embedder = BasicTextFieldEmbedder(
            {'tokens': Embedding(10,
                                 vocab=vocab,
                                 pretrained_file=None)}
        )

        self.seq2seq = LstmSeq2SeqEncoder(self.text_embedder.get_output_dim(),
                                         self.text_embedder.get_output_dim())

        self.ff = TimeDistributed(
            FeedForward(input_dim=self.seq2seq.get_output_dim(),
                        num_layers=1,
                        hidden_dims=self.num_classes,
                        activations=Activation.by_name("relu")()
                        )
        )
        self.accuracy = CategoricalAccuracy()
        self.fscores = FBetaMeasure()

    def forward(self, tokens, labels) -> Dict[str, torch.Tensor]:
        mask = get_text_field_mask(tokens)
        embedding = self.text_embedder(tokens)
        encoded = self.seq2seq(embedding, mask)
        logits = self.ff(encoded)
        probs = torch.nn.functional.softmax(logits)
        loss = sequence_cross_entropy_with_logits(logits, labels, mask)
        best = [torch.argmax(p).item() for p in probs]
        self.accuracy(logits, labels, mask)
        self.fscores(logits, labels, mask)
        return {'loss': loss,
                'probs': probs,
                'best': best}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        # Le modèle peut s'autoévaluer au cours de l'entraînement
        # on rajoute la f-mesure qui sera calculée sur la chaque classe
        fscores = self.fscores.get_metric(reset)
        result = {'accuracy': self.accuracy.get_metric(reset)}
        for i, score in enumerate(fscores['fscore']):
            result[f'cl{i}'] = score
            return result


if __name__ == '__main__':
    from readers import JPCorpusReader
    from pathlib import Path

    data_path = Path("../data/Corpus_Analyzer")
    reader = JPCorpusReader(max_instances=10)
    instances = list(reader.read(data_path / "ja_gsd-ud-dev.txt"))
    vocab = Vocabulary.from_instances(instances)
    model = SegModel(vocab)
    model.forward_on_instances(instances)
