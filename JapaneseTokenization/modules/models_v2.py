#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Projet dans le cadre du cours de "Méthodes en apprentissage automatique" du master PluriTAL
# Auteurs : BOTTERO Noélie & PHOMMADY Elodie

from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.models import Model
from allennlp.modules import Embedding, FeedForward, TimeDistributed
from allennlp.modules.seq2seq_encoders.pytorch_seq2seq_wrapper import RnnSeq2SeqEncoder, GruSeq2SeqEncoder, LstmSeq2SeqEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import Activation
from allennlp.nn.util import get_text_field_mask
from allennlp.nn.util import sequence_cross_entropy_with_logits
from allennlp.training.metrics import CategoricalAccuracy


class BaselineModel(Model):

    def __init__(self, vocab: Vocabulary,  dim: int = 10, **kwargs):
        """ Constructeur du modèle : Création des différents éléments du modèle
        :param vocab: Vocabulary, vocabulaire issu des instances créées par le reader
        :param dim: int, dimension choisie pour les embeddings et les encodings
        :param kwargs: Arbitrary Kword Arguments, à surcharger
        """
        super().__init__(vocab, **kwargs)
        self.num_classes = vocab.get_vocab_size("labels")
        # Embeddings avec ses paramètres : Transformer la séquence de tokens en vecteurs
        self.text_embedder = BasicTextFieldEmbedder(
            {'tokens': Embedding(embedding_dim=dim,
                                 vocab=vocab,
                                 pretrained_file=None)}
        )
        # Encoding seq2sec avec ses paramètres : Aplatir les vecteurs
        # -- RNN
        self.seq2seq = RnnSeq2SeqEncoder(self.text_embedder.get_output_dim(), self.text_embedder.get_output_dim())
        # -- GRU
        #self.seq2seq = GruSeq2SeqEncoder(self.text_embedder.get_output_dim(), self.text_embedder.get_output_dim())
        # -- LSTM
        #self.seq2seq = LstmSeq2SeqEncoder(self.text_embedder.get_output_dim(), self.text_embedder.get_output_dim())
        # FeedForward : Effectuer la décision finale (choisir les groupes/classes)
        self.ff = TimeDistributed(
            FeedForward(input_dim=self.seq2seq.get_output_dim(),
                        num_layers=1,
                        hidden_dims=self.num_classes,
                        activations=Activation.by_name("relu")()
                        )
        )
        # Evaluation de l'exactitude pendant l'entraînement : Categorical Top-K accuracy (K = 1)
        self.accuracy = CategoricalAccuracy()

    def forward(self, text: TextField, sequence_label: SequenceLabelField) -> Dict[str, torch.Tensor]:
        """ forward : Établit le parcours des données au sein du modèle
        :param text: TextField, correspondant à une phrase
        :param sequence_label: SequenceLabelField, correspondant à la séquence de labels de chaque caractère de la phrase
        :return: dictionnaire avec la valeur d'erreur
        """
        # Masque : Permet de ne pas entraîner le modèle sur des cases vides (dues aux batchs)
        mask = get_text_field_mask(text)    # mask a une dimension égale à celle de mon text
        # Passer le text (séquence de tokens) dans l'embedding pour avoir une séquence de vecteurs
        embeddings = self.text_embedder(text)
        # Aplatir cette séquence de vecteur avec l'encoding (en utilisant le masque)
        encoded = self.seq2seq(embeddings, mask)
        # Réaliser le feedforward (passage d'un vecteur de la dimension de sortie de l'encoding à un vecteur de taille du nombre de classes)
        logits = self.ff(encoded)
        # Calcul du score d'erreur
        loss = sequence_cross_entropy_with_logits(logits, sequence_label, mask)
        # Mettre à jour les scores d'exactitude pendant l'entraînement
        self.accuracy(logits, sequence_label, mask)
        return {'loss': loss}

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        """ get_metrics : Permet d'avoir les scores d'exactitude au fur et à mesure de l'entrainement
        :param reset: boolean, remettre à jour l'état interne ou les accumulations
        :return: dictionnaire à un élément dont la valeur est le score d'exactitude
        """
        # Le modèle peut s'autoévaluer au cours de l'entraînement
        result = {'accuracy': self.accuracy.get_metric(reset)}
        return result


if __name__ == '__main__':
    from readers import JPCorpusReader

    # Lecture du corpus
    reader = JPCorpusReader()
    instances = list(reader.read("../data/Corpus_Analyzer/ja_gsd-ud-dev.txt"))
    # Passage des instances dans le modèle
    vocab = Vocabulary.from_instances(instances)
    model = BaselineModel(vocab)
    model.forward_on_instances(instances)
