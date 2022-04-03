#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# Projet dans le cadre du cours de "Méthodes en apprentissage automatique" du master PluriTAL
# Auteur : BOTTERO Noélie


from pathlib import Path
from typing import Iterable

from allennlp.data import DatasetReader, Instance, Token
from allennlp.data.fields import TextField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer


class JPCorpusReader(DatasetReader):

    def __init__(self, **kwargs):
        """ Constructeur du reader : Définition des éléments utilisés pour la lecture des données
        :param kwargs: Arbitrary Kword Arguments, à surcharger
        """
        super().__init__(**kwargs)
        self.indexer = SingleIdTokenIndexer()

    def _read(self, file_path: str) -> Iterable[Instance]:
        """ read : Lecture des données
        :param file_path: str, chemin du fichier de données
        :return: iterable composé des instances de chaque phrase du fichier/corpus
        """
        file = Path(file_path)
        try:
            text = file.read_text()
            # Les phrases sont séparées par des lignes vides
            sentences = text.split("\n\n")
            # Création une instance par phrase
            for sentence in sentences:
                yield self.text_to_instance(sentence)
        except:     # Pour ne pas traiter les fichiers qui ne sont pas en UTF-8
            pass

    def text_to_instance(self, sentence: str) -> Instance:
        """ text_to_instance : Transformation d'un texte brut et de sa classe en instance
        :param sentence: str, informations correspondant à une phrase
        :return: instance composée d'un dictionnaire avec le texte tokenisé par caractère
                    et la séquence des labels associé à chaque caractère
        """
        tokens = []
        labels = []
        # Parcours des lignes correspondant aux informations pour une phrase
        for line in sentence.split("\n"):
            if line[0] == "#":
                continue
            try:
                number, character, label = line.split("\t")
                # Ajouter le caractère et son label dans les listes correspondantes
                tokens.append(Token(character))
                labels.append(label)
            except:
                pass    # Prévoir des cas spéciaux (ex. ligne vide)

        # Transformation des str en TextField et LabelField
        assert(len(tokens) == len(labels))
        text_field = TextField(tokens, token_indexers={'tokens': self.indexer})
        sequence_label_field = SequenceLabelField(labels, text_field)
        return Instance({
            'tokens': text_field,
            'labels': sequence_label_field
        })


if __name__ == "__main__":
    # Création du reader
    reader = JPCorpusReader()
    # Lecture du corpus
    instances = list(reader.read("../data/Corpus_Analyzer/ja_gsd-ud-dev.txt"))
    # Tests
    print(len(instances))
    print(instances[0])
