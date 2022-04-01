"""BOTTERO Noélie"""

import tempfile
import torch
from readers import NGSequenceReader
from torch.nn.functional import binary_cross_entropy
from typing import Dict, List, Tuple
from allennlp.data import DataLoader, Instance, Vocabulary, TextFieldTensors
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util
from allennlp.training.trainer import Trainer
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.util import evaluate
from allennlp.models import Model


# Création d'un modèle de classification de textes selon la langue
# Le modèle prend en entrée un paquet d'Instances, prédit le résultat et génère un score de perte


class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()

    def forward(self,
                text: TextFieldTensors,
                labels: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        embedded_text = self.embedder(text)
        # Fusion de tous les vecteurs en 1 seul
        mask = util.get_text_field_mask(text)
        # forme : (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)

        """Prédiction des résultats"""
        # logits permet de prédire un label en générant une probabilité
        # forme : (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        probs = torch.nn.functional.softmax(logits)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        output = {'probs': probs}
        best = [torch.argmax(p).item() for p in probs]
        if labels is not None:
            self.accuracy(logits, labels)
            return {
                'loss': loss,
                'probs': probs,
                'best': best
            }


def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    # Conversion de chaque identifiant de token en vecteur
    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=10, num_embeddings=vocab_size)}
    )
    encoder = BagOfEmbeddingsEncoder(embedding_dim=10)
    return SimpleClassifier(vocab, embedder, encoder)


def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    return {"accuracy": self.accuracy.get_metric(reset)}


def run_training_loop():
    dataset_reader = NGSequenceReader(max_instances=3000)
    print("Reading data")
    instances = list(dataset_reader.read("../Data/corpus.txt"))
    train_data = instances[0:1600]
    dev_data = instances[1600:2000]
    test_data = instances[2000:]
    vocab = Vocabulary.from_instances(train_data + dev_data)
    model = build_model(vocab)
    # Chargement des données de train et de développement,
    # divisées en batchs de 8 instances
    train_loader = SimpleDataLoader(train_data, 8, shuffle=True)
    dev_loader = SimpleDataLoader(dev_data, 8, shuffle=False)
    test_loader = SimpleDataLoader(test_data, 8, shuffle=False)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)
    test_loader.index_with(vocab)

    # tempfile permet de gérer les fichiers temporaires
    # Il sera effacé à la fin du run
    with tempfile.TemporaryDirectory() as serialization_dir:
        # Initialisation de l'entraînement
        trainer = build_trainer(model, serialization_dir, train_loader, dev_loader)
        print("Starting training")
        # Lancement de l'entraînement du modèle
        trainer.train()
        print("Finished training")
        # Affichage des scores
        print("Score sur les données de test :", evaluate(model, test_loader))
    return model, dataset_reader


def build_data_loaders(
    train_data: List[Instance],
    dev_data: List[Instance],
) -> Tuple[DataLoader, DataLoader]:
    train_loader = SimpleDataLoader(train_data, 8, shuffle=True)
    dev_loader = SimpleDataLoader(dev_data, 8, shuffle=False)
    return train_loader, dev_loader


def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader,
) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters)
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=40,
        optimizer=optimizer,
    )
    return trainer




run_training_loop()
