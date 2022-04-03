#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# Projet dans le cadre du cours de "Méthodes en apprentissage automatique" du master PluriTAL
# Auteur : BOTTERO Noélie


import glob
import os


def main(input_path: str, output_path: str):
    """ Pré-traitement sur le corpus original UD GSD japonais : Adaptation du corpus d'entrée
        pour qu'il corresponde au DatasetReader de notre analyseur morphologique
    :param input_path: str, chemin du corpus original UD GSD japonais
    :param output_path: str, chemin du répertoire dans lequel nous voulons que les fichiers traités soient
    """
    # Créer le répertoire de sortie s'il n'existe pas
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Parcourir le corpus original UD GSD japonais
    for doc in glob.glob(input_path + "*.conllu"):
        # Ouvrir le fichier de sortie correspondant
        output_file = open(output_path + doc.split(os.path.sep)[-1].split(".")[0] + ".txt", "w+", encoding="utf8")

        # Ouvrir le fichier d'entrée
        with open(doc, "r", encoding="utf8") as file:
            # Initialiser un compteur de caractère
            cpt = 0
            # Parcourir le fichier
            for line in file:
                if not line[0].isdigit():   # Copier les lignes qui ne sont pas au format tabulaire telles quelles
                    output_file.write(line)     # Pas d'ajout de "\n" car les lignes se finissent déjà par un saut de ligne
                    cpt = 0     # Réinitialiser le compteur
                else:   # Traitement des lignes tabulaires : Un caractère par ligne
                    # Récupérer le token et la partie du discours (universal POS tagset)
                    infos = line.split("\t")
                    token = infos[1]
                    pos = infos[3]
                    # Incrémenter le compteur de caractère
                    cpt += 1
                    assert(len(token) != 0)
                    if len(token) == 1:     # Token à un caractère
                        #output_file.write(str(cpt) + "\t" + token + "\tB-" + pos + "\n")
                        output_file.write(f"{cpt}\t{token}\tB-{pos}\n")
                    else:   # Token à plusieurs caractères
                        for idx, character in enumerate(token):
                            if idx == 0:
                                output_file.write(f"{cpt}\t{character}\tB-{pos}\n")
                            else:
                                cpt += 1
                                output_file.write(f"{cpt}\t{character}\tI-{pos}\n")

        # Fermer le fichier de sortie correspondant
        output_file.close()


if __name__ == "__main__":
    main("./UD_Japanese-GSD/", "./Corpus_Analyzer/")
