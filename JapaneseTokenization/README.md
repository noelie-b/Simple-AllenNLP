# Projet dans le cadre du cours de "Méthodes en apprentissage automatique" du master PluriTAL

> BOTTERO Noélie & PHOMMADY Elodie



## Objectif du projet

À travers de projet, nous souhaitons réaliser un analyseur morphologique du japonais qui réalise une segmentation en mots et un étiquetage en parties du discours (POS). 



## Données

### Corpus UD Japanese-GSD

Pour ce projet, nous utilisons un corpus en libre accès mis à disposition par Universal Dependencies : le corpus *Japanese-GSD*. Il contient environ 16 000 phrases tirées de Google (Google Universal Dependency Treebanks v2.0).  

Ce corpus est très bien organisé. Il est déjà séparé en sous-corpus d'entrainement, de validation et de test. Les données sont au format CoNLL-U.

*Exemple d'un fichier du corpus*

```conllu
# newdoc id = train-s1
# sent_id = train-s1
# text = ホッケーにはデンジャラスプレーの反則があるので、膝より上にボールを浮かすことは基本的に反則になるが、その例外の一つがこのスクープである。
1	ホッケー	ホッケー	NOUN	名詞-普通名詞-一般	_	9	obl	_	BunsetuBILabel=B|BunsetuPositionType=SEM_HEAD|LUWBILabel=B|LUWPOS=名詞-普通名詞-一般|SpaceAfter=No
2	に	に	ADP	助詞-格助詞	_	1	case	_	BunsetuBILabel=I|BunsetuPositionType=SYN_HEAD|LUWBILabel=B|LUWPOS=助詞-格助詞|SpaceAfter=No
3	は	は	ADP	助詞-係助詞	_	1	case	_	BunsetuBILabel=I|BunsetuPositionType=FUNC|LUWBILabel=B|LUWPOS=助詞-係助詞|SpaceAfter=No
4	デンジャラス	デンジャラス	NOUN	名詞-普通名詞-一般	_	5	compound	_	BunsetuBILabel=B|BunsetuPositionType=CONT|LUWBILabel=B|LUWPOS=名詞-普通名詞-一般|SpaceAfter=No
...
```

Il faut savoir qu'en japonais, on recense trois couches de tokénisation :

- Short Unit Word (SUW), 
- Long Unit Word (LUW), 
- et base-phrase (bunsetsu).

Dans notre projet, nous choisissons de réaliser une segmentation sur la base du Short Unit Word, tel qu'il est proposé dans le corpus UD. Les Short Unit Words sont représenté dans la seconde colonne du tabulaire.   



### Pré-traitement

Le corpus *Japanese-GSD* est très bien constitué et très complet. Néanmoins, pour notre analyseur morphologique, il y a énormément d'informations dont nous faisons abstraction. De plus, nous avons choisi de traiter les phrases caractère par caractère, auquel nous attribuons une étiquette sous le schéma BIO (plus précisément IOB2), accompagnée de la partie du discours du token dans lequel ce caractère appartient. 

Nous avons donc choisi de passer par une phase de pré-traitement pour ne récupérer que les informations qui nous intéressent. Nous voulons tout de même garder les lignes de métadonnées et celle avec les phrases entières. 

Nous faisons le choix de représenter la segmentation des tokens dans la phrase avec une annotation BIO, en précisant que nous utiliserons seulement les étiquettes B (*beginning*) et I (*inside*). Nous faisons le choix de ne pas utiliser le schéma d'annotation BIOES, et par conséquent l'étiquette S (*single element*). Les tokens à un seul caractère seront donc étiqueté d'un B. 

Pour les parties du discours, nous utilisons les annotations XPOS du corpus *Japanese-GSD*, qui correspondent aux parties du discours des Short Unit Words, sur la base du *UniDic POS tagset*. 

Suite aux pré-traitements, nos données sont stockées dans le répertoire *Corpus_Analyzer* et sont sous le format suivant : 

```text
# newdoc id = train-s1
# sent_id = train-s1
# text = ホッケーにはデンジャラスプレーの反則があるので、膝より上にボールを浮かすことは基本的に反則になるが、その例外の一つがこのスクープである。
1	ホ	B-NOUN
2	ッ	I-NOUN
3	ケ	I-NOUN
4	ー	I-NOUN
5	に	B-ADP
6	は	B-ADP
...
```



## DatasetReader

Dans notre DatasetReader, la création d'une instance se fait à l'échelle d'une phrase. Une instance correspond à un TextField contenant les caractères de la phrase en japonais et à un SequenceLabelField contenant les étiquettes BIO (avec les POS) de chaque caractère de la phrase.

Pour le TextField, vu que nous avons besoin des tokens qui sont dans notre cas des unigrammes, nous pouvons : 

- soit on récupère la phrase brut et on réalise un travail de tokénisation en unigrammes dessus, 
- soit nous pouvons directement utiliser les caractères dans la partie tabulaire, qui sont déjà par unigrammes. 

Nous avons choisi la deuxième option car nous pouvons réaliser en même temps la récupération des étiquettes pour le SequenceLabelField.



## Modèle

### Baseline

##### 1er essai (models.py et classif.jsonnet) :

- layers dans le feedforward = 1
- epochs = 4
- batch_size = 128

##### Résultats :

|                          | RNNSeq2SeqEncoder | GruSeq2SeqEncoder | LstmSeq2SeqEncoder |
| :----------------------- | :---------------: | :---------------: | :----------------: |
| best_epoch               |         0         |         0         |         4          |
| epoch                    |         4         |         4         |         4          |
| training_accuracy        |       0.129       |       0.060       |       0.076        |
| training_loss            |       3.316       |       3.305       |       3.302        |
| validation_accuracy      |       0.132       |       0.064       |       0.106        |
| validation_loss          |       3.305       |       3.294       |       3.296        |
| best_validation_accuracy |       0.132       |      0.0649       |       0.106        |
| best_validation_loss     |       3.356       |       3.339       |       3.296        |



##### 2eme essai (models_v2.py et classif_v2.jsonnet) :

- layers dans le feedforward = 1
- epochs = 50
- batch_size = 128

##### Résultats (UNIDIRECTIONNEL) :

|                          | RNNSeq2SeqEncoder | GruSeq2SeqEncoder | LstmSeq2SeqEncoder |
| :----------------------- | :---------------: | :---------------: | :----------------: |
| best_epoch               |        49         |        49         |         49         |
| epoch                    |        49         |        49         |         49         |
| training_accuracy        |       0.382       |       0.210       |       0.393        |
| training_loss            |       2.591       |       2.697       |       2.756        |
| validation_accuracy      |       0.373       |       0.219       |       0.375        |
| validation_loss          |       2.590       |       2.704       |       2.745        |
| best_validation_accuracy |       0.373       |       0.219       |       0.375        |
| best_validation_loss     |       2.590       |       2.704       |       2.745        |

<img src="/img/RNN_baseline.png" alt="RNN_baseline.png" style="zoom:50%;" />

<img src="/img/GRU_baseline.png" alt="GRU_baseline.png" style="zoom:50%;" />

<img src="/img/LSTM_baseline.png" alt="LSTM_baseline.png" style="zoom:50%;" />







## Post-traitement

Suite aux résultats donnés par le modèle, il faut pouvoir reconstruire une phrase tokénisé dont les tokens sont annotés par leur partie du discours.