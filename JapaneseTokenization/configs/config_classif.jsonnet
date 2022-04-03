local batch = 128;
local dim = 10;
{
    "dataset_reader" : {
        "type": "modules.readers.JPCorpusReader",
        "max_instances": 500
    },
    "train_data_path": "./data/Corpus_Analyzer/ja_gsd-ud-train.txt",
    "validation_data_path": "./data/Corpus_Analyzer/ja_gsd-ud-dev.txt",
    "model": {
        "type": "modules.models.BaselineModel",
        "dim": dim
    },
    "data_loader": {
        "batch_size": batch,
        "shuffle": true
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 5
    }
}