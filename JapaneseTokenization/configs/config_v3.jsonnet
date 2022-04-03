local batch_size = std.parseJson(std.extVar('batch'));
local dim = std.parseInt(std.extVar('dim'));
local lr = std.parseJson(std.extVar('lr'));
{
    "dataset_reader" : {
        "type": "modules.readers.JPCorpusReader",
        "max_instances": 500
    },
    "train_data_path": "./data/Corpus_Analyzer/ja_gsd-ud-train.txt",
    "validation_data_path": "./data/Corpus_Analyzer/ja_gsd-ud-dev.txt",
    "model": {
        "type": "modules.models_v2.BaselineModel",
        // "dim": dim
    },
    "data_loader": {
        "batch_size": 128,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
          // lr: 0.001,
          type: 'adam',
        },
        "num_epochs": 50,
        "patience": 5,
        // "validation_metric":'+accuracy',
    }
}