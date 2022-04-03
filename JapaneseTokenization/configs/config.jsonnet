//local batch = 128;
//local dim = 10;
{
    "dataset_reader" : {
        "type": "modules.readers.JPCorpusReader",
        "max_instances": 500
    },
    "train_data_path": "./data/Corpus_Analyzer/ja_gsd-ud-train.txt",
    "validation_data_path": "./data/Corpus_Analyzer/ja_gsd-ud-dev.txt",
    "model": {
        "type": "modules.models.SegModel",
        //"dim": dim
    },
    "data_loader": {
        "batch_size": 128,
        "shuffle": true
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 5,
        "validation_metric":'+accuracy'
}
}