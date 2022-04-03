# Premier test
# allennlp tune \
#    configs/config_v3.jsonnet \
#    configs/hparams.json \
#    --serialization-dir result/optunaRNN \
#    --study-name baselineRNN\
#    --timeout 100 \
#    --metrics best_validation_accuracy \
#    --direction maximize

# RNN avec timeout de 1000
# allennlp tune \
#    configs/config_v3.jsonnet \
#    configs/hparams.json \
#    --serialization-dir result/optunaRNN1000 \
#    --study-name baselineRNN1000\
#    --timeout 1000 \
#    --metrics best_validation_accuracy \
#    --direction maximize

# GRU avec timeout de 1000
# allennlp tune \
#    configs/config_v3.jsonnet \
#    configs/hparams.json \
#    --serialization-dir result/optunaGRU1000 \
#    --study-name baselineGRU1000\
#    --timeout 1000 \
#    --metrics best_validation_accuracy \
#    --direction maximize

# LSTM avec timeout de 1000
allennlp tune \
    configs/config_v3.jsonnet \
    configs/hparams.json \
    --serialization-dir result/optunaLSTM1000 \
    --study-name baselineLSTM1000\
    --timeout 1000 \
    --metrics best_validation_accuracy \
    --direction maximize