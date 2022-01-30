hyper = {
    "max_len": 100,
    "batch_size": 128,
    "epoch": 150,
    "word_dim": 512,
    "lstm_hidden": 256,
    "learning_rate": 1e-3,
    "gpu_id": 0,
    "bert": True,
    "bert_path": 'pretrained_models/bert-base-chinese',
    "train_path": 'data/train.json',
    "valid_path": 'data/dev.json',
    "output_train": 'output/train_ans.json',
    "output_valid": 'output/valid_ans.json'
}
