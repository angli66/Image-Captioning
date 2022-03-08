from file_utils import read_file_in_dir
from dataset_factory import get_datasets
from model_factory import get_model

config_data = read_file_in_dir('./', 'experiment_1' + '.json')
coco_test, vocab, train_loader, val_loader, test_loader = get_datasets(config_data)

# Prints out one caption from the train set
# _, target, _ = next(iter(train_loader))
# target = target[0].reshape(-1)
# for word_id in target:
#     print(vocab.idx2word[word_id.item()])

images, captions, _ = next(iter(train_loader))
model = get_model(config_data, vocab)
outputs = model(images, captions)

print(len(vocab))
print(images.shape)
print(captions.shape)
print(outputs.shape)