from bitarray import test
from file_utils import read_file_in_dir
from dataset_factory import get_datasets
from model_factory import get_model
import nltk

# Specify the experiment name
experiment_name = 'experiment_1'

config_data = read_file_in_dir("./", f"{experiment_name}.json")
coco_test, vocab, _, _, test_loader = get_datasets(config_data)
model = get_model(config_data, vocab)

images, captions, image_ids = next(iter(test_loader))
predictions = model.sample(images, 20, 0.1)

# for prediction in predictions:
#     for word_idx in prediction:
#         print(vocab.idx2word[word_idx.item()])
#     break

reference_captions = []
annotations = coco_test.imgToAnns[image_ids[0]]
for annotation in annotations:
    reference_caption = annotation['caption']
    reference_caption = nltk.tokenize.word_tokenize(reference_caption)
    reference_captions.append(reference_caption)

print(reference_captions)