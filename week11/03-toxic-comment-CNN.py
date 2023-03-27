import itertools
import numpy as np
import os
import pandas as pd
import re

from gensim.models import KeyedVectors

from keras.layers import concatenate, Conv1D, Dense, Dropout, Embedding, GlobalMaxPooling1D, Input
from keras.models import Model
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import precision_score, recall_score, accuracy_score

# Initialize lemmatizer, define stopword list
lemmatizer = WordNetLemmatizer()
sw = set(stopwords.words('english'))

# Set up directories and parameters
EMBEDDING_DIM = 200
MAX_SEQUENCE_LENGTH = 200
NUM_EPOCHS = 4
BATCH_SIZE = 34

# Read in data
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

# Define the CNN model
def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
	embedding_layer = Embedding(num_words,
							embedding_dim,
							weights=[embeddings],
							input_length=max_sequence_length,
							trainable=False)
	sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
	embedded_sequences = embedding_layer(sequence_input)
	convs = []
	filter_sizes = [2,3,4,5,6,7]
	for filter_size in filter_sizes:
		l_conv = Conv1D(filters=200, kernel_size=filter_size, activation='relu')(embedded_sequences)
		l_pool = GlobalMaxPooling1D()(l_conv)
		convs.append(l_pool)
	l_merge = concatenate(convs, axis=1)
	x = Dropout(0.15)(l_merge)  
	x = Dense(128, activation='relu')(x)
	x = Dropout(0.2)(x)
	preds = Dense(labels_index, activation='sigmoid')(x)
	model = Model(sequence_input, preds)
	model.compile(loss='binary_crossentropy',
				  optimizer='adam',
				  metrics=['acc'])
	model.summary()
	return model

# Load and preprocess data
def tokenize(doc):
	tokens = [t.lower() for t in word_tokenize(doc) if t not in sw]
	#tokens += [t for t in lemmatizer.lemmatize(doc)]
	return ' '.join(tokens)

# Tokenize train and test data
data_train['features'] = data_train['text'].apply(lambda x: tokenize(x))
data_test['features'] = data_test['text'].apply(lambda x: tokenize(x))

# Load the pre-trained word embeddings
# To convert Glove to W2V format run: python3 -m gensim.scripts.glove2word2vec --input glove.6B.200d.txt --output glove.6B.200d.w2vformat.txt
w2v = KeyedVectors.load_word2vec_format('glove.6B.200d.w2vformat.txt', binary=False)

# Tokenize and prepare data
data_train['pos']= data_train["attack"]
data_train['neg']= np.abs(data_train["attack"] - 1)

train_tok = [f.split(' ') for f in data_train.features]
train_sent_len = [len(s) for s in train_tok]
train_vocab = sorted(list(set(itertools.chain.from_iterable(train_tok))))

test_tok = [f.split(' ') for f in data_test.features]
test_sent_len = [len(s) for s in test_tok]
test_vocab = sorted(list(set(itertools.chain.from_iterable(test_tok))))

tokenizer = Tokenizer(num_words=len(train_vocab), lower=True, char_level=False)
tokenizer.fit_on_texts(list(data_train['features']))
training_sequences = tokenizer.texts_to_sequences(list(data_train["features"]))
train_word_index = tokenizer.word_index
# Initialize training embeddings
train_embedding_weights = np.zeros((len(train_word_index) + 1, EMBEDDING_DIM))

x_train = pad_sequences(training_sequences, maxlen=MAX_SEQUENCE_LENGTH)
label_names = ['pos', 'neg']
y_train = data_train[label_names].values

# Preprocess test set
# Set embedding values; if word not in embedding vocabulary, add random embedding
for word, index in train_word_index.items():
	train_embedding_weights[index,:] = w2v[word] if word in w2v else np.random.rand(EMBEDDING_DIM)
test_sequences = tokenizer.texts_to_sequences(list(data_test['features']))
test_cnn_data = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH)

# Initialize model
model = ConvNet(train_embedding_weights,
				MAX_SEQUENCE_LENGTH,
				len(train_word_index) + 1,
				EMBEDDING_DIM, 
				len(label_names))

# Weight the loss function by class proportion to deal with imbalance
class_weight = {0: np.sum(data_train['neg']), 1: np.sum(data_train['pos'])}
hist = model.fit(x_train, y_train, epochs=NUM_EPOCHS, validation_split=0.1, shuffle=True, batch_size=BATCH_SIZE, class_weight=class_weight)
print("\n\nFinished Training CNN. Predicting on the test set:")
predictions = model.predict(test_cnn_data, batch_size=1024, verbose=1)
labels = [1, 0]
# Assign label to class with the largest score
y_pred = [labels[np.argmax(p)] for p in predictions]
y_true = data_test["attack"]

# Calculate precision, recall, and accuracy
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

print(f"\n\nAccuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

def predict_examples(example_texts):
	data_example = pd.DataFrame({'text':example_texts})
	data_example['features'] = data_example['text'].apply(lambda x: tokenize(x))
	example_sequences = tokenizer.texts_to_sequences(list(data_example['features']))
	example_cnn_data = pad_sequences(example_sequences, maxlen=MAX_SEQUENCE_LENGTH)
	example_predictions = model.predict(example_cnn_data, batch_size=1024, verbose=0)
	labels = [1, 0]
	# Assign label to class with the largest score
	example_y_pred = [labels[np.argmax(p)] for p in example_predictions]
	print(example_y_pred)

example_texts = [
	"You might think you are great, but you're not that brilliant.",
	"You'd probably have more success in your edits if you tried growing a brain first.",
	"You're not very helpful. These edits are not as fabulous as you might think.",
	"While sounding smart and interesting to a lay audience, you're quite uninformed and are hopelessly ignorant."
]

predict_examples(example_texts)

