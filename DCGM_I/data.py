import numpy as np
import csv
import itertools
import operator
import nltk
import sys
from datetime import datetime
import keras

class DataPreparation():
	def __init__(self, word_dim, sentence_start_token, sentence_end_token, unknown_token):
		self.word_dim=word_dim
		self.sentence_start_token=sentence_start_token
		self.sentence_end_token=sentence_end_token
		self.unknown_token = unknown_token

	def data_preprocessing(self):
		# Read the data and append SENTENCE_START and SENTENCE_END tokens
		print("Reading CSV file...")
		with open('../data/data.csv', 'rb') as f:
			reader = csv.reader(f, skipinitialspace=True)
			reader.next()
			# Split full comments into sentences
			sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
			# Append SENTENCE_START and SENTENCE_END
			sentences = ["%s %s %s" % (self.sentence_start_token, x, self.sentence_end_token) for x in sentences]
		print("Parsed %d sentences." % (len(sentences)))
			
		# Tokenize the sentences into words
		tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]

		# Count the word frequencies
		word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
		print("Found %d unique words tokens." % len(word_freq.items()))

		# Get the most common words and build index_to_word and word_to_index vectors
		vocab = word_freq.most_common(self.word_dim-1)
		index_to_word = [x[0] for x in vocab]
		index_to_word.append(self.unknown_token)
		word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

		print("Using vocabulary size %d." % self.word_dim)
		print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

		# Replace all words not in our vocabulary with the unknown token
		for i, sent in enumerate(tokenized_sentences):
			tokenized_sentences[i] = [w if w in word_to_index else self.unknown_token for w in sent]

		print("\nExample sentence: '%s'" % sentences[0])
		print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[0])

		# Create the training data
		X = self.To_categorical(data=tokenized_sentences, word_to_index=word_to_index)
		return self.tripled_transformation(X)

	def To_categorical(self, data, word_to_index):
		a = np.zeros(self.word_dim)
		X = np.zeros((len(data), self.word_dim))
		for i, sent in enumerate(data):
		  for w in sent[:-1]:
		    a += keras.utils.to_categorical(word_to_index[w], num_classes=self.word_dim)
		  X[i] = a 
		  a = np.zeros(self.word_dim)
		return X

	def tripled_transformation(self, data):
		context = data
		message = np.insert(data[1:], 0, np.zeros((1,4000)), axis=0)
		response = np.insert(data[2:], 0, np.zeros((2,4000)), axis=0)
		return context, message, response