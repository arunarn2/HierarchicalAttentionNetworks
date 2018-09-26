from __future__ import print_function, division

import os
import re

import numpy as np
import pandas as pd
import tensorflow as tf
from bs4 import BeautifulSoup
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tqdm import tqdm

max_sentences = 15
max_words = 20000
maxlen = 250
embedding_dim = 100
validation_split = 0.2
hidden_size = 150
attention_size = 50
keepprob = 0.8
batch_size = 256
num_epochs = 10
loss_delta = 0.5
model_path = './model'
glove_dir = "./glove.6B"
reviews = []
labels = []
texts = []
embeddings_index = {}


def attention(inputs, att_size, time_major=False, return_alphas=False):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    """
    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hiddensize = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hiddensize, att_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([att_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([att_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas


def batch_generator(X, y, batchsize):
    size = X.shape[0]
    x_copy = X.copy()
    y_copy = y.copy()
    ind = np.arange(size)
    np.random.shuffle(ind)
    x_copy = x_copy[ind]
    y_copy = y_copy[ind]
    i = 0
    while True:
        if i + batchsize <= size:
            yield x_copy[i:i + batchsize], y_copy[i:i + batchsize]
            i += batchsize
        else:
            i = 0
            ind = np.arange(size)
            np.random.shuffle(ind)
            x_copy = x_copy[ind]
            y_copy = y_copy[ind]
            continue


def remove_html(str_a):
    p = re.compile(r'<.*?>')
    return p.sub('', str_a)


# replace all non-ASCII (\x00-\x7F) characters with a space
def replace_non_ascii(str_a):
    return re.sub(r'[^\x00-\x7f]', r'', str_a)


# Tokenization/string cleaning for dataset
def clean_str(string):
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


input_data = pd.read_csv('labeledTrainData.tsv', sep='\t')

for idx in range(input_data.review.shape[0]):
    text = BeautifulSoup(input_data.review[idx], features="html5lib")
    text = clean_str(text.get_text().encode('ascii', 'ignore'))
    texts.append(text)
    labels.append(input_data.sentiment[idx])

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
data = pad_sequences(sequences, maxlen=maxlen)

# labels = to_categorical(np.asarray(labels))
labels = np.array(labels)
print('Shape of reviews (data) tensor:', data.shape)
print('Shape of sentiment (label) tensor:', labels.shape)

indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(validation_split * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = labels[-nb_validation_samples:]

print('Number of positive and negative reviews in training and validation set')
print(y_train.sum(axis=0))
print(y_val.sum(axis=0))

f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors.' % len(embeddings_index))

# building Hierachical Attention network
embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# Different placeholders
with tf.name_scope('Input_layer'):
    input_x = tf.placeholder(tf.int32, [None, maxlen], name='input_x')
    output_y = tf.placeholder(tf.float32, [None], name='output_y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# Embedding layer
with tf.name_scope('Embedding_layer'):
    embeddings_var = tf.Variable(tf.random_uniform([len(word_index) + 1, embedding_dim], -1.0, 1.0), trainable=True)
    tf.summary.histogram('embeddings_var', embeddings_var)
    batch_embedded = tf.nn.embedding_lookup(embeddings_var, input_x)

# BiDirectional RNN Layer
rnn_outputs, _ = bi_rnn(GRUCell(hidden_size), GRUCell(hidden_size), inputs=batch_embedded, dtype=tf.float32)
tf.summary.histogram('RNN_outputs', rnn_outputs)

# Attention layer
with tf.name_scope('Attention_layer'):
    attention_output, alphas = attention(rnn_outputs, attention_size, return_alphas=True)
    tf.summary.histogram('alphas', alphas)

# Dropout for attention layer
drop = tf.nn.dropout(attention_output, keep_prob)

# Fully connected layer
with tf.name_scope('Fully_connected_layer'):
    W = tf.Variable(tf.truncated_normal([hidden_size * 2, 1], stddev=0.1))  # Hidden size is multiplied by 2 for Bi-RNN
    b = tf.Variable(tf.constant(0., shape=[1]))
    y_hat = tf.nn.xw_plus_b(drop, W, b)
    y_hat = tf.squeeze(y_hat)
    tf.summary.histogram('W', W)

with tf.name_scope('Metrics'):
    # Cross-entropy loss and optimizer initialization
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=output_y))
    tf.summary.scalar('loss', loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)

    # Accuracy metric
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(y_hat)), output_y), tf.float32))
    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all()

# Batch generators
train_batch_generator = batch_generator(x_train, y_train, batch_size)
test_batch_generator = batch_generator(x_val, y_val, batch_size)

train_writer = tf.summary.FileWriter('./logdir/train', accuracy.graph)
test_writer = tf.summary.FileWriter('./logdir/test', accuracy.graph)

session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

saver = tf.train.Saver()

if __name__ == "__main__":
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        print("Begin training...")
        for epoch in range(num_epochs):
            loss_train = 0
            loss_test = 0
            accuracy_train = 0
            accuracy_test = 0

            print("epoch: {}\t".format(epoch), end="")

            # Training in batches
            num_batches = x_train.shape[0] // batch_size
            for b in tqdm(range(num_batches)):
                x_batch, y_batch = next(train_batch_generator)

                loss_tr, acc, _, summary = sess.run([loss, accuracy, optimizer, merged],
                                                    feed_dict={input_x: x_batch, output_y: y_batch,
                                                               keep_prob: keepprob})
                accuracy_train += acc
                loss_train = loss_tr * loss_delta + loss_train * (1 - loss_delta)
                train_writer.add_summary(summary, b + num_batches * epoch)
            accuracy_train /= num_batches

            print("Training complete...")
            # Testing
            num_batches = x_val.shape[0] // batch_size
            for b in tqdm(range(num_batches)):
                x_batch, y_batch = next(test_batch_generator)

                loss_test_batch, acc, summary = sess.run([loss, accuracy, merged],
                                                         feed_dict={input_x: x_batch, output_y: y_batch,
                                                                    keep_prob: 1.0})
                accuracy_test += acc
                loss_test += loss_test_batch
                test_writer.add_summary(summary, b + num_batches * epoch)
            accuracy_test /= num_batches
            loss_test /= num_batches

            print("loss: {:.3f}, val_loss: {:.3f}, acc: {:.3f}, val_acc: {:.3f}".format(loss_train, loss_test,
                                                                                        accuracy_train, accuracy_test))
        train_writer.close()
        test_writer.close()
        saver.save(sess, model_path)
        print("Run 'tensorboard --logdir=./logdir' to checkout tensorboard logs.")
