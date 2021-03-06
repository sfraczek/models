#encoding=utf-8
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import io
import os
import sys
import requests
from collections import OrderedDict
import math
import random
import numpy as np
import paddle
import paddle.fluid as fluid

from paddle.fluid.dygraph.nn import Embedding


#下载语料用来训练word2vec
def download():

    corpus_url = "https://dataset.bj.bcebos.com/word2vec/text8.txt"
    web_request = requests.get(corpus_url)
    corpus = web_request.content

    with open("./text8.txt", "wb") as f:
        f.write(corpus)
    f.close()


download()


#读取text8数据
def load_text8():
    corpus = []
    with open("./text8.txt", "r") as f:
        for line in f:
            line = line.strip()
            corpus.append(line)
    f.close()

    return corpus


corpus = load_text8()

#打印前500个字符，简要看一下这个语料的样子


#对语料进行预处理（分词）
def data_preprocess(corpus):
    new_corpus = []
    for line in corpus:
        line = line.strip().lower()
        line = line.split(" ")
        new_corpus.append(line)

    return new_corpus


corpus = data_preprocess(corpus)


#构造词典，统计每个词的频率，并根据频率将每个词转换为一个整数id
def build_dict(corpus, min_freq=3):
    word_freq_dict = dict()
    for line in corpus:
        for word in line:
            if word not in word_freq_dict:
                word_freq_dict[word] = 0
            word_freq_dict[word] += 1

    word_freq_dict = sorted(
        word_freq_dict.items(), key=lambda x: x[1], reverse=True)

    word2id_dict = dict()
    word2id_freq = dict()
    id2word_dict = dict()

    word2id_freq[0] = 1.
    word2id_dict['[oov]'] = 0
    id2word_dict[0] = '[oov]'

    for word, freq in word_freq_dict:

        if freq < min_freq:
            word2id_freq[0] += freq
            continue

        curr_id = len(word2id_dict)
        word2id_dict[word] = curr_id
        word2id_freq[word2id_dict[word]] = freq
        id2word_dict[curr_id] = word

    return word2id_freq, word2id_dict, id2word_dict


word2id_freq, word2id_dict, id2word_dict = build_dict(corpus)
vocab_size = len(word2id_freq)
print("there are totoally %d different words in the corpus" % vocab_size)
for _, (word, word_id) in zip(range(50), word2id_dict.items()):
    print("word %s, its id %d, its word freq %d" %
          (word, word_id, word2id_freq[word_id]))


#把语料转换为id序列
def convert_corpus_to_id(corpus, word2id_dict):
    new_corpus = []
    for line in corpus:
        new_line = [
            word2id_dict[word]
            if word in word2id_dict else word2id_dict['[oov]'] for word in line
        ]
        new_corpus.append(new_line)
    return new_corpus


corpus = convert_corpus_to_id(corpus, word2id_dict)


#使用二次采样算法（subsampling）处理语料，强化训练效果
def subsampling(corpus, word2id_freq):
    def keep(word_id):
        return random.uniform(0, 1) < math.sqrt(1e-4 / word2id_freq[word_id] *
                                                len(corpus))

    new_corpus = []
    for line in corpus:
        new_line = [word for word in line if keep(word)]
        new_corpus.append(line)
    return new_corpus


corpus = subsampling(corpus, word2id_freq)


#构造数据，准备模型训练
def build_data(corpus,
               word2id_dict,
               word2id_freq,
               max_window_size=3,
               negative_sample_num=10):

    dataset = []

    for line in corpus:
        for center_word_idx in range(len(line)):
            window_size = random.randint(1, max_window_size)
            center_word = line[center_word_idx]

            positive_word_range = (max(0, center_word_idx - window_size), min(
                len(line) - 1, center_word_idx + window_size))
            positive_word_candidates = [
                line[idx]
                for idx in range(positive_word_range[0], positive_word_range[1]
                                 + 1)
                if idx != center_word_idx and line[idx] != line[center_word_idx]
            ]

            if not positive_word_candidates:
                continue

            for positive_word in positive_word_candidates:
                dataset.append((center_word, positive_word, 1))

            i = 0
            while i < negative_sample_num:
                negative_word_candidate = random.randint(0, vocab_size - 1)

                if negative_word_candidate not in positive_word_candidates:
                    dataset.append((center_word, negative_word_candidate, 0))
                    i += 1

    return dataset


dataset = build_data(corpus, word2id_dict, word2id_freq)
for _, (center_word, target_word, label) in zip(range(50), dataset):
    print("center_word %s, target %s, label %d" %
          (id2word_dict[center_word], id2word_dict[target_word], label))


def build_batch(dataset, batch_size, epoch_num):

    center_word_batch = []
    target_word_batch = []
    label_batch = []
    eval_word_batch = []

    for epoch in range(epoch_num):

        random.shuffle(dataset)

        for center_word, target_word, label in dataset:
            center_word_batch.append([center_word])
            target_word_batch.append([target_word])
            label_batch.append(label)

            if len(eval_word_batch) < 5:
                eval_word_batch.append([random.randint(0, 99)])
            elif len(eval_word_batch) < 10:
                eval_word_batch.append([random.randint(0, vocab_size - 1)])

            if len(center_word_batch) == batch_size:
                yield np.array(center_word_batch).astype("int64"), np.array(
                    target_word_batch).astype("int64"), np.array(
                        label_batch).astype("float32"), np.array(
                            eval_word_batch).astype("int64")
                center_word_batch = []
                target_word_batch = []
                label_batch = []
                eval_word_batch = []

    if len(center_word_batch) > 0:
        yield np.array(center_word_batch).astype("int64"), np.array(
            target_word_batch).astype("int64"), np.array(label_batch).astype(
                "float32"), np.array(eval_word_batch).astype("int64")


for _, batch in zip(range(10), build_batch(dataset, 128, 3)):
    print(batch)

#定义skip-gram训练网络结构


class SkipGram(fluid.dygraph.Layer):
    def __init__(self, name_scope, vocab_size, embedding_size, init_scale=0.1):
        super(SkipGram, self).__init__(name_scope)
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        self.embedding = Embedding(
            self.full_name(),
            size=[self.vocab_size, self.embedding_size],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-0.5 / self.embedding_size,
                    high=0.5 / self.embedding_size)))

        self.embedding_out = Embedding(
            self.full_name(),
            size=[self.vocab_size, self.embedding_size],
            dtype='float32',
            param_attr=fluid.ParamAttr(
                name='embedding_out_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-0.5 / self.embedding_size,
                    high=0.5 / self.embedding_size)))

    def forward(self, center_words, target_words, label):
        center_words_emb = self.embedding(center_words)
        target_words_emb = self.embedding_out(target_words)

        # center_words_emb = [batch_size, embedding_size]
        # target_words_emb = [batch_size, embedding_size]
        word_sim = fluid.layers.elementwise_mul(center_words_emb,
                                                target_words_emb)
        word_sim = fluid.layers.reduce_sum(word_sim, dim=-1)

        pred = fluid.layers.sigmoid(word_sim)

        loss = fluid.layers.sigmoid_cross_entropy_with_logits(word_sim, label)
        loss = fluid.layers.reduce_mean(loss)

        return pred, loss


#开始训练
batch_size = 512
epoch_num = 3
embedding_size = 200
step = 0
learning_rate = 1e-3
total_steps = len(dataset) * epoch_num // batch_size


def get_similar_tokens(query_token, k, embed):
    W = embed.numpy()
    x = W[word2id_dict[query_token]]
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    flat = cos.flatten()
    indices = np.argpartition(flat, -k)[-k:]
    indices = indices[np.argsort(-flat[indices])]
    for i in indices:  # Remove the input words
        print('for word %s, the similar word is %s' %
              (query_token, str(id2word_dict[i])))


with fluid.dygraph.guard(fluid.CUDAPlace(0)):
    skip_gram_model = SkipGram("skip_gram_model", vocab_size, embedding_size)
    adam = fluid.optimizer.AdamOptimizer(learning_rate=learning_rate)

    for center_words, target_words, label, eval_words in build_batch(
            dataset, batch_size, epoch_num):
        center_words_var = fluid.dygraph.to_variable(center_words)
        target_words_var = fluid.dygraph.to_variable(target_words)
        label_var = fluid.dygraph.to_variable(label)
        pred, loss = skip_gram_model(center_words_var, target_words_var,
                                     label_var)

        loss.backward()
        adam.minimize(loss)
        skip_gram_model.clear_gradients()

        step += 1
        if step % 100 == 0:
            print("step %d / %d, loss %.3f" %
                  (step, total_steps, loss.numpy()[0]))

        if step % 10000 == 0:
            get_similar_tokens('king', 5, skip_gram_model.embedding._w)
            get_similar_tokens('one', 5, skip_gram_model.embedding._w)
            get_similar_tokens('chip', 5, skip_gram_model.embedding._w)
