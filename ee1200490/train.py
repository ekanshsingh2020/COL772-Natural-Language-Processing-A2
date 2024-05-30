import nltk
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gensim.downloader as api
from gensim.test.utils import datapath
import gensim
import math
import re
import random
import numpy as np
import sys
from unidecode import unidecode

torch.manual_seed(0)
np.random.seed(0)

nltk.download('punkt')
# from google.colab import drive
# drive.mount('data')

# def clean_questions(questions):
#     cleaned_questions = []
#     for question in questions:
#         question = question.lower()
#         question = unidecode(question)
#         cleaned_question = re.sub(r"[^\w\s]", "", question) 
#         cleaned_questions.append(cleaned_question)
#     return cleaned_questions


training_loc = sys.argv[1]
testing_loc = sys.argv[2]


questions_train = []
tables_train = []
actual_col_train = []
label_cols_train = []
label_rows_train = []
with open(training_loc, 'r', encoding='utf-8') as file:
    for line in file:
        parsed_data = json.loads(line)
        questions_train.append(parsed_data['question'])
        tables_train.append(parsed_data['table'])
        label_cols_train.append(parsed_data['label_col'][0])
        actual_col_train.append(list(parsed_data['table']['cols']))
        label_rows_train.append(list(parsed_data['label_row']))

# questions_train = clean_questions(questions_train)

print('Number of questions:', len(questions_train))
print('Number of tables:', len(tables_train))
print('Number of label columns:', len(label_cols_train))
print('Number of actual columns:', len(actual_col_train))
print('Number of label rows:', len(label_rows_train))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

questions_test = []
tables_test = []
actual_col_test = []
label_cols_test = []
label_rows_test = []
qid_test = []

with open(testing_loc, 'r', encoding='utf-8') as file:
    for line in file:
        parsed_data = json.loads(line)
        questions_test.append(parsed_data['question'])
        tables_test.append(parsed_data['table'])
        label_cols_test.append(parsed_data['label_col'][0])
        actual_col_test.append(list(parsed_data['table']['cols']))
        qid_test.append(parsed_data['qid'])
        label_rows_test.append(list(parsed_data['label_row']))

# questions_test = clean_questions(questions_test)

print('Number of questions:', len(questions_test))
print('Number of tables:', len(tables_test))
print('Number of label columns:', len(label_cols_test))
print('Number of actual columns:', len(actual_col_test))
print('Number of qids is ', len(qid_test))
print('Number of label rows:', len(label_rows_test))


model = gensim.downloader.load('glove-wiki-gigaword-100')

# predictions_train_row = []
# for i in range(len(questions_train)):
#     question = unidecode(questions_train[i]).lower()
#     table = tables_train[i]
#     actual_col = actual_col_train[i]
#     label_col = label_cols_train[i][0]
#     label_row = label_rows_train[i]
#     predicted_row = []
#     score_of_row = []
#     for j in range(len(table['rows'])):
#         row = table['rows'][j]
#         score = 0.0
#         for k in range(len(row)):
#             cell = unidecode(row[k]).lower()
#             that_col = unidecode(actual_col[k]).lower()
#             if cell in question:
#                 if that_col in question:
#                     distance = abs(question.index(cell) - question.index(that_col))
#                     score += (43.0)*(len(cell)*len(cell))/(distance+1)
#                 else:
#                     score += len(cell)*len(cell)
#         score_of_row.append(score)
    
#     max_score = max(score_of_row)
#     for i in range(len(score_of_row)):
#         if score_of_row[i] == max_score:
#             predicted_row.append(i)
#     predictions_train_row.append(predicted_row)



max_len_question = 60

class PositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        pos_em = torch.zeros(max_len_question, embedding_dim)
        division = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        position = torch.arange(0, max_len_question, dtype=torch.float).unsqueeze(1)
        pos_em[:, 0::2] = torch.sin(position * division)
        pos_em[:, 1::2] = torch.cos(position * division)
        self.register_buffer('pos_em', pos_em)

    def forward(self, temp):
        return temp+self.pos_em
    
embedding_dimension = 100
hidden_dimension = 250
num_layers = 2
num_heads = 1
dropout = 0.03

class Classifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_heads, dropout):
        super(Classifier, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.pos_embed = PositionalEmbedding(embedding_dim)

    def forward(self, text_vectors, column):
        input_embedding = self.pos_embed(text_vectors)
        contextual_embedding = self.transformer_encoder(input_embedding)
        question_embedding  = torch.sum(contextual_embedding,dim = 1)
        mat_mul = torch.nn.functional.normalize(column,dim = 2) * torch.nn.functional.normalize(question_embedding.unsqueeze(1), dim = 2)
        dot_prod = torch.sum(mat_mul, dim=2)
        return dot_prod


def word2vec_questions(questions):
    final_word2vec = []
    for i in range(len(questions)):
        ques = questions[i]
        ques = unidecode(ques)
        ques_tokens = nltk.word_tokenize(ques.lower())
        word2vec = []
        for j in range(min(60,len(ques_tokens))):
            token = ques_tokens[j]
            try:
                word2vec.append(torch.tensor(model[token]))
            except:
                pass
        while len(word2vec) < max_len_question:
            word2vec.append(torch.zeros(100))
        word2vec = torch.stack(word2vec, dim=0)
        final_word2vec.append(word2vec)
    return final_word2vec

def one_hot_label(actual_col, label_col):
    one_hot = torch.zeros((len(actual_col), 64), dtype=float)
    for i in range(len(actual_col)):
        for j in range(len(actual_col[i])):
            if actual_col[i][j] == label_col[i]:
                one_hot[i][j] = 1.0
    return one_hot

def column_embed(actual_col):
    final_embed = []
    for i in range(len(actual_col)):
        col = actual_col[i]
        word_embed = []
        for j in range(len(col)):
            temp = unidecode(col[j])
            tokens = nltk.word_tokenize(temp.lower())
            within_word_embed = []
            for token in tokens:
                try:
                    within_word_embed.append(torch.tensor(model[token]))
                except:
                    within_word_embed.append(torch.zeros(100))
            within_word_embed = torch.sum(torch.stack(within_word_embed, dim=0), dim = 0)
            word_embed.append(within_word_embed)
        while len(word_embed) < 64:
            word_embed.append(torch.zeros(100))
        final_embed.append(torch.stack(word_embed, dim=0))
    return final_embed


questions_vectors_train = word2vec_questions(questions_train)
questions_vectors_test = word2vec_questions(questions_test)

one_hot_label_train = one_hot_label(actual_col_train, label_cols_train)
one_hot_label_test = one_hot_label(actual_col_test, label_cols_test)

col_embeddings_train = column_embed(actual_col_train)
col_embeddings_test = column_embed(actual_col_test)

classifier = Classifier(embedding_dimension, hidden_dimension, num_layers, num_heads, dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.005)

classifier.train()

batched_data_train = []
for i in range(len(questions_vectors_train)):
    temp_list = []
    temp_list.append(questions_vectors_train[i])
    temp_list.append(col_embeddings_train[i])
    temp_list.append(one_hot_label_train[i])
    batched_data_train.append(temp_list)

batched_data_test = []
for i in range(len(questions_vectors_test)):
    temp_list = []
    temp_list.append(questions_vectors_test[i])
    temp_list.append(col_embeddings_test[i])
    temp_list.append(one_hot_label_test[i])
    batched_data_test.append(temp_list)

print("Training the model...")
random.shuffle(batched_data_train)
best_acc = 0
best_model = './model'


for epoch in range(600):
    acc_train = 0
    # random.shuffle(batched_data)
    k=0
    for i in range(0, len(batched_data_train), 5000):
        k+=1
        batch = batched_data_train[i:i+5000]
        # print(len(batch))
        batch_ques_train = []
        batch_col_train = []
        batch_lab_train = []
        for bat in batch:
            batch_ques_train.append(bat[0])
            batch_col_train.append(bat[1])
            batch_lab_train.append(bat[2])
        batch_ques_train = torch.stack(batch_ques_train, dim=0).to(device)
        batch_col_train = torch.stack(batch_col_train, dim=0).to(device)
        batch_lab_train = torch.stack(batch_lab_train, dim=0).to(device)
        optimizer.zero_grad()
        outputs_batch_train = classifier(batch_ques_train, batch_col_train)
        loss = criterion(outputs_batch_train, batch_lab_train)
        acc_train += (outputs_batch_train.argmax(dim=1) == batch_lab_train.argmax(dim=1)).sum().item()
        loss.backward()
        optimizer.step()
        if k == 5:
            print(f'Epoch {epoch + 1}')
            acc_train= 0
    classifier.eval()
    with torch.no_grad():
        acc_test = 0
        for i in range(0, len(batched_data_test), 1000):
            batch = batched_data_test[i:i+1000]
            batch_ques_test = []
            batch_col_test = []
            batch_lab_test = []
            for bat in batch:
                batch_ques_test.append(bat[0])
                batch_col_test.append(bat[1])
                batch_lab_test.append(bat[2])
            batch_ques_test = torch.stack(batch_ques_test, dim=0).to(device)
            batch_col_test = torch.stack(batch_col_test, dim=0).to(device)
            batch_lab_test = torch.stack(batch_lab_test, dim=0).to(device)
            outputs_batch_test = classifier(batch_ques_test, batch_col_test)
            acc_test += (outputs_batch_test.argmax(dim=1) == batch_lab_test.argmax(dim=1)).sum().item()
        print(f'Accuracy of the network on the test data: {acc_test/len(questions_test)}')
        if acc_test > best_acc:
            best_acc = acc_test
            torch.save(classifier.state_dict(), best_model)


print("Training complete")

# torch.save(classifier.state_dict(), './model')