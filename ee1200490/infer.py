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


# nltk.download('punkt')
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

testing_loc = sys.argv[1]
pred_file = sys.argv[2]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

questions_test = []
tables_test = []
actual_col_test = []
qid_test = []

with open(testing_loc, 'r', encoding='utf-8') as file:
    for line in file:
        parsed_data = json.loads(line)
        questions_test.append(parsed_data['question'])
        tables_test.append(parsed_data['table'])
        actual_col_test.append(list(parsed_data['table']['cols']))
        qid_test.append(parsed_data['qid'])

# questions_test = clean_questions(questions_test)

print('Number of questions:', len(questions_test))
print('Number of tables:', len(tables_test))
print('Number of actual columns:', len(actual_col_test))
print('Number of qids is ', len(qid_test))

model = gensim.downloader.load('glove-wiki-gigaword-100')


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
dropout = 0.08

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


questions_vectors_test = word2vec_questions(questions_test)


col_embeddings_test = column_embed(actual_col_test)

classifier = Classifier(embedding_dimension, hidden_dimension, num_layers, num_heads, dropout).to(device)
criterion = nn.CrossEntropyLoss()

classifier.load_state_dict(torch.load('./model'))

classifier.eval()




print("Inferring rows.............")


class neuralNet(nn.Module):
    def __init__(self):
        super(neuralNet,self).__init__()
        self.first_layer = nn.Linear(3, 10)
        self.relu = nn.ReLU()
        self.second_layer = nn.Linear(10, 25)
        self.tanh = nn.Tanh()
        self.third_layer = nn.Linear(25, 2)

    def forward(self, x):
        x = self.first_layer(x)
        x = self.relu(x)
        x = self.second_layer(x)
        x = self.tanh(x)
        x = self.third_layer(x)
        return x

row_model = neuralNet()
criterion2 = nn.CrossEntropyLoss()

test_set = []
total_num_row =0
for i in range(len(questions_test)):
    question = unidecode(questions_test[i]).lower()
    table = tables_test[i]
    actual_col = actual_col_test[i]
    for j in range(len(table['rows'])):
        total_num_row+=1
        row = table['rows'][j]
        score = torch.empty(3)
        for k in range(len(row)):
            cell = unidecode(row[k]).lower()
            that_col = unidecode(actual_col[k]).lower()
            score[0]= score[0]+len(cell)
            score[1]= score[1]+len(cell)*len(cell)
            score[2]= score[2]+len(cell)*len(cell)*len(cell)
        test_set.append(score)


row_model.load_state_dict(torch.load('./row_final_model'))
row_model.eval()

output_row = row_model(test_set)

output_rows = []
for i in range(len(output_row)):
    if output_row[i][0]>output_row[i][1]:
        output_rows.append(0)
    else:
        output_rows.append(1)

predictions_train_row = []
idx=0
for i in range(len(questions_test)):
    ques_table_row = tables_test['rows'][i]
    temp_list = []
    for j in range (len(ques_table_row)):
        if output_rows[idx]==1:
            temp_list.append(j)
        idx+=1
    predictions_train_row.append(temp_list)


batched_data_test = []
for i in range(len(questions_vectors_test)):
    temp_list = []
    temp_list.append(questions_vectors_test[i])
    temp_list.append(col_embeddings_test[i])
    batched_data_test.append(temp_list)

print("Inferring columns.............")
# random.shuffle(batched_data_train)

acc_test = 0

column_predictions = []
for i in range(0,len(batched_data_test),1000):
    batch = batched_data_test[i:i+1000]
    batch_ques_test = []
    batch_col_test = []
    for bat in batch:
        batch_ques_test.append(bat[0])
        batch_col_test.append(bat[1])
    batch_ques_test = torch.stack(batch_ques_test, dim=0).to(device)
    batch_col_test = torch.stack(batch_col_test, dim=0).to(device)
    outputs_batch_test = classifier(batch_ques_test, batch_col_test)
    for j in range(len(outputs_batch_test)):
        column_predictions.append(torch.argmax(outputs_batch_test[j]).item())

column_name_predictions = []
for i in range(len(column_predictions)):
    if column_predictions[i] < len(actual_col_test[i]):
        column_name_predictions.append(actual_col_test[i][column_predictions[i]])
    else:
        column_name_predictions.append(actual_col_test[i][0])

print("Inferring cells.............")

final_col = column_name_predictions
final_row = predictions_train_row
final_cell = []
for i in range(len(final_row)):
    tempo = []
    for j in range(len(final_row[i])):
        tempo.append([final_row[i][j], column_name_predictions[i]])
    final_cell.append(tempo)


# make a json in this format { "qid": "<question_id>", "label_row": ["<correct_rows>"], "label_col": ["<correct_cols>"],
# "label_cell": [["<row>", "<col>"]] }
    
print("Writing to file.............")
    
final_json = []
for i in range(len(qid_test)):
    temp_dict = {}
    temp_dict['qid'] = qid_test[i]
    temp_dict['label_row'] = final_row[i]
    temp_dict['label_col'] = [final_col[i]]
    temp_dict['label_cell'] = final_cell[i]
    final_json.append(temp_dict)

with open(pred_file, 'w', encoding='utf-8') as file:
    for line in final_json:
        json.dump(line, file)
        file.write('\n')



print("Testing complete")
