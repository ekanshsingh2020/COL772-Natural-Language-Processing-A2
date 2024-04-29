import sys,os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ['CURL_CA_BUNDLE'] = ''

import nltk 
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from gensim.models import FastText
import gensim.downloader as api
from gensim.models import KeyedVectors
from gensim.test.utils import datapath
from gensim.models.fasttext import load_facebook_vectors
import gensim
import math
import random
import numpy as np
from unidecode import unidecode

torch.manual_seed(0)
np.random.seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'

training_data = []
with open('data/A2_train.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        training_data.append(data)

# training data has question, data table, and answer labels from the the data table
        
training_tables = []
training_questions = []
columns = []
for data in training_data:
    json_table = data['table']
    table = pd.DataFrame(json_table['rows'], columns=json_table['cols'])
    for col,type in zip(json_table['cols'],json_table['types']):
        if type == 'real':
            try :
                table[col] = pd.to_numeric(table[col].str.replace(',', ''), errors='raise')
            except:
                pass
    training_tables.append(table)
    training_questions.append(data['question'])
    columns.append(data['label_col'][0])

val_data = []
with open('data/A2_val.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        val_data.append(data)

# training data has question, data table, and answer labels from the the data table
        
val_tables = []
val_questions = []
val_columns = []
for data in val_data:
    json_table = data['table']
    table = pd.DataFrame(json_table['rows'], columns=json_table['cols'])
    for col,type in zip(json_table['cols'],json_table['types']):
        if type == 'real':
            try :
                table[col] = pd.to_numeric(table[col].str.replace(',', ''), errors='raise')
            except:
                # print(table[col])
                pass
    val_tables.append(table)
    val_questions.append(data['question'])
    val_columns.append(data['label_col'][0])

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=60):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe


class TextClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers, num_heads, dropout, max_len=60):
        super(TextClassifier, self).__init__()
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers)
        

    def forward(self, text_vectors, column):
        embedded = self.positional_encoding(text_vectors)
        encoded = self.transformer_encoder(embedded)
        pooled  = torch.sum(encoded,dim = 1)
        output = torch.nn.functional.normalize(column,dim = 2) * torch.nn.functional.normalize(pooled.unsqueeze(1), dim = 2)
        output = torch.sum(output, dim=2)
        return output

# Load the FastText word vectors
model = gensim.downloader.load('glove-wiki-gigaword-100')
# model = load_facebook_vectors('cc.en.300.bin')

# Tokenize the questions and convert to torch tensors
tokenized_questions = []
for question in training_questions:
    tokens = nltk.word_tokenize(unidecode(question).lower())

    # Convert the tokens to word vectors
    vectors = []
    for token in tokens:
        try:
            vectors.append(torch.tensor(model[token]))
        except:
            pass
    # pad to 100 tokens
    while len(vectors) < 60:
        vectors.append(torch.zeros(100))
    # concatenate the vectors to one tensor
    vectors = torch.stack(vectors, dim=0)
    tokenized_questions.append(vectors)

val_tokenized_questions = []
for question in val_questions:
    tokens = nltk.word_tokenize(unidecode(question).lower())

    # Convert the tokens to word vectors
    vectors = []
    for token in tokens:
        try:
            vectors.append(torch.tensor(model[token]))
        except:
            pass
    # pad to 100 tokens
    while len(vectors) < 60:
        vectors.append(torch.zeros(100))
    # concatenate the vectors to one tensor
    vectors = torch.stack(vectors, dim=0)
    val_tokenized_questions.append(vectors)

train_labels = np.zeros((len(training_questions),64),dtype=float)
column_embeddings = []
for idx,table in enumerate(training_tables):
    cols = table.columns
    table_column_tensor = []
    x = 0
    for j,col in enumerate(cols):
        if col == columns[idx]:
            x += 1
            train_labels[idx][j] = 1.0
        tokens = nltk.word_tokenize(unidecode(col).lower())
        vectors = []
        for token in tokens:
            try:
                vectors.append(torch.tensor(model[token]))
            except:
                vectors.append(torch.zeros(100))
        # sum
        vectors = torch.sum(torch.stack(vectors, dim=0), dim = 0)
        table_column_tensor.append(vectors)
    assert(x == 1)
    while len(table_column_tensor) < 64:
        table_column_tensor.append(torch.zeros(100))
    column_embeddings.append(torch.stack(table_column_tensor,dim = 0))
train_labels = torch.Tensor(np.array(train_labels))

val_labels = np.zeros((len(val_questions),64),dtype=float)
val_column_embeddings = []
for idx,table in enumerate(val_tables):
    cols = table.columns
    table_column_tensor = []
    x = 0
    for j,col in enumerate(cols):
        if col == val_columns[idx]:
            x += 1
            val_labels[idx][j] = 1.0
        tokens = nltk.word_tokenize(unidecode(col).lower())
        vectors = []
        for token in tokens:
            try:
                vectors.append(torch.tensor(model[token]))
            except:
                vectors.append(torch.zeros(100))
        # sum
        vectors = torch.sum(torch.stack(vectors, dim=0), dim = 0)
        table_column_tensor.append(vectors)
    assert(x == 1)
    while len(table_column_tensor) < 64:
        table_column_tensor.append(torch.zeros(100))
    val_column_embeddings.append(torch.stack(table_column_tensor,dim = 0))
val_labels = torch.Tensor(np.array(val_labels))

# Create the model
embedding_dim = 100
hidden_dim = 256
output_dim = 64
num_layers = 2
num_heads = 1
dropout = 0.02
classifier = TextClassifier(embedding_dim, hidden_dim, output_dim, num_layers, num_heads, dropout).to(device)
# Train the model

class_weights = torch.tensor([1.0, 1.0/2, 1.0/3, 1.0/4, 1.0/5, 1.0/6, 1.0/7, 1.0/8, 1.0/9, 1.0/10, 1.0/11, 1.0/12, 1.0/13, 1.0/14, 1.0/15, 1.0/16, 1.0/17, 1.0/18, 1.0/19, 1.0/20, 1.0/21, 1.0/22, 1.0/23, 1.0/24, 1.0/25, 1.0/26, 1.0/27, 1.0/28, 1.0/29, 1.0/30, 1.0/31, 1.0/32, 1.0/33, 1.0/34, 1.0/35, 1.0/36, 1.0/37, 1.0/38, 1.0/39, 1.0/40, 1.0/41, 1.0/42, 1.0/43, 1.0/44, 1.0/45, 1.0/46, 1.0/47, 1.0/48, 1.0/49, 1.0/50, 1.0/51, 1.0/52, 1.0/53, 1.0/54, 1.0/55, 1.0/56, 1.0/57, 1.0/58, 1.0/59, 1.0/60, 1.0/61, 1.0/62, 1.0/63, 1.0/64]).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights) #nn.BCELoss()
optimizer = optim.Adam(classifier.parameters(), lr=0.005)
classifier.train()
class CustomDataset(Dataset):
    def __init__(self, data_list, columns_list, labels_list):
        self.data = data_list
        self.columns = columns_list
        self.labels = labels_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {
            'data': self.data[idx],
            'columns': self.columns[idx],
            'labels': self.labels[idx]
        }
        return sample

dataset = CustomDataset(tokenized_questions, column_embeddings, train_labels)
dataloader = DataLoader(dataset, batch_size=5000, shuffle=True)

val_dataset = CustomDataset(val_tokenized_questions, val_column_embeddings, val_labels)
val_dataloader = DataLoader(val_dataset, batch_size=1000, shuffle=True)

print("Training start")

# Train the model
for epoch in range(1000):
    running_loss = 0.0
    accuracy = 0
    classifier.train()
    for i, data in enumerate(dataloader, 0):
        inputs = data['data'].to(device)
        columns = data['columns'].to(device)
        labels = data['labels'].to(device)
        optimizer.zero_grad()
        outputs = classifier(inputs, columns)
        loss = criterion(outputs, labels)
        accuracy += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 5 == 4:
            print(f'Epoch {epoch + 1}, batch {i + 1}: loss {running_loss / 5}')
            print(f'Accuracy: {accuracy/(25 * 1000)}')
            running_loss = 0.0
            accuracy = 0
    classifier.eval()
    with torch.no_grad():
        val_accuracy = 0
        for i, data in enumerate(val_dataloader, 0):
            inputs = data['data'].to(device)
            columns = data['columns'].to(device)
            labels = data['labels'].to(device)
            outputs = classifier(inputs, columns)
            val_accuracy += (outputs.argmax(dim=1) == labels.argmax(dim=1)).sum().item()
        print(f'Validation accuracy: {val_accuracy/len(val_questions)}')
        if val_accuracy/len(val_questions) > 0.9:
            break
