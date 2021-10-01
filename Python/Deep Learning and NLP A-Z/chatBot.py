# -*- coding: utf-8 -*-
"""
ML Lab for the udemy course "Deep Learning and NLP A-Z"
Topic: Implementing a chatbot using Deep Learning(Seq2Seq)
@author: Anand
"""

import numpy as np
import tensorflow as tf
import re
import time

#import the dataset
lines = open("../../Data/cornell movie-dialogs corpus/movie_lines.txt", encoding="utf-8", errors="ignore").read().split("\n")
conversations = open("../../Data/cornell movie-dialogs corpus/movie_conversations.txt", encoding="utf-8", errors="ignore").read().split("\n")

#Create a dictionary for theconveration id and the conversation
id2Line = {}

for line in lines:
    values = line.split(" +++$+++ ")
    if len(values) == 5:
        id2Line[values[0]] = values[4]
 
#Create a list of conversations    
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(" +++$+++ ")[-1].replace("[", "").replace("]", "").replace("'", "").replace(" ","").split(",")
    conversations_ids.append(_conversation)
    
#Getting the Questions and Answers
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2Line[conversation[i]])
        answers.append(id2Line[conversation[i + 1]])
        
#Cleaning the data
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

#Cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))
    
#Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))    
    
    
#Create a count of each words
word2count = {}
for question in clean_questions:
   for word in question.split():
       if word in word2count.keys():
           word2count[word] += 1
       else:
            word2count[word] = 1
           
for answer in answers:
   for word in answer.split():
       if word in word2count.keys():
           word2count[word] += 1
       else:
            word2count[word] = 1

#Mapping values with the required threshold into 2 dictionaries
threshold = 20
word_number = 0
questionwords2int = {}
for word, count in word2count.items():
    if count >= threshold:
        questionwords2int[word] = word_number
        word_number += 1
word_number = 0
answerwords2int = {}
for word, count in word2count.items():
    if count >= threshold:
        answerwords2int[word] = word_number
        word_number += 1

#Adding the last tokens
tokens = ["<PAD>", "<EOS>", "<OUT>", "<SOS>"]
for token in tokens:
    questionwords2int[token] = len(questionwords2int) + 1
for token in tokens:
    answerwords2int[token] = len(answerwords2int) + 1

#Create an inverse dictionaly of the answerwords2int dictionary
answersint2words = {w_i : w for w, w_i in answerwords2int.items()}


#Add EOS to every answer
for i in range(len(clean_answers)):
    clean_answers[i] += " <EOS>"


#Translate Questions and Answers into integers
#Replace all the filtered words with <OUT>
question_to_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionwords2int :
            ints.append(questionwords2int['<OUT>'])
        else:
            ints.append(questionwords2int[word])
    question_to_int.append(ints)
            
answer_to_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerwords2int :
            ints.append(answerwords2int['<OUT>'])
        else:
            ints.append(answerwords2int[word])
    answer_to_int.append(ints)


#Sorting the questions and answers by the length of the questions.


sorted_clean_questions = []
sorted_clean_answers = []

for length in range(1,26):
    for i in enumerate(question_to_int):
        if len(i[1]) == length:
               sorted_clean_questions.append(question_to_int[i[0]])
               sorted_clean_answers.append(answer_to_int[i[0]])




#Building the Seq2Seq Mofdel

#Create placeholder for inpouts and targets

def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob


#Preprocessing the targets
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1, 1])
    preprocessed_target = tf.concat([left_side, right_side], 1)
    return preprocessed_target



# Creating the Encoder RNN
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                                    cell_bw = encoder_cell,
                                                                    sequence_length = sequence_length,
                                                                    inputs = rnn_inputs,
                                                                    dtype = tf.float32)
    return encoder_state
 
# Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)
 
# Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions
 
# Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions
 
# Building the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions




























