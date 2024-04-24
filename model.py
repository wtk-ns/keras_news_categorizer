import joblib
import numpy as np
import pandas as pd
from keras.layers import Embedding, Dense, Bidirectional, LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split


class Categorizer:
    __CATEGORY_TITLE = 'category'
    __TITLE_TITLE = 'title'

    __tokenizer = None

    def __get_prepared_dataset(self, data, tokenizer):
        # titles to sequences
        title_sequences = tokenizer.texts_to_sequences(data[self.__TITLE_TITLE])
        # create padded sequences from sequence titles
        padded_title_sequences = pad_sequences(title_sequences,
                                               maxlen=1000,
                                               padding='post',
                                               truncating='post')
        # tokenize labels
        label_tokenizer = Tokenizer()
        label_tokenizer.fit_on_texts(data[self.__CATEGORY_TITLE])
        label_seq = label_tokenizer.texts_to_sequences(data[self.__CATEGORY_TITLE])
        label_seq_np = np.array(label_seq) - 1
        unique_labels = np.unique(label_seq_np)

        # get categories that will match tokenized cats
        cats = [label_tokenizer.index_word[idx + 1] for idx in unique_labels.flatten()]

        return label_seq_np, padded_title_sequences, cats

    def train_model(self, dataframe: str, delimiter: str, save_model: str):
        # read data from dataset
        data = pd.read_csv(dataframe, delimiter=delimiter)
        self.__tokenizer = Tokenizer(num_words=1000)
        self.__tokenizer.fit_on_texts(data[self.__TITLE_TITLE])

        # prepare and modify dataset to work with keras
        category, title, category_list = self.__get_prepared_dataset(data, self.__tokenizer)

        # configure model
        model = Sequential()
        model.add(Embedding(1000, 32, input_length=1000))
        model.add(Bidirectional(LSTM(32, return_sequences=True)))
        model.add(Bidirectional(LSTM(16)))
        model.add(Dense(5, activation='softmax'))

        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # split training data and test data
        train_label, test_label, train_title, test_title = train_test_split(category, title, test_size=0.2, random_state=42)

        # train model
        model.fit(train_title, train_label, epochs=10, batch_size=32, validation_split=0.2)

        # Evaluate the model
        loss, accuracy = model.evaluate(test_title, test_label)
        print(f'Model accuracy: {accuracy}')

        # create model dump
        joblib.dump(model, save_model)

        return model, category_list

    def load_trained_model(self, dataframe: str, delimiter: str, save_model: str):
        # read dataset
        data = pd.read_csv(dataframe, delimiter=delimiter)
        self.__tokenizer = Tokenizer(num_words=1000)
        self.__tokenizer.fit_on_texts(data[self.__TITLE_TITLE])

        # prepare and modify dataset to work with keras
        category, title, category_list = self.__get_prepared_dataset(data, self.__tokenizer)

        # load model
        model = joblib.load(save_model)

        return model, category_list

    def prepare_text_ro_predict(self, title):
        title_sequences = self.__tokenizer.texts_to_sequences([title])
        padded_title_sequences = pad_sequences(title_sequences,
                                               maxlen=1000,
                                               padding='post',
                                               truncating='post')
        return padded_title_sequences
