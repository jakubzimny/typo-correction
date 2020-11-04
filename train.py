import numpy as np
from math import ceil
from typing import List, Tuple
from tokenizer import CharacterTokenizer
from model import autoencoder
from tensorflow.keras import optimizers, losses

def add_padding_to_dataset(X: List, Y: List, max_len:int) -> Tuple:
    for i in range(0, len(X)):
        for _ in range(0, max_len - len(X[i])):
            X[i] += ' '
        for _ in range(0, max_len - len(Y[i])):
            Y[i] += ' '
    return (X, Y)

def load_dataset(filename: str) -> Tuple:
    X = []
    Y = []
    with open(filename) as f:
        content = f.read()
    lines = content.strip().split('\n')
    for line in lines:
        elements = line.split('\t')
        X.append(elements[0])
        Y.append(elements[1])
    return (X, Y)

def get_max_char_len(X, Y) -> int:
    X_max = len(max(X, key=len))
    Y_max = len(max(Y, key=len))
    return X_max if X_max > Y_max else Y_max

def encode(X: List, Y: List, tokenizer: CharacterTokenizer) -> Tuple:
    for idx, _ in enumerate(X):
        X[idx] = tokenizer.encode_one_hot(X[idx])
        Y[idx] = tokenizer.encode_one_hot(Y[idx])
    return (X, Y)

def split_data(X: List, Y: List, ratio: float = 0.8) -> Tuple:
    train_num = ceil(len(X) * ratio)
    return (np.array(X[:train_num]),
            np.array(Y[:train_num]),
            np.array(X[train_num:]),
            np.array(Y[train_num:]))

if __name__ == '__main__':
    X, Y = load_dataset('typo-corpus-r1.txt')
    max_len = get_max_char_len(X, Y)
    tokenizer = CharacterTokenizer(max_word_len=max_len)
    X, Y = add_padding_to_dataset(X, Y, max_len)
    X, Y = encode(X, Y, tokenizer)
    X_train, Y_train, X_test, Y_test = split_data(X, Y, ratio=0.9)

    ae = autoencoder(input_shape=(max_len, tokenizer.get_charset_len()))
    adam = optimizers.Adam(learning_rate=0.001)
    #cosine_loss = losses.CosineSimilarity(axis=-1)

    ae.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

    ae.summary()

    ae.fit(X_train, Y_train, epochs=150, batch_size=128)#, shuffle=True)

    score = ae.evaluate(X_test, Y_test, verbose=0)
    print(f'## \nTest set: \nLoss: {score[0]}\nAccuracy: {score[1]}')
    print('## \nSample predictions:')

    for i in range (0, 50):
        predicted_y = ae.predict(X_test[i])
        print(f'Input: {tokenizer.decode_word(X_test[i])}')
        print(f'Predicted: {tokenizer.decode_one_hot_prediction(predicted_y)}')
        print(f'Real: {tokenizer.decode_word(Y_test[i])}\n')
    
    ae.save('models/')