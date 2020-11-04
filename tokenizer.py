from typing import List
import numpy as np

class CharacterTokenizer():

    def __init__(self, max_word_len: int):
        self._charset = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ _-')
        self._char_to_index_dict = {}
        self._index_to_char_dict = {}
        self._max_word_len = max_word_len
        self._init_translate_dicts()

    def _init_translate_dicts(self):
        for index, char in enumerate(self._charset):
            self._char_to_index_dict[char] = index
            self._index_to_char_dict[index] = char

    def get_charset_len(self) -> int:
        return len(self._charset)

    def encode_one_hot(self, word: str) -> List:
        one_hot = np.zeros((self._max_word_len, len(self._charset)), dtype=np.float32)
        for index, char in enumerate(word):
            one_hot[index, self._char_to_index_dict[char]] = 1.0
        return one_hot

    def decode_one_hot_prediction(self, one_hot_matrix: List) -> str:
        #cutoff = 1.0e-1
        adjusted_one_hot = []
        for vector in one_hot_matrix:
            max_idx = np.argmax(vector)
            adjusted_vector = np.zeros(len(vector))
            #if vector[max_idx] > cutoff:
            adjusted_vector[max_idx] = 1.0
            adjusted_one_hot.append(adjusted_vector)
        return self.decode_word(adjusted_one_hot)
                

    def decode_word(self, one_hot: List) -> str:
        word = ''
        for encoded_char in one_hot:
            if max(encoded_char) != 0:
                word += self._index_to_char_dict[np.where(encoded_char == 1.0)[0][0]]
        return word