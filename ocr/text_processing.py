import re

from textblob import Word
from typing import *

from .constants import *
from .custom_types import *


def get_rows(text: str) -> List[str]:
    return text.split('\n')


def tokenize(text: str) -> List[str]:
    return text.split()


def is_abbreviation(word: str) -> bool:
    pattern = r'(\w+\.\w+)+'
    return bool(re.match(pattern, word))


def is_special_word(word: str) -> bool:
    return word.lower() in SPECIAL_WORDS


def is_garbage(word: str) -> bool:
    pattern = r'\w'
    return not bool(re.findall(pattern, word))


def start_with_number(word: str) -> bool:
    pattern = r'\d+\w+'
    return bool(re.fullmatch(pattern, word))


def is_end_sentence(word: str) -> bool:
    return word[-1] in ['.', ';', ':', '?', '!']


def split_number_and_word(word: str) -> List[str]:
    pattern = r'(\d+)(\w+)'
    return re.match(pattern, word).groups()


def ends_with_number(word: str) -> bool:
    pattern = r'\w+\d+'
    return bool(re.fullmatch(pattern, word))


def split_word_and_number(word: str) -> List[str]:
    pattern = r'([a-z]+)([0-9]+)'
    return re.match(pattern, word, re.I).groups()


def correct_word(word: str) -> str:
    check = Word(word.lower())
    correct, score = check.spellcheck()[0]
    if correct == word.lower() or score < SPELLING_THRESHOLD:
        return word
    else:
        return correct


def process_token(token: str) -> str:
    token = token.strip('_')
    if is_garbage(token):
        return ""
    elif is_special_word(token) or is_abbreviation(token):
        return token
    elif len(token) < 3:
        return token
    elif start_with_number(token):
        num, word = split_number_and_word(token)
        correct = correct_word(word)
        return num + correct
    elif ends_with_number(token):
        word, num = split_word_and_number(token)
        correct = correct_word(word)
        return correct + num
    elif is_end_sentence(token):
        word, sign = token[:-1], token[-1]
        correct = correct_word(word)
        return correct + sign
    else:
        return correct_word(token)
