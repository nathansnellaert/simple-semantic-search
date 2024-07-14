import re

def replace_numbers_with_token(text, token='<NUMBER>'):
    """
    Replace all numbers in the text with a custom token.
    """
    return re.sub(r'\d+(?:\.\d+)?', token, text)

def remove_numbers(text):
    return re.sub(r'\d+', '', text)