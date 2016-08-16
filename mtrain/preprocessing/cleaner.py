#!/usr/bin/env python3

'''
Replaces characters with special meaning in Moses.
'''

from mtrain.constants import *

def clean(segment):
    return segment # no additional cleaning by now

def escape_special_chars(segment):
    for char, replacement in MOSES_SPECIAL_CHARS.items():
        segment = segment.replace(char, replacement)
    return segment

def deescape_special_chars(segment):
    for char in reversed(list(MOSES_SPECIAL_CHARS)): # ugly, but needed for python 3.4.2
        segment = segment.replace(MOSES_SPECIAL_CHARS[char], char)
    return segment
