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
    for char, replacement in reversed(MOSES_SPECIAL_CHARS.items()):
        segment = segment.replace(replacement, char)
    return segment
