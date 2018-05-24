#!/usr/bin/env python3

from mtrain import constants as C

"""
Replaces or cleans characters with special meaning in Moses or Nematus.
"""


def clean(segment):
    """
    @param segment segment to be cleaned
    """
    return segment  # no additional cleaning by now


def escape_special_chars(segment):
    """
    Escapes characters in @param segment with special meaning in Moses.
    """
    for char, replacement in C.MOSES_SPECIAL_CHARS.items():
        segment = segment.replace(char, replacement)
    return segment


def deescape_special_chars(segment):
    """
    Deespaces characters in @param segment with special meaning in Moses.
    """
    for char in reversed(list(C.MOSES_SPECIAL_CHARS)):  # ugly, but needed for python 3.4.2
        segment = segment.replace(C.MOSES_SPECIAL_CHARS[char], char)
    return segment
