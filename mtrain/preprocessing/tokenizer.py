#!/usr/bin/env python3

from mtrain.constants import *
from mtrain.commander import run, run_parallel

'''
Tokenizes files using the default Moses tokenizer.
'''

def _get_tokenizer_command(filepath_origin, filepath_dest, lang_code):
    tokenizer_args = ["-l", lang_code, "<", filepath_origin, ">", filepath_dest]
    return " ".join([tokenizer_args])

def tokenize(filepath_origin, filepath_dest, lang_code):
    '''
    Tokenizes the file located at @filepath_origin using the Moses tokenizer
    for the language @lang_code. The tokenized output is stored at @param
    filepath_dest.
    '''
    run(
        _get_tokenizer_command(filepath_origin, filepath_dest, lang_code),
        "Tokenizing %s" % filepath_origin
    )

def tokenize_parallel(arguments):
    '''
    Tokenizes multiple files in parallel and waits untill all processes have
    completed.

    @param arguments a list of (filepath_origin, filepath_dest, lang_code)
        tripples, one per file to be tokenized.
    '''
    run_parallel(
        [_get_tokenizer_command(*a) for a in arguments],
        "Tokenizing %s files in parallel: %s" %
            (len(arguments), ", ".join([a[0] for a in arguments]))
    )
