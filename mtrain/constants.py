#!/usr/bin/env python3

'''
Stores the constants needed to execute commands.
'''

import os

# Base paths
MOSES_HOME = str(os.environ.get('MOSES_HOME')) # Moses base directory
FASTALIGN_HOME = str(os.environ.get('FASTALIGN_HOME')) # directory storing the fast_align binaries (fast_align, atools)

# Valid language codes
MOSES_TOKENIZER_LANG_CODES = {
    "ca": "Catalan",
    "cs": "Czech",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fi": "Finnish",
    "fr": "French",
    "ga": "Irish",
    "hu": "Hungarian",
    "is": "Icelandic",
    "it": "Italian",
    "lv": "Latvian",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sv": "Swedish",
    "ta": "Tamil",
}

# Valid casing strategies
CASING_STRATEGIES = {
    "selfcasing": "the decoder is trained on lowercased input and cased output",
    "truecasing": "the decoder is trained on truecased input and output " +
        "(trains a separate truecasing model)",
    "recasing": "the decoder is trained on lowercased input and output " +
        "(trains a separate recasing model)",
}
