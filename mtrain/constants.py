#!/usr/bin/env python3

'''
Stores the constants needed to execute commands.
'''

import os
from collections import OrderedDict

# Paths to 3rd party packages
MOSES_HOME = str(os.environ.get('MOSES_HOME')) # Moses base directory
FASTALIGN_HOME = str(os.environ.get('FASTALIGN_HOME')) # directory storing the fast_align binaries (fast_align, atools)

# Paths to Moses files/scripts
MOSES_TOKENIZER = MOSES_HOME + os.sep + 'scripts/tokenizer/tokenizer.perl'

# Characters with special meanings in Moses
# Replacement is ordered: First char listed here is replaced first, etc.
MOSES_SPECIAL_CHARS = OrderedDict()
MOSES_SPECIAL_CHARS["&"] = "&amp;"
MOSES_SPECIAL_CHARS["|"] = "&#124;"
MOSES_SPECIAL_CHARS["<"] = "&lt;"
MOSES_SPECIAL_CHARS[">"] = "&gt;"
MOSES_SPECIAL_CHARS['"'] = "&quot;"
MOSES_SPECIAL_CHARS["'"] = "&apos;"
MOSES_SPECIAL_CHARS["["] = "&#91;"
MOSES_SPECIAL_CHARS["]"] = "&#93;"

# Relative paths to components inside working directory
PATH_COMPONENT = {
    # Maps components to their base directory name
    "corpus": "corpus",
    "engine": "engine",
    "evaluation": "evaluation",
    "logs": "logs",
}

# Default file names and affixes
BASENAME_TRAINING_CORPUS = 'train'
BASENAME_TUNING_CORPUS = 'tune'
BASENAME_EVALUATION_CORPUS = 'test'
SUFFIX_TOKENIZED = 'tokenized'
SUFFIX_CLEANED = 'cleaned'
SUFFIX_LOWERCASED = 'lowercased'
SUFFIX_TRUECASED = 'truecased'

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
SELFCASING = "selfcasing"
TRUECASING = "truecasing"
RECASING = "recasing"
CASING_STRATEGIES = {
    SELFCASING: "the decoder is trained on lowercased input and cased output",
    TRUECASING: "the decoder is trained on truecased input and output " +
        "(trains a separate truecasing model)",
    RECASING: "the decoder is trained on lowercased input and output " +
        "(trains a separate recasing model)",
}
