#!/usr/bin/env python3

'''
Stores the constants needed to execute commands.
'''

import logging
import os
from collections import OrderedDict


##################################
# Paths to software
##################################


# Paths to 3rd party packages
MOSES_HOME = os.environ.get('MOSES_HOME') if os.environ.get('MOSES_HOME') else ''  # Moses base directory
MOSES_BIN = MOSES_HOME + os.sep + 'bin'
FASTALIGN_HOME = os.environ.get('FASTALIGN_HOME') if os.environ.get('FASTALIGN_HOME') else ''  # directory storing the fast_align binaries (fast_align, atools)
MULTEVAL_HOME = os.environ.get('MULTEVAL_HOME') if os.environ.get('MULTEVAL_HOME') else ''  # MultEval base directory
# Paths to 3rd party packages for nematus backend implementation
# Cf. https://github.com/rsennrich/subword-nmt
SUBWORD_NMT_HOME = os.environ.get('SUBWORD_NMT_HOME') if os.environ.get('SUBWORD_NMT_HOME') else ''  # Subword NMT base directory
# Cf. https://github.com/EdinburghNLP/nematus
NEMATUS_HOME = os.environ.get('NEMATUS_HOME') if os.environ.get('NEMATUS_HOME') else ''  # Nematus base directory

# Paths to Moses files/scripts
MOSES = MOSES_HOME + os.sep + 'bin/moses'
MOSES_TRAIN_MODEL = MOSES_HOME + os.sep + 'scripts/training/train-model.perl'
MOSES_TOKENIZER = MOSES_HOME + os.sep + 'scripts/tokenizer/tokenizer.perl'
MOSES_DETOKENIZER = MOSES_HOME + os.sep + 'scripts/tokenizer/detokenizer.perl'
MOSES_TRUECASER = MOSES_HOME + os.sep + 'scripts/recaser/truecase.perl'
MOSES_TRAIN_TRUECASER = MOSES_HOME + os.sep + 'scripts/recaser/train-truecaser.perl'
MOSES_RECASER = ''
MOSES_TRAIN_RECASER = MOSES_HOME + os.sep + 'scripts/recaser/train-recaser.perl'
MOSES_MERT = MOSES_HOME + os.sep + 'scripts/training/mert-moses.pl'
MOSES_COMPRESS_PHRASE_TABLE = MOSES_HOME + os.sep + 'bin/processPhraseTableMin'
MOSES_COMPRESS_REORDERING_TABLE = MOSES_HOME + os.sep + 'bin/processLexicalTableMin'
MOSES_NORMALIZER = MOSES_HOME + os.sep + 'scripts/tokenizer/normalize-punctuation.perl'
MOSES_DETRUECASER = MOSES_HOME + os.sep + 'scripts/recaser/detruecase.perl'
MOSES_MULTI_BLEU = MOSES_HOME + os.sep + 'scripts/generic/multi-bleu.perl'

# Paths to KenLM files/scripts (included in Moses)
KENLM_TRAIN_MODEL = MOSES_HOME + os.sep + 'bin/lmplz'
KENLM_BUILD_BINARY = MOSES_HOME + os.sep + 'bin/build_binary'

# Paths to fast_align files/scripts
FAST_ALIGN = FASTALIGN_HOME + os.sep + 'fast_align'
ATOOLS = FASTALIGN_HOME + os.sep + 'atools'
# Path to multeval script
MULTEVAL = MULTEVAL_HOME + os.sep + 'multeval.sh'

# Paths to Subword NMT files/scripts
SUBWORD_NMT_LEARN = SUBWORD_NMT_HOME + os.sep + 'learn_bpe.py'
SUBWORD_NMT_APPLY = SUBWORD_NMT_HOME + os.sep + 'apply_bpe.py'
SUBWORD_NMT_JOINT = SUBWORD_NMT_HOME + os.sep + 'learn_joint_bpe_and_vocab.py'

# Paths to Nematus files/scripts
NEMATUS_BUILD_DICT = NEMATUS_HOME + os.sep + 'data/build_dictionary.py'
NEMATUS_NMT = NEMATUS_HOME + os.sep + 'nematus/nmt.py'
NEMATUS_TRANSLATE = NEMATUS_HOME + os.sep + 'nematus/translate.py'


######################################
# Constants related to preprocessing
# and default names of files
######################################

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

# Protected patterns relevant for tokenization and masking
# Dictionary[name of pattern / mask token] = 'regular expression'
PROTECTED_PATTERNS_FILE_NAME = 'protected-patterns.dat'
PROTECTED_PATTERNS = {}
PROTECTED_PATTERNS['xml'] = r'<\/?[a-zA-Z_][a-zA-Z_.\-0-9]*[^<>]*\/?>'
PROTECTED_PATTERNS['entity'] = r'&[a-zA-Z0-9#]+;'
PROTECTED_PATTERNS['email'] = r'[\w\-\_\.]+\@([\w\-\_]+\.)+[a-zA-Z]{2,}'
PROTECTED_PATTERNS['url'] = r'(https?:\/\/(?:www\.|(?!www))[^\s\.]+\.[^\s]{2,}|www\.[^\s]+\.[^\s]{2,})'

# Relative paths to components inside working directory
PATH_COMPONENT = {
    # Maps components to their base directory name
    "corpus": "corpus",
    "engine": "engine",
    "evaluation": "evaluation"
}

CONFIG = "config.json"

# Subfolder for bpe model and file suffix when byte-pair encoded in nematus
BPE = 'bpe'

# Default file names and affixes
BASENAME_TRAINING_CORPUS = 'train'
BASENAME_TUNING_CORPUS = 'tune'
BASENAME_EVALUATION_CORPUS = 'eval' #@MM: changed value from 'test'
SUFFIX_TOKENIZED = 'tokenized'
SUFFIX_DETOKENIZED = 'detokenized'
SUFFIX_MASKED = 'masked'
SUFFIX_UNMASKED = 'unmasked'
SUFFIX_CLEANED = 'cleaned'
SUFFIX_LOWERCASED = 'lowercased'
SUFFIX_CASED = 'cased'
SUFFIX_TRUECASED = 'truecased'
SUFFIX_WITH_MARKUP = 'with_markup'
SUFFIX_WITHOUT_MARKUP = 'without_markup'
SUFFIX_FINAL = 'final'

# Valid language codes for Moses tokenizer
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

# Valid language codes for METEOR 1.4 (used by MultEval)
METEOR_LANG_CODES = {
    # fully supported:
    "en": "English",
    "ar": "Arabic",
    "cz": "Czech",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    # only partially supported:
    "da": "Danish",
    "fi": "Finnish",
    "hu": "Hungarian",
    "it": "Italian",
    "nl": "Dutch",
    "no": "Norwegian",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "se": "Swedish",
    "tr": "Turkish"
}

# Valid casing strategies
SELFCASING = "selfcasing"
TRUECASING = "truecasing"
RECASING = "recasing"
CASING_STRATEGIES = {
    SELFCASING: "the decoder is trained on lowercased input and cased output",
    TRUECASING: "the decoder is trained on truecased input and output " +
                "(trains a separate truecasing model)",
    RECASING: "default, the decoder is trained on lowercased input and output " +
              "(trains a separate recasing model)"
}


# BPE encoding defaults
BPE_NUM_SINGLE_OPERATIONS = 89500
BPE_NUM_JOINT_OPERATIONS = 50000  # joint vocabulary


######################################
# Constants related to XML processing
######################################


# Masking
MASKING = "masking"
# Valid masking strategies
MASKING_ALIGNMENT = 'alignment'
MASKING_IDENTITY = "identity"
MASKING_STRATEGIES = {
    MASKING_ALIGNMENT: "mask tokens are not unique, content is restored based on " +
                       "the source segment and word alignment",
    MASKING_IDENTITY: "all mask tokens in a segment have unique IDs, content " +
                      "is restored based solely on mapping information",
}
# More fine-grained defaults for masking
FORCE_MASK_TRANSLATION = False  # constraint decoding for the mask token
REMOVE_ALL_MASKS = True  # whether superfluous mask tokens should be removed

# Markup reinsertion
REINSERTION = 'reinsertion'
# Valid reinsertion strategies
REINSERTION_FULL = 'full'
REINSERTION_SEGMENTATION = 'segmentation'
REINSERTION_ALIGNMENT = 'alignment'
REINSERTION_STRATEGIES = {
    REINSERTION_FULL: "reinsert markup with a hybrid method that uses both " +
                      "phrase segmentation and alignment information",
    REINSERTION_SEGMENTATION: "reinsert markup based solely on information " +
                      "about phrase segmentation",
    REINSERTION_ALIGNMENT: "reinsert markup based solely on information " +
                           "about word alignments"
}
# More fine-grained defaults for reinsertion and masking
FORCE_REINSERT_ALL = True  # whether unplaceable markup should be inserted anyway

# XML processing
XML_PASS_THROUGH = 'pass-through'
XML_STRIP = 'strip'  # for training
XML_STRIP_REINSERT = 'strip-reinsert'  # for translation
XML_MASK = 'mask'
# Valid strategies for training
XML_STRATEGIES = {
    XML_PASS_THROUGH: "do nothing, except for properly escaping special " +
                      "characters before training the models (not recommended if your " +
                      "input contains markup)",
    XML_STRIP: "remove markup from all segments before training, and do " +
               "not store the markup anywhere",
    XML_STRIP_REINSERT: "remove markup from all segments before translation " +
                        "and reinsert into the translated segment afterwards",
    XML_MASK: "replace stretches of markup with mask tokens before " +
              "training. Then train the models with segments that contain " +
              "mask tokens"
}
# More fine-grained defaults for XML processing
XML_STRATEGIES_DEFAULTS = {
    XML_STRIP: REINSERTION_FULL,
    XML_STRIP_REINSERT: REINSERTION_FULL,
    XML_MASK: MASKING_ALIGNMENT
}

# Evaluation
MULTEVAL_TOOL = 'MultEval'
EVALUATION_TOOLS = {
    MULTEVAL_TOOL: "evaluate with MultEval, computes BLEU, TER and " +
                   "METEOR scores (the latter only if target language is supported)"
}

# Python logging levels
LOGGING_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

# Backend choices
BACKEND_MOSES = 'moses'
BACKEND_NEMATUS = 'nematus'
# Valid backend choices
BACKEND_CHOICES = {
    BACKEND_MOSES: "Trains a phrase-based, statistical machine translation system based on Moses (default)",
    BACKEND_NEMATUS: "Trains a neural machine translation system based on Nematus"
}


########################################
# Constants specific to Nematus backend
########################################

TRAIN_DEVICE = 'cuda0'
VALIDATE_DEVICE = 'cuda1'
TRANS_DEVICE = TRAIN_DEVICE

TRAIN_PREALLOCATE = 0.8
VALIDATE_PREALLOCATE = 0.2
TRANS_PREALLOCATE = 0.1

VALIDATION_FREQ = 10000
MAX_EPOCHS = 5000
MAX_UPDATES = 10000000


NEMATUS_OPTIONS = {
    # training progress
    "--reload": "",  # will reload progress
    "--max_epochs": MAX_EPOCHS,
    "--finish_after": MAX_UPDATES,
    # intervals
    "--dispFreq": 100,
    "--validFreq": VALIDATION_FREQ,
    "--saveFreq": 30000,
    "--sampleFreq": 10000,
    # model
    "--dim": 512,
    "--dim_word": 1024,
    "--n_words": 90000,
    "--n_words_src": 90000,
    "--enc_depth": 1,
    "--dec_depth": 1,
    "--dropout_embedding": 0.2,
    "--dropout_hidden": 0.2,
    "--dropout_source": 0.1,
    "--dropout_target": 0.1,
    "--layer_normalisation": "",  # will use layer normalization
    "--tie_decoder_embeddings": "",  # will tie decoder embeddings
    # training procedure
    "--maxlen": 50,
    "--batch_size": 80,
    "--valid_batch_size": 80,
    "--decay_c": 0.,
    "--clip_c": 1.,
    "--lrate": 0.0001,
    "--optimizer": "adam"
}


##################################
# Python versions
##################################

# Assumes that `python` points to a Python 3 exectuable. If not, set explicitly as
# environment variable.
PYTHON2 = os.environ.get('PYTHON2') if os.environ.get('PYTHON2') else 'python2' # Python 2 base directory
PYTHON3 = os.environ.get('PYTHON3') if os.environ.get('PYTHON3') else 'python' # Python 3 base directory
