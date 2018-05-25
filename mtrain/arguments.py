#!/usr/bin/env python3

"""
Defines command line arguments.
"""

import os
import sys
import logging
import argparse

from mtrain import constants as C

def add_lang_arguments(parser):
    """
    Adds language arguments.
    """
    parser.add_argument(
        "src_lang",
        type=str,
        help="source language code: valid choices are `" +
             "`, `".join([lang for lang in sorted(C.MOSES_TOKENIZER_LANG_CODES.keys())]) +
             "`",
        choices=C.MOSES_TOKENIZER_LANG_CODES.keys(),
        metavar='src_lang'  # overrides ugly double-listing of available choices in '--help'
    )
    parser.add_argument(
        "trg_lang",
        type=str,
        help="target language code: same valid choices as in src_lang",
        choices=C.MOSES_TOKENIZER_LANG_CODES.keys(),
        metavar='trg_lang'  # overrides ugly double-listing of available choices in '--help'
    )

def add_required_arguments(parser):
    """
    Adds positional arguments.
    """
    parser.add_argument(
        "basepath",
        type=str,
        help="common path/file prefix of the training corpus' source and " +
        "target side, e.g., `/foo/bar/training_corpus`"
    )


def add_backend_arguments(parser):
    """
    General backend arguments.
    """
    parser.add_argument(
        "--backend",
        type=str,
        help="decide which backend is to be used as training engine." +
             " Valid choices are: " +
             "; ".join(["`%s`: %s" % (name, descr) for name, descr in C.BACKEND_CHOICES.items()]),
        choices=C.BACKEND_CHOICES.keys(),
        default=C.BACKEND_MOSES
    )
    parser.add_argument(
        "--threads",
        type=int,
        help="the number of threads to be used at most, default=`8`",
        default=8
    )


def add_io_arguments(parser):
    """
    Adds arguments related to files and paths.
    """
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        help="target directory for all output. The curent working directory " +
             "($PWD) is used by default.",
        default=os.getcwd()
    )
    parser.add_argument(
        "--temp_dir",
        help="directory for temporary files created during trainig, default=`/tmp`",
        default="/tmp"
    )
    parser.add_argument(
        "--logging",
        help="logging level in STDERR, default=`INFO`",
        choices=C.LOGGING_LEVELS.keys(),
        default="INFO"
    )


def add_moses_arguments(parser):
    """
    Adds arguments specific to Moses.
    """
    moses_args = parser.add_argument_group("Moses arguments")

    moses_args.add_argument(
        "-n", "--n_gram_order",
        type=int,
        help="the language model's n-gram order, default=`5`",
        default=5
    )
    moses_args.add_argument(
        "--keep_uncompressed_models",
        help="do not delete uncompressed models created during training",
        action='store_true'
    )


def add_preprocessing_arguments(parser):
    """
    Arguments for data preprocessing.
    """
    parser.add_argument(
        "--min_tokens",
        type=int,
        help="the minimum number of tokens per segments; segments with less " +
             "tokens will be discarded, default=`1`",
        default=1
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        help="the maximum number of tokens per segments; segments with more " +
             "tokens will be discarded, default=`80`",
        default=80
    )
    parser.add_argument(
        "-c", "--caser",
        type=str,
        help="casing strategy: " +
             "; ".join(["`%s`: %s" % (name, descr) for name, descr in C.CASING_STRATEGIES.items()]),
        choices=C.CASING_STRATEGIES.keys(),
        default=C.TRUECASING
    )
    parser.add_argument(
        "-t", "--tune",
        help="enable tuning. If an integer is provided, the given number of " +
             "segments will be randomly taken from the training corpus " +
             "(basepath). Alternatively, the basepath to a separate tuning " +
             "corpus can be provided. Examples: `2000`, `/foo/bar/tuning_corpus`"
    )
    parser.add_argument(
        "--preprocess_external_tune",
        help="preprocess external tuning corpus. Don't use if " +
             "the external files provided in '--tune' are already preprocessed.",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--masking",
        help="enable masking. Valid strategies are: " +
             "; ".join(["`%s`: %s" % (name, descr) for name, descr in C.MASKING_STRATEGIES.items()]),
        choices=C.MASKING_STRATEGIES.keys(),
        default=None
    )
    parser.add_argument(
        "--xml_input",
        type=str,
        help="decide how XML fragments in the input segments should " +
             "be dealt with. Valid choices are: " +
             "; ".join(["`%s`: %s" % (name, descr) for name, descr in C.XML_STRATEGIES.items()]),
        choices=C.XML_STRATEGIES.keys(),
        default=None
    )


def add_eval_arguments(parser):
    """
    Arguments for automatic evaluation.
    """
    parser.add_argument(
        "-e", "--eval",
        help="enable evaluation. If an integer is provided, the given number " +
             "of segments will be randomly taken from the training corpus " +
             "(basepath). Alternatively, the basepath to a separate evaluation " +
             "corpus can be provided. Examples: `2000`, `/foo/bar/eval_corpus`"
    )
    parser.add_argument(
        "--eval_lowercase",
        help="lowercase reference and translation before evaluation. Otherwise," +
             "evaluation uses the engine's casing strategy.",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--extended_eval",
        action="store_true",
        help="perform multiple evaluations that vary the appearance of the test files: " +
             "lowercased or not, detokenized or not, with markup or without.",
        default=False
    )


def add_nematus_train_arguments(parser):
    """
    Adds arguments specific to Nematus.
    """
    nematus_args = parser.add_argument_group("Nematus arguments")

    nematus_args.add_argument(
        "--bpe_ops",
        type=int,
        help="decide how many byte-pair operations are to be performed when learning the " +
        "byte-pair encoding model with nematus backend, default=`89500` operations",
        default=C.BPE_NUM_JOINT_OPERATIONS
    )
    nematus_args.add_argument(
        "--device_train",
        type=str,
        help="A GPU or CPU device for training.",
        default=C.TRAIN_DEVICE
    )
    nematus_args.add_argument(
        "--preallocate_train",
        type=float,
        help="Preallocate memory on a GPU device for training.",
        default=C.TRAIN_PREALLOCATE
    )
    nematus_args.add_argument(
        "--device_validate",
        type=str,
        help="A GPU or CPU device for validation. "
             "Omit if '--external_validation_script' is provided.",
        default=C.VALIDATE_DEVICE
    )
    nematus_args.add_argument(
        "--preallocate_validate",
        type=float,
        help="Preallocate memory on a GPU device for validation.",
        default=C.VALIDATE_PREALLOCATE
    )
    nematus_args.add_argument(
        "--validation_freq",
        type=int,
        help="Perform validation of the model after X updates.",
        default=C.VALIDATION_FREQ
    )
    nematus_args.add_argument(
        "--save_freq",
        type=int,
        help="Saving a model checkpoint after X updates.",
        default=C.VALIDATION_FREQ
    )
    nematus_args.add_argument(
        "--external_validation_script",
        type=str,
        help="Optional path to external validation script that is called during " +
             "training of nematus engine. Do not use if you want mtrain to manage external validation.",
        default=None
    )
    nematus_args.add_argument(
        "--max_epochs",
        type=int,
        help="Maximum number of epochs for training.",
        default=C.MAX_EPOCHS
    )
    nematus_args.add_argument(
        "--max_updates",
        type=int,
        help="Maximum number of updates to the model.",
        default=C.MAX_UPDATES
    )


def get_training_parser():
    """
    Training arguments.
    """
    parser = argparse.ArgumentParser()
    parser.description = ("Trains either an SMT system with Moses " +
                          "or an NMT system with Nematus.")

    add_required_arguments(parser)
    add_lang_arguments(parser)

    add_backend_arguments(parser)
    add_io_arguments(parser)
    add_preprocessing_arguments(parser)
    add_eval_arguments(parser)

    # options specific to a backend
    add_moses_arguments(parser)
    add_nematus_train_arguments(parser)

    return parser


def add_pre_postprocessing_arguments(parser):
    """
    Processing before and after translation.
    """
    parser.add_argument(
        "--xml_input",
        type=str,
        help="decide how XML fragments in the input segments should " +
        "be dealt with. Valid choices are: " +
        "; ".join(["`%s`: %s" % (name, descr) for name, descr \
             in C.XML_STRATEGIES.items()]),
        choices=C.XML_STRATEGIES.keys()
    )
    parser.add_argument(
        "-l", "--lowercase",
        help="lowercase segments after translation",
        default=False,
        action='store_true'
    )
    parser.add_argument(
        "--skip_preprocess",
        help="do not preprocess segments at all before translation",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--skip_detokenize",
        help="do not detokenize segments after translation",
        default=False,
        action="store_true"
    )

def add_nematus_trans_arguments(parser):
    """
    Translation options specific to Nematus.
    """
    nematus_args = parser.add_argument_group("Nematus arguments")

    nematus_args.add_argument(
        "--device",
        type=str,
        help="GPU or CPU device for translation.",
        default=C.TRANS_DEVICE
    )
    nematus_args.add_argument(
        "--preallocate",
        type=float,
        help="Preallocate memory on a GPU device for translation.",
        default=C.TRANS_PREALLOCATE
    )
    nematus_args.add_argument(
        "--adjust_dictionary",
        help="Ensures that dictionary paths referred to in model config (file model.npz.json in model basepath) " +
        "match the actual dictionaries (.json files for source and target language in the corpus basepath). " +
        "Adjustment is necessary when the model was trained with `mtrain` and saved to a path different to the " +
        "`mtrans` 'basepath'. If paths are not matching, nematus returns an empty string for any " +
        "translation without error message OR may use the wrong .json files for translation.",
        default=False,
        action="store_true"
    )

def get_translation_parser():
    """
    Command line argument for translation.
    """
    parser = argparse.ArgumentParser()
    parser.description = ("Translates text using a trained mtrain engine.")

    parser.add_argument(
        "basepath",
        type=str,
        help="basepath of the machine translation system, i.e., the output " +
        "directory ('-o') used in `mtrain`."
    )
    parser.add_argument(
        "--logging",
        help="logging level in STDERR, default=`INFO`",
        choices=C.LOGGING_LEVELS.keys(),
        default="INFO"
    )

    add_pre_postprocessing_arguments(parser)
    add_nematus_trans_arguments(parser)

    return parser


def check_train_arguments_moses(args):
    """
    Check for incompatible arguments.

    @param args all arguments passed from 'get_argument_parser()'
    """
    # generic masking and XML masking currently not possible at the same time for backend moses
    if args.masking and args.xml_input == C.XML_MASK:
        logging.critical("Invalid command line options. Choose either '--masking' or '--xml_input mask', but not both. See '-h'/'--help' for more information.")
        sys.exit()


def check_train_arguments_nematus(args):
    """
    Check for incompatible arguments.

    @param args all arguments passed from 'get_argument_parser()'
    """
    pass


def check_train_arguments(args):
    """
    Check for incompatible arguments.
    """
    if args.backend == C.BACKEND_MOSES:
        check_train_arguments_moses(args)
    else:
        check_train_arguments_nematus(args)


def check_trans_arguments_moses(args):
    """
    Check for incompatible arguments.

    @param args all arguments passed from 'get_argument_parser()'
    """
    pass


def check_trans_arguments_nematus(args):
    """
    Check for arguments if fit for nematus, either combination or specific argument may
    be not (yet) applicable for the backend. Depending on severity, user is warned and maybe
    program terminated.

    @param args all arguments passed from 'get_argument_parser()'
    """
    pass


def check_trans_arguments(args):
    """
    Check for incompatible arguments.
    """
    if args.backend == C.BACKEND_MOSES:
        check_trans_arguments_moses(args)
    else:
        check_trans_arguments_nematus(args)
