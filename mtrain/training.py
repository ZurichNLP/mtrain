#!/usr/bin/env python3

#from mtrain.preprocessing import cleaner, lowercaser, tokenizer
from mtrain.corpus import ParallelCorpus
from mtrain.constants import *
from mtrain.preprocessing import tokenizer, lowercaser
from mtrain import assertions

import logging
import random
import os

class Training(object):
    '''
    Models the training process of a Moses engine, including all files and
    folders that are required and generated.
    '''

    PATHS = {
        # Maps components to their base directory name
        "corpus": "corpus",
        "engine": "engine",
        "evaluation": "evaluation",
        "logs": "logs",
    }

    def __init__(self, basepath, src_lang, trg_lang, casing_strategy,
        tuning, evaluation, min_tokens, max_tokens):
        '''
        Creates a new project structure at @param basepath.

        @param basepath the working directory for this training. All files and
            folders will be created and/or linked here. Should be an empty
            directory; existing files will be overwritten
        @param src_lang the language code of the source language, e.g., `en`
        @param trg_lang the language code of the target language, e.g., `fr`
        @param tuning the tuning set to be used. The options are
            NONE:   do not tune (not recommended)
            INT:    select INT random lines from the base corpus for tuning
            STR:    use the parallel tuning corpus located at STR (base path
                    without language code suffixes)
        @param evaluation the evaluation set to be used. The same options as in
            @param tuning apply.
        @param min_tokens the minimum number of tokens in a segment. Segments
            with less tokens will be discarded
        @param max_tokens the maximum number of tokens in a segment. Segments
            with more tokens will be discarded

        Note: @param src_lang, @param trg_lang must be language codes recognised
            by the built-in Moses tokenizer.
        '''
        self._basepath = basepath.rstrip(os.sep)
        self._src_lang = src_lang
        self._trg_lang = trg_lang
        self._casing_strategy = casing_strategy
        self._tuning = tuning
        self._evaluation = evaluation
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens
        assertions.dir_exists(
            self._basepath,
            raise_exception="%s does not exist" % self._basepath
        )
        if not assertions.dir_is_empty(self._basepath):
            logging.warning("Base path %s is not empty. Existing files will be overwritten.", self._basepath)
        # create sub-directories
        for component in self.PATHS:
            subdir = self._get_path(component)
            if not os.path.exists(subdir):
                os.mkdir(subdir)

    def _get_path(self, component):
        '''
        Returns the absolute path to the base directory of the given
        @param component.
        '''
        assert component in self.PATHS, "Unknown component %s" % component
        return os.path.sep.join([self._basepath, component])

    def _get_path_corpus(self, corpus, lang):
        '''
        Returns the absolute path to a corpus related to this training.
        '''
        return self._get_path('corpus') + os.sep + corpus + "." + lang

    def preprocess(self, corpus_base_path):
        '''
        Preprocesses the given parallel corpus.

        @param corpus_base_path the common file prefix of the training corpus'
            source and target side

        Note: The source and target side files of the parallel corpus are
            induced from concatenating @param corpus_base_path with
            self.src_lang and self.trg_lang, respectively. Example:
            `/foo/corpus`, `en` and `fr` result in `/foo/corpus.en` and
            `/foo/corpus.fr`.
        '''
        # split base corpus into training, tuning, and evaluation corpus (if applicable)
        self._split_corpus(corpus_base_path)
        # create symlinks to tuning and evaluation corpus (if applicable)
        self._create_symlinks()
        # tokenize
        self._tokenize()
        # clean
        self._clean()
        # lowercase as needed
        self._lowercase()

    def _split_corpus(self, corpus_base_path):
        '''
        Splits @param corpus_base_path into training, tuning, and evaluation
        sections (as applicable). Outputs are stored in /corpus.
        '''
        # helpers
        def get_corpus_filename(corpus_type, lang):
            return self._get_path('corpus') + os.sep + corpus_type + "." + lang
        # determine number of segments for tuning and evaluation, if any
        num_tune = 0 if not isinstance(self._tuning, int) else self._tuning
        num_eval = 0 if not isinstance(self._evaluation, int) else self._evaluation
        # open parallel input corpus for reading
        corpus_source = open(corpus_base_path + "." + self._src_lang, 'r')
        corpus_target = open(corpus_base_path + "." + self._trg_lang, 'r')
        # create parallel output corpora
        corpus_train = ParallelCorpus(
            get_corpus_filename(BASENAME_TRAINING_CORPUS, self._src_lang),
            get_corpus_filename(BASENAME_TRAINING_CORPUS, self._trg_lang),
            max_size=None
        )
        corpus_tune = ParallelCorpus(
            get_corpus_filename(BASENAME_TUNING_CORPUS, self._src_lang),
            get_corpus_filename(BASENAME_TUNING_CORPUS, self._trg_lang),
            max_size=num_tune
        )
        corpus_eval = ParallelCorpus(
            get_corpus_filename(BASENAME_EVALUATION_CORPUS, self._src_lang),
            get_corpus_filename(BASENAME_EVALUATION_CORPUS, self._trg_lang),
            max_size=num_eval
        )
        # distribute segments from input corpus to output corpora
        for i, (segment_source, segment_target) in enumerate(zip(corpus_source, corpus_target)):
            # clean segments (most importantly, remove trailing \n)
            segment_source = segment_source.strip()
            segment_target = segment_target.strip()
            # reservoir sampling (Algorithm R)
            if corpus_tune.get_size() < num_tune:
                corpus_tune.insert(segment_source, segment_target)
            elif corpus_eval.get_size() < num_eval:
                corpus_eval.insert(segment_source, segment_target)
            else:
                if num_tune > 0 and random.randint(0, i) < (num_tune + num_eval):
                    segment_source, segment_target = corpus_tune.insert(segment_source, segment_target)
                elif num_eval > 0 and random.randint(0, i) < (num_tune + num_eval):
                    segment_source, segment_target = corpus_eval.insert(segment_source, segment_target)
                corpus_train.insert(segment_source, segment_target)
        # close file handles
        corpus_source.close()
        corpus_target.close()
        corpus_train.close()
        corpus_tune.close()
        corpus_eval.close()
        # delete empty corpora, if any
        if num_tune == 0:
            corpus_tune.delete()
        if num_eval == 0:
            corpus_eval.delete()

    def _create_symlinks(self):
        '''
        Creates a symlink from a provided tuning or evaluation corpus into the
        project's corpus directory.
        '''
        def create_symlink(basepath, basename, lang):
            filename_origin = basepath + "." + lang
            filename_dest = self._get_path('corpus') + os.sep + basename + "." + lang
            assertions.file_exists(
                filename_origin,
                "The provided %s corpus file (%s) doesn't exist" % (filename_origin, basename)
            )
            os.symlink(filename_origin, filename_dest)
        if isinstance(self._tuning, str):
            for lang in (self._src_lang, self._trg_lang):
                create_symlink(self._tuning, BASENAME_TUNING_CORPUS, lang)
        if isinstance(self._evaluation, str):
            for lang in (self._src_lang, self._trg_lang):
                create_symlink(self._evaluation, BASENAME_EVALUATION_CORPUS, lang)

    def _tokenize(self):
        '''
        Tokenizes the training, tuning, and evaluation corpus.
        '''
        def get_args(basename, lang):
            filename_origin = self._get_path_corpus(basename, lang)
            filename_dest = self._get_path_corpus(basename + "." + SUFFIX_TOKENIZED, lang)
            return (filename_origin, filename_dest, lang)
        files_to_tokenize = [
            get_args(BASENAME_TRAINING_CORPUS, self._src_lang),
            get_args(BASENAME_TRAINING_CORPUS, self._trg_lang),
        ]
        if self._tuning:
            files_to_tokenize.append(get_args(BASENAME_TUNING_CORPUS, self._src_lang))
            files_to_tokenize.append(get_args(BASENAME_TUNING_CORPUS, self._trg_lang))
        if self._evaluation:
            files_to_tokenize.append(get_args(BASENAME_EVALUATION_CORPUS, self._src_lang))
            files_to_tokenize.append(get_args(BASENAME_EVALUATION_CORPUS, self._trg_lang))
        tokenizer.tokenize_parallel(files_to_tokenize)

    def _lowercase(self):
        '''
        Lowercases the training, tuning, and evaluation corpus.
        '''
        pass #todo

    def _clean(self):
        '''
        Cleans the the training, tuning, and evaluation corpus using Moses'
        `clean-corpus-n.perl` script.
        '''
        # use self._min_tokens, self._max_tokens
        pass
