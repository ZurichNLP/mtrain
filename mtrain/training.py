#!/usr/bin/env python3

import logging
import os

import assertions
from preprocessing import cleaner, lowercaser, tokenizer

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

    def __init__(self, basepath):
        '''
        Creates a new project structure at @param basepath.
        '''
        self.basepath = basepath
        assertions.dir_exists(
            self.basepath,
            raise_exception="%s does not exist" % self.basepath
        )
        if not assertions.dir_is_empty(self.basepath):
            logging.warning("Base path %s is not empty. Existing files will be overwritten.", self.basepath)
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
        return os.path.sep.join([self.basepath, component])

    def preprocess(self, corpus_base_path, src_lang, trg_lang, tuning=None, evaluation=None):
        '''
        Preprocesses the given parallel corpus.

        Note: @param src_lang, @param trg_lang must be language codes recognised
            by the built-in Moses tokenizer.

        Note: The source and target side files of the parallel corpus are
            induced from concatenating @param corpus_base_path with @param
            src_lang and @param trg_lang, respectively. Example: `/foo/corpus`,
            `en` and `fr` result in `/foo/corpus.en` and `/foo/corpus.fr`.

        @param corpus_base_path the common file prefix of the training corpus'
            source and target side
        @param src_lang the language code of the source language, e.g., `en`
        @param trg_lang the language code of the target language, e.g., `fr`
        @param tuning the tuning set to be used. The options are
            NONE:   do not tune (not recommended)
            INT:    select INT random lines from the base corpus for tuning
            STR:    use the parallel tuning corpus located at STR (base path
                    without language code suffixes)
        @param evaluation the evaluation set to be used. The same options as in
            @param tuning apply.
        '''

    def _tokenize(self, origin, dest):
        '''
        Tokenizes the file located at @param origin and stores the tokenized
        version at @param dest.
        '''
        pass #todo

    def _lowercase(self, origin, dest):
        '''
        Lowercases the file located at @param origin and stores the lowercased
        version at @param dest.
        '''
        pass #todo

    def _clean(self, basepath, src_lang, trg_lang, min_tokens=1, max_tokens=80):
        '''
        Cleans the parallel corpus concatenated from @param basepath, @param
        src_lang and @param trg_lang in place, using Moses'
        `clean-corpus-n.perl` script.
        '''
