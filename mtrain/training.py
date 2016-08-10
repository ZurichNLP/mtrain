#!/usr/bin/env python3

#from mtrain.preprocessing import cleaner, lowercaser, tokenizer
from mtrain.corpus import ParallelCorpus
from mtrain.constants import *
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

    def __init__(self, basepath):
        '''
        Creates a new project structure at @param basepath.
        '''
        self._basepath = basepath.rstrip(os.sep)
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
        num_tuning = 0 if not isinstance(tuning, int) else tuning
        num_eval = 0 if not isinstance(evaluation, int) else evaluation
        self._split_corpus(corpus_base_path, src_lang, trg_lang, num_tuning, num_eval)

    def _split_corpus(self, corpus_base_path, src_lang, trg_lang, num_tuning=0, num_eval=0):
        '''
        Selects @num_tuning and @num_eval random segments from the parallel
        corpus located at @source_filepath/@target_filepath for tuning and
        evaluation, respectively. Outputs are stored at /corpus.
        '''
        corpus_base_name = os.path.basename(corpus_base_path)
        def get_corpus_filename(corpus_type, lang):
            return self._basepath + os.sep + self.PATHS['corpus'] + os.sep + corpus_base_name + "." + corpus_type + "." + lang
        # open parallel input corpus for reading
        corpus_source = open(corpus_base_path + "." + src_lang, 'r')
        corpus_target = open(corpus_base_path + "." + trg_lang, 'r')
        # create parallel output corpora
        corpus_train = ParallelCorpus(
            get_corpus_filename(FILE_SUFFIX_TRAINING_CORPUS, src_lang),
            get_corpus_filename(FILE_SUFFIX_TRAINING_CORPUS, trg_lang),
            max_size=None
        )
        corpus_tune = ParallelCorpus(
            get_corpus_filename(FILE_SUFFIX_TUNING_CORPUS, src_lang),
            get_corpus_filename(FILE_SUFFIX_TUNING_CORPUS, trg_lang),
            max_size=num_tuning
        )
        corpus_eval = ParallelCorpus(
            get_corpus_filename(FILE_SUFFIX_EVALUATION_CORPUS, src_lang),
            get_corpus_filename(FILE_SUFFIX_EVALUATION_CORPUS, trg_lang),
            max_size=num_eval
        )
        # distribute segments from input corpus to output corpora
        for i, (segment_source, segment_target) in enumerate(zip(corpus_source, corpus_target)):
            # clean segments (most importantly, remove trailing \n)
            segment_source = segment_source.strip()
            segment_target = segment_target.strip()
            # reservoir sampling (Algorithm R)
            if corpus_tune.get_size() < num_tuning:
                corpus_tune.insert(segment_source, segment_target)
            elif corpus_eval.get_size() < num_eval:
                corpus_eval.insert(segment_source, segment_target)
            else:
                if num_tuning > 0 and random.randint(0, i) < (num_tuning + num_eval):
                    segment_source, segment_target = corpus_tune.insert(segment_source, segment_target)
                elif num_eval > 0 and random.randint(0, i) < (num_tuning + num_eval):
                    segment_source, segment_target = corpus_eval.insert(segment_source, segment_target)
                corpus_train.insert(segment_source, segment_target)
        # close file handles
        corpus_source.close()
        corpus_target.close()
        corpus_train.close()
        corpus_tune.close()
        corpus_eval.close()
        # delete empty corpora, if any
        if num_tuning == 0:
            corpus_tune.delete()
        if num_eval == 0:
            corpus_eval.delete()

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
