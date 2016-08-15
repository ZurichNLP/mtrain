#!/usr/bin/env python3

import logging
import random
import os

from mtrain.corpus import ParallelCorpus
from mtrain.constants import *
from mtrain.preprocessing.tokenizer import Tokenizer
from mtrain.preprocessing import lowercaser, cleaner
from mtrain import assertions, commander

class Training(object):
    '''
    Models the training process of a Moses engine, including all files and
    folders that are required and generated.
    '''

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
        self._tokenizer_source = Tokenizer(self._src_lang)
        self._tokenizer_target = Tokenizer(self._trg_lang)
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
        for component in PATH_COMPONENT:
            subdir = self._get_path(component)
            if not os.path.exists(subdir):
                os.mkdir(subdir)

    def _get_path(self, component):
        '''
        Returns the absolute path to the base directory of the given
        @param component.
        '''
        assert component in PATH_COMPONENT, "Unknown component %s" % component
        return os.path.sep.join([self._basepath, PATH_COMPONENT[component]])

    def _get_path_corpus(self, corpus, lang):
        '''
        Returns the absolute path to a corpus related to this training.
        '''
        if isinstance(corpus, list):
            corpus = '.'.join(corpus)
        return self._get_path('corpus') + os.sep + corpus + "." + lang

    def _get_path_corpus_final(self, corpus, lang):
        '''
        Returns the absolute path to a final corpus related to this training.
        '''
        return self._get_path_corpus([corpus, SUFFIX_FINAL], lang)

    def preprocess(self, base_corpus_path):
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
        # tokenize, clean, and split base corpus
        self._preprocess_base_corpus(base_corpus_path)
        # tokenize and clean separate tuning and training corpora (if applicable)
        if isinstance(self._tuning, str):
            self._preprocess_external_corpus(self._tuning, BASENAME_TUNING_CORPUS)
        if isinstance(self._evaluation, str):
            self._preprocess_external_corpus(self._evaluation, BASENAME_EVALUATION_CORPUS)
        # lowercase as needed
        self._lowercase()
        # create casing models
        #todo
        # mark final files (.final symlinks)
        self._mark_final_files()
        # close subprocesses
        self._tokenizer_source.close()
        self._tokenizer_target.close()

    def _preprocess_segment(self, segment, tokenizer):
        '''
        Tokenizes a bisegment and removes special charcters for use in Moses.
        Also checks for minimum/maximum number of tokens.

        @return the preprocessed segment. None means that the segment should be
            discarded.
        '''
        segment = segment.strip()
        tokens = tokenizer.tokenize(segment)
        if len(tokens) < self._min_tokens or len(tokens) > self._max_tokens:
            return None # means segment should be discarded
        else:
            return cleaner.clean(" ".join(tokens))

    def _preprocess_base_corpus(self, corpus_base_path):
        '''
        Splits @param corpus_base_path into training, tuning, and evaluation
        sections (as applicable). Outputs are stored in /corpus.
        '''
        # determine number of segments for tuning and evaluation, if any
        num_tune = 0 if not isinstance(self._tuning, int) else self._tuning
        num_eval = 0 if not isinstance(self._evaluation, int) else self._evaluation
        # open parallel input corpus for reading
        corpus_source = open(corpus_base_path + "." + self._src_lang, 'r')
        corpus_target = open(corpus_base_path + "." + self._trg_lang, 'r')
        # create parallel output corpora
        corpus_train = ParallelCorpus(
            self._get_path_corpus(BASENAME_TRAINING_CORPUS, self._src_lang),
            self._get_path_corpus(BASENAME_TRAINING_CORPUS, self._trg_lang),
            max_size=None
        )
        corpus_tune = ParallelCorpus(
            self._get_path_corpus(BASENAME_TUNING_CORPUS, self._src_lang),
            self._get_path_corpus(BASENAME_TUNING_CORPUS, self._trg_lang),
            max_size=num_tune
        )
        corpus_eval = ParallelCorpus(
            self._get_path_corpus(BASENAME_EVALUATION_CORPUS, self._src_lang),
            self._get_path_corpus(BASENAME_EVALUATION_CORPUS, self._trg_lang),
            max_size=num_eval
        )
        # distribute segments from input corpus to output corpora
        for i, (segment_source, segment_target) in enumerate(zip(corpus_source, corpus_target)):
            # clean segments (most importantly, remove trailing \n)
            segment_source = self._preprocess_segment(segment_source, self._tokenizer_source)
            segment_target = self._preprocess_segment(segment_target, self._tokenizer_target)
            if None in [segment_source, segment_target]:
                continue # discard segments with too few or too many tokens
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

    def _preprocess_external_corpus(self, basepath_external_corpus, basename):
        '''
        Pre-processes an external corpus into /corpus.

        @param basepath_external_corpus the basepath where the external corpus
            is located, without language code suffixes. Example: `/foo/bar/eval`
            will be expanded into `/foo/bar/eval.en` and `/foo/bar/eval.fr` in
            an EN to FR training
        @param the basename of the external corpus in the context of this
            training, e.g., `test`
        '''
        corpus = ParallelCorpus(
            self._get_path_corpus(basename, self._src_lang),
            self._get_path_corpus(basename, self._trg_lang)
        )
        # pre-process segments
        corpus_source = basepath_external_corpus + "." + self._src_lang
        corpus_target = basepath_external_corpus + "." + self._trg_lang
        for segment_source, segment_target in zip(corpus_source, corpus_target):
            # clean segments (most importantly, remove trailing \n)
            segment_source = self._preprocess_segment(segment_source, self._tokenizer_source)
            segment_target = self._preprocess_segment(segment_target, self._tokenizer_target)
            if None in [segment_source, segment_target]:
                continue # discard segments with too few or too many tokens
            else:
                corpus.insert(corpus_source, corpus_target)
        corpus.close()

    def _lowercase(self):
        '''
        Lowercases the training, tuning, and evaluation corpus.
        '''
        files_to_be_lowercased = []
        if self._evaluation:
            # always include target-side of eval corpus for case-insensitive evaluation
            files_to_be_lowercased.append((BASENAME_EVALUATION_CORPUS, self._trg_lang))
        if self._casing_strategy == SELFCASING:
            files_to_be_lowercased.append(
                (BASENAME_TRAINING_CORPUS, self._src_lang)
            )
            if self._tuning:
                files_to_be_lowercased.append(
                    (BASENAME_TUNING_CORPUS, self._src_lang)
                )
            if self._evaluation:
                files_to_be_lowercased.append(
                    (BASENAME_EVALUATION_CORPUS, self._src_lang)
                )
        elif self._casing_strategy == RECASING:
            files_to_be_lowercased.append(
                (BASENAME_TRAINING_CORPUS, self._src_lang),
                (BASENAME_TRAINING_CORPUS, self._trg_lang)
            )
            if self._tuning:
                files_to_be_lowercased.append(
                    (BASENAME_TUNING_CORPUS, self._src_lang),
                    (BASENAME_TUNING_CORPUS, self._trg_lang)
                )
            if self._evaluation:
                files_to_be_lowercased.append(
                    (BASENAME_EVALUATION_CORPUS, self._src_lang)
                )
        elif self._casing_strategy == TRUECASING:
            pass
        for basename, lang in files_to_be_lowercased:
            filepath_origin = self._get_path_corpus(basename, lang)
            filepath_dest = self._get_path_corpus([basename, SUFFIX_LOWERCASED], lang)
            lowercaser.lowercase_file(filepath_origin, filepath_dest)

    def train_truecaser(self):
        '''
        Trains a truecasing model.
        '''
        # create folder
        basepath = os.sep.join([self._get_path('engine'), 'truecaser'])
        if not assertions.dir_exists(basepath):
            os.mkdir(basepath)
        # train truecaser
        def command(lang):
            return '{script} --model {basepath}/truecaser.{lang} --corpus {corpus}'.format(
                script=MOSES_TRAIN_TRUECASER,
                basepath=basepath,
                lang=lang,
                corpus=self._get_path_corpus(BASENAME_TRAINING_CORPUS, lang)
            )
        commands = [command(self._src_lang), command(self._trg_lang)]
        commander.run_parallel(commands, "Training truecasing models")

    def truecase(self):
        '''
        Truecases the training, tuning, and evaluation corpus.
        '''
        def command(corpus, lang):
            return '{script} --model {model} < {corpus_in} > {corpus_out}'.format(
                script=MOSES_TRUECASER,
                model=os.sep.join([self._get_path('engine'), 'truecaser', 'truecaser.%s' % lang]),
                corpus_in=self._get_path_corpus(corpus, lang),
                corpus_out=self._get_path_corpus([corpus, SUFFIX_TRUECASED], lang)
            )
        commands = [
            command(BASENAME_TRAINING_CORPUS, self._src_lang),
            command(BASENAME_TRAINING_CORPUS, self._trg_lang),
        ]
        if self._tuning:
            commands.append(command(BASENAME_TUNING_CORPUS, self._src_lang))
            commands.append(command(BASENAME_TUNING_CORPUS, self._trg_lang))
        if self._evaluation:
            commands.append(command(BASENAME_EVALUATION_CORPUS, self._src_lang))
            commands.append(command(BASENAME_EVALUATION_CORPUS, self._trg_lang))
        commander.run_parallel(commands, "Truecasing corpora")

    def _train_recaser(self, num_threads, path_temp_files, keep_uncompressed=False):
        '''
        Trains a recasing engine.

        @num_threads the number of threads to be used
        @path_temp_files the directory where temp files shall be stored
        @keep_uncompressed whether or not the uncompressed language model and
            phrase table should be kept
        '''
        # create target directory
        base_dir_recaser = self._get_path('engine') + os.sep + 'recaser'
        if not assertions.dir_exists(base_dir_recaser):
            os.mkdir(base_dir_recaser)
        # train model
        commander.run(
            '{script} --corpus "{training_corpus}" --dir "{base_dir_recaser}" --train-script "{training_script}"'.format(
                script=MOSES_TRAIN_RECASER,
                training_corpus=self._get_path_corpus(BASENAME_TRAINING_CORPUS, self._trg_lang),
                base_dir_recaser=base_dir_recaser,
                training_script=MOSES_TRAIN_MODEL
            ),
            "Training recasing model"
        )
        # binarize language model
        commander.run(
            '{script} "{base_dir_recaser}/cased.kenlm.gz" "{base_dir_recaser}/cased.kenlm.bin"'.format(
                script=KENLM_BUILD_BINARY,
                base_dir_recaser=base_dir_recaser,
            ),
            "Binarizing the recaser's language model"
        )
        # compress phrase table
        commander.run(
            '{script} -in "{base_dir_recaser}/phrase-table.gz" -out "{base_dir_recaser}/phrase-table" -threads {num_threads} -T "{path_temp_files}"'.format(
                script=MOSES_COMPRESS_PHRASE_TABLE,
                base_dir_recaser=base_dir_recaser,
                num_threads=num_threads,
                path_temp_files=path_temp_files
            ),
            "Compressing the recaser's phrase table"
        )
        # Adjust moses.ini
        moses_ini = ''
        with open("%s/moses.ini" % base_dir_recaser) as f:
            moses_ini = f.read()
        moses_ini.replace('PhraseDictionaryMemory', 'PhraseDictionaryCompact')
        moses_ini.replace('phrase-table.gz', 'phrase-table')
        moses_ini.replace('cased.kenlm.gz', 'cased.kenlm.bin')
        moses_ini.replace('lazyken=1', 'lazyken=0')
        with open("%s/moses.ini" % base_dir_recaser) as f:
            f.write(moses_ini)
        # Remove uncompressed models
        if not keep_uncompressed:
            os.remove("%s/cased.kenlm.gz" % base_dir_recaser)
            os.remove("%s/phrase-table.gz" % base_dir_recaser)

    def _mark_final_files(self):
        '''
        Creates a symlink to the final training, tuning, and evaluation files.
        '''
        fp = {
            'train': {
                'src': self._get_path_corpus(BASENAME_TRAINING_CORPUS, self._src_lang),
                'trg': self._get_path_corpus(BASENAME_TRAINING_CORPUS, self._trg_lang)
            },
            'tune': {
                'src': self._get_path_corpus(BASENAME_TUNING_CORPUS, self._src_lang),
                'trg': self._get_path_corpus(BASENAME_TUNING_CORPUS, self._trg_lang)
            },
            'test': {
                'src': self._get_path_corpus(BASENAME_EVALUATION_CORPUS, self._src_lang),
                'trg': self._get_path_corpus(BASENAME_EVALUATION_CORPUS, self._trg_lang)
            }
        }
        def add_suffix(corpus, lang, suffix):
            path = fp[corpus][lang].split('.')
            path.insert(-1, suffix)
            fp[corpus][lang] = '.'.join(path)
        if self._casing_strategy == SELFCASING:
            # lowercased input side, unmodified output side
            add_suffix('train', 'src', SUFFIX_LOWERCASED)
            add_suffix('tune', 'src', SUFFIX_LOWERCASED)
            add_suffix('test', 'src', SUFFIX_LOWERCASED)
        elif self._casing_strategy == TRUECASING:
            # truecased input side, truecased output side
            add_suffix('train', 'src', SUFFIX_TRUECASED)
            add_suffix('train', 'trg', SUFFIX_TRUECASED)
            add_suffix('tune', 'src', SUFFIX_TRUECASED)
            add_suffix('tune', 'trg', SUFFIX_TRUECASED)
            add_suffix('test', 'src', SUFFIX_TRUECASED)
            add_suffix('test', 'trg', SUFFIX_TRUECASED)
        elif self._casing_strategy == RECASING:
            # truecased input side, truecased output side
            add_suffix('train', 'src', SUFFIX_LOWERCASED)
            add_suffix('train', 'trg', SUFFIX_LOWERCASED)
            add_suffix('tune', 'src', SUFFIX_LOWERCASED)
            add_suffix('tune', 'trg', SUFFIX_LOWERCASED)
            add_suffix('test', 'src', SUFFIX_LOWERCASED)
            add_suffix('test', 'trg', SUFFIX_LOWERCASED)
        # create symlinks
        def symlink_path(basename, lang):
            return self._get_path_corpus([basename, SUFFIX_FINAL], lang)
        os.symlink(
            fp['train']['src'],
            symlink_path(BASENAME_TRAINING_CORPUS, self._src_lang)
        )
        os.symlink(
            fp['train']['src'],
            symlink_path(BASENAME_TRAINING_CORPUS, self._trg_lang)
        )
        if self._tuning:
            os.symlink(
                fp['tune']['src'],
                symlink_path(BASENAME_TUNING_CORPUS, self._src_lang)
            )
            os.symlink(
                fp['tune']['trg'],
                symlink_path(BASENAME_TUNING_CORPUS, self._trg_lang)
            )
        if self._evaluation:
            os.symlink(
                fp['test']['src'],
                symlink_path(BASENAME_EVALUATION_CORPUS, self._src_lang)
            )
            os.symlink(
                fp['test']['trg'],
                symlink_path(BASENAME_EVALUATION_CORPUS, self._trg_lang)
            )

    def _train_language_model(self, n, path_temp_files, keep_uncompressed=False):
        '''
        Trains an n-gram language model with modified Kneser-Ney smoothing of
        order @param n.
        '''
        # create target directory
        base_dir_lm = self._get_path('engine') + os.sep + 'lm'
        if not assertions.dir_exists(base_dir_lm):
            os.mkdir(base_dir_lm)
        # train language model
        commander.run(
            '{script} -o {n} -S 40% -T "{path_temp_files}" < "{training_corpus}" > "{base_dir_lm}/{n}-grams.{trg_lang}.arpa"'.format(
                script=KENLM_TRAIN_MODEL,
                n=n,
                path_temp_files=path_temp_files,
                training_corpus=self._get_path_corpus_final(BASENAME_TRAINING_CORPUS, self._trg_lang),
                base_dir_lm=base_dir_lm,
                trg_lang=self._trg_lang
            ),
            "Training %s language model" % self._trg_lang
        )
        # binarize language model
        commander.run(
            '{script} "{base_dir_lm}/{n}-grams.{trg_lang}.arpa" "{base_dir_lm}/{n}-grams.{trg_lang}.bin"'.format(
                script=KENLM_BUILD_BINARY,
                base_dir_lm=base_dir_lm,
                n=n,
                trg_lang=self._trg_lang
            ),
            "Binarizing language model"
        )
        # remove uncompressed model
        if not keep_uncompressed:
            os.remove(
                '{base_dir_lm}/{n}-grams.{trg_lang}.arpa'.format(
                    base_dir_lm=base_dir_lm,
                    n=n,
                    trg_lang=self._trg_lang
                )
            )
