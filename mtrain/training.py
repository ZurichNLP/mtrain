#!/usr/bin/env python3

import logging
import random
import shutil
import errno
import sys
import os
import re

from mtrain.corpus import ParallelCorpus
from mtrain.constants import *
from mtrain.preprocessing.masking import Masker, write_masking_patterns
from mtrain.preprocessing.xmlprocessor import XmlProcessor
from mtrain.preprocessing.tokenizer import Tokenizer
from mtrain.preprocessing import lowercaser, cleaner
from mtrain.translation import TranslationEngine
from mtrain import assertions, commander

class Training(object):
    '''
    Models the training process of a Moses engine, including all files and
    folders that are required and generated.
    '''

    def __init__(self, basepath, src_lang, trg_lang, casing_strategy,
                 masking_strategy, xml_strategy, tuning, evaluation):
        '''
        Creates a new project structure at @param basepath.

        @param basepath the working directory for this training. All files and
            folders will be created and/or linked here. Should be an empty
            directory; existing files will be overwritten
        @param src_lang the language code of the source language, e.g., `en`
        @param trg_lang the language code of the target language, e.g., `fr`
        @param casing_strategy how case should be handled in preprocessing
        @param masking_strategy whether and how mask tokens should be
            introduced into segments
        @param xml_strategy whether fragments of markup in the data should be
            passed through, removed or masked
        @param tuning the tuning set to be used. The options are
            None:   do not tune (not recommended)
            INT:    select INT random lines from the base corpus for tuning
            STR:    use the parallel tuning corpus located at STR (base path
                    without language code suffixes)
        @param evaluation the evaluation set to be used. The same options as in
            @param tuning apply.

        Note: @param src_lang, @param trg_lang must be language codes recognised
            by the built-in Moses tokenizer.
        '''
        # base directory and languages
        self._basepath = basepath.rstrip(os.sep)
        self._src_lang = src_lang
        self._trg_lang = trg_lang
        # set strategies
        self._masking_strategy = masking_strategy
        self._casing_strategy = casing_strategy
        self._xml_strategy = xml_strategy
        # load components
        self._load_tokenizer()
        self._load_masker()
        self._load_xmlprocessor()
        self._tuning = tuning
        self._evaluation = evaluation
        # create base directory
        assertions.dir_exists(
            self._basepath,
            raise_exception="%s does not exist" % self._basepath
        )
        if not assertions.dir_is_empty(self._basepath, exceptions=['training.log']):
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


    def _symlink(self, orig, link_name):
        try:
            os.symlink(orig, link_name)
        except OSError as e:
            if e.errno == errno.EEXIST:
                os.remove(link_name)
                os.symlink(orig, link_name)

    def _get_path_masking_patterns(self, strategy):
        '''
        Make sure a masking directory and masking strategy subfolder exist,
        @return the path to the masking patterns file
        '''
        protected_patterns_dir = os.sep.join([self._get_path('engine'), MASKING, strategy])
        if not assertions.dir_exists(protected_patterns_dir):
            os.makedirs(protected_patterns_dir, exist_ok=True)
        return os.sep.join([protected_patterns_dir, PROTECTED_PATTERNS_FILE_NAME])

    def _load_tokenizer(self):
        # create tokenizers: masking strategy has an impact on tokenizer behaviour
        tokenizer_protects = False
        if self._masking_strategy:
            tokenizer_protects = True
            protected_patterns_path = self._get_path_masking_patterns(strategy=self._masking_strategy)
        elif self._xml_strategy == XML_MASK:
            tokenizer_protects = True
            protected_patterns_path = self._get_path_masking_patterns(strategy=XML_STRATEGIES_DEFAULTS[XML_MASK])

        if tokenizer_protects:
            write_masking_patterns(protected_patterns_path, markup_only=self._xml_strategy == XML_MASK)
            
            self._tokenizer_source = Tokenizer(
                self._src_lang,
                protect=True,
                protected_patterns_path=protected_patterns_path,
                escape=False
            )
            self._tokenizer_target = Tokenizer(
                self._trg_lang,
                protect=True,
                protected_patterns_path=protected_patterns_path,
                escape=False
            )
        else:
            self._tokenizer_source = Tokenizer(self._src_lang)
            self._tokenizer_target = Tokenizer(self._trg_lang)
    
    def _load_masker(self):
        if self._masking_strategy:
            self._masker = Masker(self._masking_strategy, escape=True)

    def _load_xmlprocessor(self):
        self._xml_processor = XmlProcessor(self._xml_strategy)

    def preprocess(self, base_corpus_path, min_tokens, max_tokens, mask, preprocess_external, process_xml):
        '''
        Preprocesses the given parallel corpus.

        @param corpus_base_path the common file prefix of the training corpus'
            source and target side
        @param min_tokens the minimum number of tokens in a segment. Segments
            with less tokens will be discarded
        @param max_tokens the maximum number of tokens in a segment. Segments
            with more tokens will be discarded
        @param mask whether or not segments should be masked
        @param preprocess_external whether or not external corpora (tune, eval)
            should be preprocessed (tokenized, masked and markup processing)
        @param process_xml whether or not the XML processing strategy should be
            applied to segments or not

        Note: The source and target side files of the parallel corpus are
            induced from concatenating @param corpus_base_path with
            self.src_lang and self.trg_lang, respectively. Example:
            `/foo/corpus`, `en` and `fr` result in `/foo/corpus.en` and
            `/foo/corpus.fr`.
        '''
        logging.info("Processing parallel corpus")
        # tokenize, clean, mask and split base corpus
        self._preprocess_base_corpus(base_corpus_path, min_tokens, max_tokens, mask, process_xml)
        # tokenize, clean, mask and xml-process separate tuning and training corpora (if applicable)
        if isinstance(self._tuning, str):
            self._preprocess_external_corpus(
                self._tuning,
                BASENAME_TUNING_CORPUS,
                min_tokens,
                max_tokens,
                preprocess_external
            )
        if isinstance(self._evaluation, str):
            self._preprocess_external_corpus(
                self._evaluation,
                BASENAME_EVALUATION_CORPUS,
                min_tokens,
                max_tokens,
                preprocess_external
            )
        # lowercase as needed
        self._lowercase()
        # create casing models
        #todo
        # mark final files (.final symlinks)
        self._mark_final_files()
        # close subprocesses
        self._tokenizer_source.close()
        self._tokenizer_target.close()

    def train_truecaser(self):
        '''
        Trains a truecasing model.
        '''
        # create folder
        basepath = os.sep.join([self._get_path('engine'), TRUECASING])
        if not assertions.dir_exists(basepath):
            os.mkdir(basepath)
        # train truecaser
        def command(lang):
            return '{script} --model {basepath}/model.{lang} --corpus {corpus}'.format(
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
                model=os.sep.join([self._get_path('engine'), 'truecasing', 'model.%s' % lang]),
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

    def train_recaser(self, num_threads, path_temp_files, keep_uncompressed=False):
        '''
        Trains a recasing engine.

        @num_threads the number of threads to be used
        @path_temp_files the directory where temp files shall be stored
        @keep_uncompressed whether or not the uncompressed language model and
            phrase table should be kept
        '''
        # create target directory
        base_dir_recaser = self._get_path('engine') + os.sep + RECASING
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
        moses_ini = moses_ini.replace('PhraseDictionaryMemory', 'PhraseDictionaryCompact')
        moses_ini = moses_ini.replace('phrase-table.gz', 'phrase-table')
        moses_ini = moses_ini.replace('cased.kenlm.gz', 'cased.kenlm.bin')
        moses_ini = moses_ini.replace('lazyken=1', 'lazyken=0')
        with open("%s/moses.ini" % base_dir_recaser, 'w') as f:
            f.write(moses_ini)
        # Remove uncompressed models
        if not keep_uncompressed:
            os.remove("%s/cased.kenlm.gz" % base_dir_recaser)
            os.remove("%s/phrase-table.gz" % base_dir_recaser)

    def train_engine(self, n=5, alignment='grow-diag-final-and',
              max_phrase_length=7, reordering='msd-bidirectional-fe',
              num_threads=1, path_temp_files='/tmp', keep_uncompressed=False):
        '''
        Trains the language, translation, and reordering models.

        @param n the order of the n-gram language model
        @param alignment the word alignment symmetrisation heuristic
        @param max_phrase_length the maximum numbers of tokens per phrase
        @param reordering the reordering paradigm
        @param num_threads the number of threads available for training
        @param path_temp_files the directory where temporary training files can
            be stored
        @param keep_uncompressed whether or not uncompressed model files should
            be kept after binarization
        '''
        self._train_language_model(n, path_temp_files, keep_uncompressed)
        self._word_alignment(alignment, num_threads)
        self._train_moses_engine(n, max_phrase_length, alignment, reordering, num_threads, path_temp_files, keep_uncompressed)

    def tune(self, num_threads=1):
        '''
        Maximises the engine's performance on the tuning corpus
        '''
        self._MERT(num_threads)

    def evaluate(self, num_threads=1, lowercase=False):
        '''
        Evaluates the engine by translating and scoring an  evaluation set
        '''
        self._multeval(num_threads, lowercase=lowercase)

    def _preprocess_segment(self, segment, min_tokens, max_tokens, tokenize=True,
                            tokenizer=None, mask=False, process_xml=False, return_mapping=False):
        '''
        Tokenizes a bisegment, escapes special characters, introduces mask tokens or
            processes markup found in the segment. Also checks for minimum and
            maximum number of tokens.

        @return the preprocessed segment. None means that the segment should be
            discarded.
        '''
        segment = segment.strip()
        if tokenize:
            segment = tokenizer.tokenize(segment, split=False)
        if process_xml:
            segment, _ = self._xml_processor.preprocess_markup(segment)
        if mask:
            segment, mapping = self._masker.mask_segment(segment)
        # check length of segment after masking and xml processing, otherwise
        # the counts will not be meaningful
        tokens = [token for token in segment.split(" ") if token != '']
        if len(tokens) < min_tokens or len(tokens) > max_tokens:
            return None # means segment should be discarded
        segment = cleaner.clean(segment)
        if return_mapping:
            return segment, mapping
        else:
            return segment

    def _preprocess_base_corpus(self, corpus_base_path, min_tokens, max_tokens, mask, process_xml):
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
            # tokenize, clean (among other things, remove trailing \n) and otherwise process segments
            segment_source = self._preprocess_segment(segment_source, min_tokens, max_tokens,
                                  tokenizer=self._tokenizer_source, mask=mask, process_xml=process_xml)
            segment_target = self._preprocess_segment(segment_target, min_tokens, max_tokens,
                                  tokenizer=self._tokenizer_source, mask=mask, process_xml=process_xml)

            if None in [segment_source, segment_target]:
                continue # discard segments with too few or too many tokens
            # reservoir sampling (Algorithm R)
            elif corpus_tune.get_size() < num_tune:
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
        # logging
        logging.info("Training corpus: %s segments", corpus_train.get_size())
        # delete empty corpora, if any
        if num_tune == 0:
            corpus_tune.delete()
        else:
            logging.info("Tuning corpus: %s segments", corpus_tune.get_size())
        if num_eval == 0:
            corpus_eval.delete()
        else:
            logging.info("Evaluation corpus: %s segments", corpus_eval.get_size())

    def _preprocess_external_corpus(self, basepath_external_corpus, basename,
                                    min_tokens, max_tokens, preprocess_segments):
        '''
        Pre-processes an external corpus into /corpus.

        @param basepath_external_corpus the basepath where the external corpus
            is located, without language code suffixes. Example: `/foo/bar/eval`
            will be expanded into `/foo/bar/eval.en` and `/foo/bar/eval.fr` in
            an EN to FR training
        @param basename the basename of the external corpus in the context of this
            training, e.g., `test`
        '''
        mask = bool(self._masking_strategy)
        process_xml = False if self._xml_strategy == XML_PASS_THROUGH else True
        corpus = ParallelCorpus(
            self._get_path_corpus(basename, self._src_lang),
            self._get_path_corpus(basename, self._trg_lang)
        )
        # pre-process segments
        corpus_source = open(basepath_external_corpus + "." + self._src_lang, 'r')
        corpus_target = open(basepath_external_corpus + "." + self._trg_lang, 'r')
        for segment_source, segment_target in zip(corpus_source, corpus_target):
            # tokenize, clean and mask segments (most importantly, remove trailing \n)
            segment_source = self._preprocess_segment(segment_source, min_tokens, max_tokens,
                tokenizer=self._tokenizer_source, mask=mask, process_xml=process_xml)
            segment_target = self._preprocess_segment(segment_target, min_tokens, max_tokens,
                tokenizer=self._tokenizer_target, mask=mask, process_xml=process_xml)
            if None in [segment_source, segment_target]:
                continue # discard segments with too few or too many tokens
            else:
                corpus.insert(segment_source, segment_target)
        corpus.close()
        corpus_source.close()
        corpus_target.close()
        # logging
        if basename == BASENAME_TUNING_CORPUS:
            logging.info("Tuning corpus: %s segments", corpus.get_size())
        elif basename == BASENAME_EVALUATION_CORPUS:
            logging.info("Evaluation corpus: %s segments", corpus.get_size())

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
                (BASENAME_TRAINING_CORPUS, self._src_lang)
            )
            files_to_be_lowercased.append(
                (BASENAME_TRAINING_CORPUS, self._trg_lang)
            )
            if self._tuning:
                files_to_be_lowercased.append(
                    (BASENAME_TUNING_CORPUS, self._src_lang)
                )
                files_to_be_lowercased.append(
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
        self._symlink(
            fp['train']['src'],
            symlink_path(BASENAME_TRAINING_CORPUS, self._src_lang)
        )
        self._symlink(
            fp['train']['trg'],
            symlink_path(BASENAME_TRAINING_CORPUS, self._trg_lang)
        )
        if self._tuning:
            self._symlink(
                fp['tune']['src'],
                symlink_path(BASENAME_TUNING_CORPUS, self._src_lang)
            )
            self._symlink(
                fp['tune']['trg'],
                symlink_path(BASENAME_TUNING_CORPUS, self._trg_lang)
            )
        if self._evaluation:
            self._symlink(
                fp['test']['src'],
                symlink_path(BASENAME_EVALUATION_CORPUS, self._src_lang)
            )
            self._symlink(
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

    def _word_alignment(self, symmetrization_heuristic, num_threads=1):
        '''
        Performs word alignment using fast_align
        '''
        # create target directory
        base_dir_tm = self._get_path('engine') + os.sep + 'tm'
        if not assertions.dir_exists(base_dir_tm):
            os.mkdir(base_dir_tm)
        base_dir_model = base_dir_tm + os.sep + 'model'
        if not assertions.dir_exists(base_dir_model):
            os.mkdir(base_dir_model)
        # join source and target corpus for use with fast_align
        corpus_source = open(self._get_path_corpus_final(BASENAME_TRAINING_CORPUS, self._src_lang), 'r')
        corpus_target = open(self._get_path_corpus_final(BASENAME_TRAINING_CORPUS, self._trg_lang), 'r')
        path_joined_corpus = base_dir_model + os.sep + BASENAME_TRAINING_CORPUS + '.%s-%s' % (self._src_lang, self._trg_lang)
        with open(path_joined_corpus, 'w') as joined_corpus:
            for segment_source, segment_target in zip(corpus_source, corpus_target):
                segment_source = segment_source.strip()
                segment_target = segment_target.strip()
                joined_corpus.write(" ||| ".join([segment_source, segment_target]) + '\n')
        # create source-target and target-source alignments in parallel
        command_forward = '{threads_environment}{script} -i {corpus} -d -o -v > {corpus}.forward'.format(
            threads_environment=('OMP_NUM_THREADS=%d ' % num_threads) if num_threads else '',
            script=FAST_ALIGN,
            corpus=path_joined_corpus
        )
        command_backward = '{threads_environment}{script} -i {corpus} -d -o -v > {corpus}.backward'.format(
            threads_environment=('OMP_NUM_THREADS=%d ' % num_threads) if num_threads else '',
            script=FAST_ALIGN,
            corpus=path_joined_corpus
        )
        commander.run_parallel(
            [command_forward, command_backward],
            "Aligning {src}–{trg} and {trg}-{src} words".format(
                src=self._src_lang,
                trg=self._trg_lang
            )
        )
        # symmetrize forward and backward alignments
        commander.run(
            '{script} -i {corpus}.forward -j {corpus}.backward -c {heuristic} > {base_dir_model}/aligned.{heuristic}'.format(
                script=ATOOLS,
                base_dir_model=base_dir_model,
                corpus=path_joined_corpus,
                heuristic=symmetrization_heuristic
            ),
            "Symmetrizing word alignment"
        )
        # remove intermediate files
        os.remove(path_joined_corpus)
        os.remove(path_joined_corpus + '.forward')
        os.remove(path_joined_corpus + '.backward')

    def _train_moses_engine(self, n, max_phrase_length, alignment, reordering,
                            num_threads, path_temp_files, keep_uncompressed):
        # create target directory (normally created in self._word_alignment() already)
        base_dir_tm = self._get_path('engine') + os.sep + 'tm'
        if not assertions.dir_exists(base_dir_tm):
            os.mkdir(base_dir_tm)
        base_dir_model = base_dir_tm + os.sep + 'model'
        if not assertions.dir_exists(base_dir_model):
            os.mkdir(base_dir_model)
        # train Moses engine
        training_command = '{script} -root-dir "{base_dir}" -corpus "{basepath_training_corpus}" -f {src} -e {trg} -alignment {alignment} -reordering {reordering} -lm "0:{n}:{path_lm}:8" -temp-dir "{temp_dir}" -cores {num_threads_half} -parallel -alignment-file "{base_dir_model}/aligned" -first-step 4 -max-phrase-length {max_phrase_length}'.format(
            script=MOSES_TRAIN_MODEL,
            base_dir=base_dir_tm,
            base_dir_model=base_dir_model,
            basepath_training_corpus=self._get_path('corpus') + os.sep + BASENAME_TRAINING_CORPUS + '.' + SUFFIX_FINAL,
            src=self._src_lang,
            trg=self._trg_lang,
            alignment=alignment,
            n=n,
            path_lm=self._get_path('engine') + os.sep + 'lm' + os.sep + "%s-grams.%s.bin" % (n, self._trg_lang),
            reordering=reordering,
            max_phrase_length=max_phrase_length,
            temp_dir=path_temp_files,
            num_threads_half=int(num_threads/2)
        )
        commander.run(training_command, "Training Moses engine")
        # compress translation and reordering models
        base_dir_compressed = base_dir_tm + os.sep + 'compressed'
        if not assertions.dir_exists(base_dir_compressed):
            os.mkdir(base_dir_compressed)
        pt_command = '{script} -in "{base_dir_model}/phrase-table.gz" -out "{base_dir_compressed}/phrase-table" -threads {num_threads} -T "{temp_dir}"'.format(
            script=MOSES_COMPRESS_PHRASE_TABLE,
            base_dir_model=base_dir_model,
            base_dir_compressed=base_dir_compressed,
            num_threads=num_threads,
            temp_dir=path_temp_files
        )
        rt_command = '{script} -in "{base_dir_model}/reordering-table.wbe-{reordering}.gz" -out "{base_dir_compressed}/reordering-table" -threads {num_threads} -T "{temp_dir}"'.format(
            script=MOSES_COMPRESS_REORDERING_TABLE,
            base_dir_model=base_dir_model,
            base_dir_compressed=base_dir_compressed,
            reordering=reordering,
            num_threads=num_threads,
            temp_dir=path_temp_files
        )
        commander.run(pt_command, "Compressing phrase table")
        commander.run(rt_command, "Compressing reordering table")
        # create moses.ini with compressed models
        path = re.compile(r'path=[^\s]+')
        with open(base_dir_model + os.sep + 'moses.ini', 'r') as orig:
            with open(base_dir_compressed + os.sep + 'moses.ini', 'w') as new:
                for line in orig.readlines():
                    line = line.strip()
                    if line.startswith("PhraseDictionaryMemory"):
                        line = line.replace("PhraseDictionaryMemory", "PhraseDictionaryCompact")
                        line = path.sub("path=" + base_dir_compressed + os.sep + 'phrase-table', line)
                    if line.startswith("LexicalReordering"):
                        line = path.sub("path=" + base_dir_compressed + os.sep + 'reordering-table', line)
                    line = line.replace("lazyken=1", "lazyken=0")
                    new.write(line + '\n')
        # delete uncompressed files
        if not keep_uncompressed:
            os.remove(base_dir_model + os.sep + 'phrase-table.gz')
            os.remove(base_dir_model + os.sep + 'reordering-table.wbe-%s.gz' % reordering)
            # todo: also remove other files related to training (word alignment, ...)

    def _MERT(self, num_threads):
        '''
        Tunes the system through Maximum Error Rate Trainign (MERT)
        '''
        # create target directory
        base_dir_tm = self._get_path('engine') + os.sep + 'tm'
        if not assertions.dir_exists(base_dir_tm):
            os.mkdir(base_dir_tm)
        base_dir_mert = base_dir_tm + os.sep + 'mert'
        if not assertions.dir_exists(base_dir_mert):
            os.mkdir(base_dir_mert)
        # tune
        mert_command = '{script} "{corpus_tuning_src}" "{corpus_tuning_trg}" "{moses_bin}" "{moses_ini}" --mertdir "{moses_bin_dir}" --working-dir "{base_dir_mert}/" --decoder-flags "-threads {num_threads} -minphr-memory -minlexr-memory" --no-filter-phrase-table'.format(
            script=MOSES_MERT,
            corpus_tuning_src=self._get_path_corpus_final(BASENAME_TUNING_CORPUS, self._src_lang),
            corpus_tuning_trg=self._get_path_corpus_final(BASENAME_TUNING_CORPUS, self._trg_lang),
            moses_bin=MOSES,
            moses_ini=base_dir_tm + os.sep + 'compressed' + os.sep + 'moses.ini',
            moses_bin_dir=MOSES_BIN,
            base_dir_mert=base_dir_mert,
            num_threads=num_threads
        )
        commander.run(mert_command, "Tuning engine through Maximum Error Rate Training (MERT)")

    def _multeval(self, num_threads, lowercase=False):
        '''
        Evaluates the engine using MultEval.
        @param num_threads max number of threads used
        @param lowercase whether to lowercase the translated segments before evaluation
        '''
        # create target directories
        base_dir_evaluation = self._get_path('evaluation')
        if not assertions.dir_exists(base_dir_evaluation):
            os.mkdir(base_dir_evaluation)
        base_dir_multeval = base_dir_evaluation + os.sep + 'multeval'
        if not assertions.dir_exists(base_dir_multeval):
            os.mkdir(base_dir_multeval)

        # translate source side of test corpus
        logging.info("Translating evaluation corpus")
        engine = TranslationEngine(self._basepath, self._src_lang, self._trg_lang)
        
        if lowercase:
            logging.info("Evaluating lowercased text")
            path_hypothesis = base_dir_multeval + os.sep + 'hypothesis.lowercased.' + self._trg_lang
            path_reference = self._get_path_corpus_final(BASENAME_EVALUATION_CORPUS, self._src_lang)
        else:
            logging.info("Evaluating cased text")
            path_hypothesis = base_dir_multeval + os.sep + 'hypothesis.' + self._trg_lang
            path_reference = self._get_path_corpus(BASENAME_EVALUATION_CORPUS, self._src_lang)

        with open(path_reference, 'r') as corpus_evaluation_src:
            with open(path_hypothesis, 'w') as hypothesis:
                for segment_source in corpus_evaluation_src:
                    segment_source = segment_source.strip()
                    translated_segment = engine.translate(segment_source, preprocess=False, lowercase=lowercase, detokenize=False)
                    hypothesis.write(translated_segment + '\n')

        # remove all engine processes
        engine.close()

        # evaluate with multeval
        output_file = base_dir_multeval + os.sep + 'hypothesis' + ('.lowercased' if lowercase else '') + '.multeval'
        meteor_argument = '--meteor.language "%s"' % self._trg_lang
        if self._trg_lang not in METEOR_LANG_CODES.keys():
            logging.warning("Target language not supported by METEOR library. Evaluation will not include METEOR scores.")
            meteor_argument = '--metrics bleu,ter,length'

        multeval_command = '{script} eval --verbosity 0 --bleu.verbosity 0 --threads "{num_threads}" {meteor_argument} --refs "{corpus_evaluation_trg}" --hyps-baseline "{hypothesis}" > "{output_file}"'.format(
            script=MULTEVAL,
            num_threads=num_threads,
            meteor_argument=meteor_argument,
            corpus_evaluation_trg=self._get_path_corpus_final(BASENAME_EVALUATION_CORPUS, self._trg_lang),
            hypothesis=path_hypothesis,
            output_file=output_file
        )
        commander.run(multeval_command, "Evaluating engine with MultEval")

    def write_final_ini(self):
        '''
        Symlinks the final moses.ini file to /engine/moses.ini
        '''
        final_moses_ini = self._get_path('engine') + os.sep + 'moses.ini'
        if self._tuning:
            moses_ini = self._get_path('engine') + os.sep + os.sep.join(['tm', 'mert', 'moses.ini'])
        else:
            moses_ini = self._get_path('engine') + os.sep + os.sep.join(['tm', 'compressed', 'moses.ini'])
        self._symlink(moses_ini, final_moses_ini)
