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
from mtrain.preprocessing.bpe import Encoder
from mtrain.preprocessing import lowercaser, cleaner, reinsertion 
from mtrain.translation import TranslationEngine
from mtrain import assertions, commander
from mtrain import evaluator

class Training(object):
    '''
    Models the training process of a Moses or Nematus engine, including all files and
    folders that are required and generated.
    '''

    def __init__(self, basepath, src_lang, trg_lang, backend, casing_strategy,
                 masking_strategy, xml_strategy, tuning, evaluation):
        '''
        Creates a new project structure at @param basepath.

        @param basepath the working directory for this training. All files and
            folders will be created and/or linked here. Should be an empty
            directory; existing files will be overwritten
        @param src_lang the language code of the source language, e.g., `en`
        @param trg_lang the language code of the target language, e.g., `fr`
        @param backend the backend to be used for training (moses or nematus)
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
        # set backend choice
        self._backend = backend
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
        self._num_malformed_segments = 0
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

    def _get_path_masking_patterns(self, overall_strategy, detailed_strategy):
        '''
        Make sure a masking directory and masking strategy subfolder exist,
        @return the path to the masking patterns file.
        @param overall_strategy coarse-grained masking or XML strategy
        @param detailed_strategy fine-grained masking or XML strategy
        '''
        protected_patterns_dir = os.sep.join([self._get_path('engine'), overall_strategy, detailed_strategy])
        if not assertions.dir_exists(protected_patterns_dir):
            os.makedirs(protected_patterns_dir, exist_ok=True)
        return os.sep.join([protected_patterns_dir, PROTECTED_PATTERNS_FILE_NAME])

    def _load_tokenizer(self):
        # create tokenizers: masking strategy has an impact on tokenizer behaviour

        # for backend choice moses
        if self._backend == BACKEND_MOSES:

            tokenizer_protects = False

            if self._masking_strategy:
                tokenizer_protects = True
                protected_patterns_path = self._get_path_masking_patterns(
                    overall_strategy=MASK,
                    detailed_strategy=self._masking_strategy
                )
            elif self._xml_strategy == XML_MASK:
                tokenizer_protects = True
                protected_patterns_path = self._get_path_masking_patterns(
                    overall_strategy=XML_MASK,
                    detailed_strategy=XML_STRATEGIES_DEFAULTS[XML_MASK]
                )
            elif self._xml_strategy == XML_STRIP or self._xml_strategy == XML_STRIP_REINSERT:
                tokenizer_protects=True
                protected_patterns_path = self._get_path_masking_patterns(
                    overall_strategy=self._xml_strategy,
                    detailed_strategy=XML_STRATEGIES_DEFAULTS[self._xml_strategy]
                )

            if tokenizer_protects:
                write_masking_patterns(
                    protected_patterns_path,
                    markup_only=bool(self._xml_strategy)
                )

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

        # backend choice nematus (so far neither masking_strategy nor xml_strategy)
        elif self._backend == BACKEND_NEMATUS:

            self._tokenizer_source = Tokenizer(self._src_lang)
            self._tokenizer_target = Tokenizer(self._trg_lang)

    def _load_masker(self):
        if self._masking_strategy:
            self._masker = Masker(self._masking_strategy, escape=True)

    def _load_xmlprocessor(self):
        self._xml_processor = XmlProcessor(self._xml_strategy)

    def preprocess(self, corpus_base_path, min_tokens, max_tokens, mask, preprocess_external, process_xml): ###BH changed 'base_corpus_path' to 'corpus_base_path'
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

        # logging info for backend choice and processing
        logging.info("Initiating preprocessing for backend %s", self._backend)
        logging.info("Processing parallel corpus")

        # tokenize, clean, mask and split base corpus
        self._preprocess_base_corpus(corpus_base_path, min_tokens, max_tokens, mask, process_xml) ###BH changed 'base_corpus_path' to 'corpus_base_path'
        # tokenize, clean, mask and xml-process separate tuning and training corpora (if applicable)
        if isinstance(self._tuning, str):
            self._preprocess_external_corpus(
                self._tuning,
                BASENAME_TUNING_CORPUS,
                min_tokens,
                max_tokens,
                preprocess_external=preprocess_external,
                process_xml=process_xml
            )
        if isinstance(self._evaluation, str):
            self._preprocess_external_corpus(
                self._evaluation,
                BASENAME_EVALUATION_CORPUS,
                min_tokens,
                max_tokens,
                preprocess_external=False, # never preprocess EVAL corpus
                process_xml=process_xml
            )
        # lowercase as needed
        self._lowercase()
        # mark final files (.final symlinks)
        self._mark_final_files()

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

    def bpe_encoding(self, bpe_operations):
        '''
        Further preprocessing for nematus backend by byte-pair encoding the given parallel corpora.

        @param bpe_operations "Create this many new symbols (each representing a character n-gram)"
                    Rico Sennrich, Barry Haddow and Alexandra Birch (2016). Neural Machine Translation of Rare Words with Subword Units.
                    Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
        '''

        # get input for learning: paths of truecased training corpus and (if any) evaluation corpus. no language ending
        corpus_train_tc=self._get_path('corpus') + os.sep + BASENAME_TRAINING_CORPUS + '.' + SUFFIX_TRUECASED
        if self._evaluation: 
            corpus_eval_tc=self._get_path('corpus') + os.sep + BASENAME_EVALUATION_CORPUS + '.' + SUFFIX_TRUECASED
        else:
            corpus_eval_tc=None

        # create target directory for bpe model
        bpe_model_path = os.sep.join([self._get_path('engine'), BPE])
        if not assertions.dir_exists(bpe_model_path):
            os.mkdir(bpe_model_path)

        # create encoder instance
        self._encoder = Encoder(corpus_train_tc, corpus_eval_tc, bpe_model_path, bpe_operations, self._src_lang, self._trg_lang, self._evaluation)

        # learn bpe model using truecased training corpus
        self._encoder.learn_bpe_model()

        # apply bpe model on truecased training corpus (and if present, truecased evaluation corpus)
        self._encoder.apply_bpe_model()

        # build bpe dictionary (JSON files) for truecased training corpus
        self._encoder.build_bpe_dictionary()

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

    def evaluate(self, num_threads, lowercase_eval=False, detokenize_eval=True,
                 strip_markup_eval=False, extended=False):
        '''
        Evaluates the engine by translating and scoring an evaluation set.
        @param num_threads the maximum number of threads to be used
        @param lowercase_eval whether to lowercase all segments before evaluation
        @param detokenize_eval whether to detokenize all segments before evaluation
        @param strip_markup_eval whether all markup should be removed before evaluation
        @param extended perform multiple evaluations that vary processing steps applied
            to the test files: lowercased or not, detokenized or not, with markup or without
        '''
        self._evaluator = evaluator.Evaluator(
            basepath=self._basepath,
            eval_corpus_path=self._get_path('corpus'),
            src_lang=self._src_lang,
            trg_lang=self._trg_lang,
            xml_strategy=self._xml_strategy,
            num_threads=num_threads,
            eval_tool=MULTEVAL_TOOL,
            tokenizer_trg=self._tokenizer_target,
            xml_processor=self._xml_processor,
            extended_eval=extended
        )
        self._evaluator.evaluate(
            lowercase=lowercase_eval,
            detokenize=detokenize_eval,
            strip_markup=strip_markup_eval
        )

    def _check_segment_length(self, segment, min_tokens, max_tokens, tokenizer,
            accurate=False):
        '''
        Checks the length of segments (in tokens). An accurate check strips,
            tokenizes the segment and XML element tags count as single tokens.
        @param segment the segment the length of which should be determined
        @param min_tokens minimal number of tokens
        @param max_tokens maximal number of tokens
        @param tokenizer the right tokenizer depending on the language
        @param whether the counting should be naive (faster) or accurate (slower)
        @return the input segment. None means that the segment should be
            discarded.
        '''
        original_segment = segment

        # more accurate count if preprocessing steps are applied
        if accurate:
            segment = segment.strip()
            segment = tokenizer.tokenize(segment, split=False)
            if self._masking_strategy:
                segment, _ = self._masker.mask_segment(segment)
            try:
                tokens = reinsertion.tokenize_keep_markup(segment)
            except:
                logging.debug("Segment is not well-formed: '%s'" % segment)
                # segment is not well-formed, discard
                self._num_malformed_segments += 1
                return None
        else:
            tokens = [token for token in segment.split(" ") if token != '']
        if len(tokens) < min_tokens or len(tokens) > max_tokens:
            # None means segment should be discarded
            return None

        return original_segment

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
            max_size=None,
            preprocess=True,
            tokenize=True,
            tokenizer_src=self._tokenizer_source,
            tokenizer_trg=self._tokenizer_target,
            mask=mask,
            masker=self._masker if self._masking_strategy else None,
            process_xml=process_xml,
            xml_processor=self._xml_processor if self._xml_strategy else None
        )
        corpus_tune = ParallelCorpus(
            self._get_path_corpus(BASENAME_TUNING_CORPUS, self._src_lang),
            self._get_path_corpus(BASENAME_TUNING_CORPUS, self._trg_lang),
            max_size=num_tune,
            preprocess=True,
            tokenize=True,
            tokenizer_src=self._tokenizer_source,
            tokenizer_trg=self._tokenizer_target,
            mask=mask,
            masker=self._masker if self._masking_strategy else None,
            process_xml=process_xml,
            xml_processor=self._xml_processor if self._xml_strategy else None
        )
        corpus_eval = ParallelCorpus(
            self._get_path_corpus(BASENAME_EVALUATION_CORPUS, self._src_lang),
            self._get_path_corpus(BASENAME_EVALUATION_CORPUS, self._trg_lang),
            max_size=num_eval,
            preprocess=False
        )
        # distribute segments from input corpus to output corpora
        for i, (segment_source, segment_target) in enumerate(zip(corpus_source, corpus_target)):
            # check lengths of segments
            segment_source = self._check_segment_length(segment_source, min_tokens, max_tokens,
                tokenizer=self._tokenizer_source, accurate=True)
            segment_target = self._check_segment_length(segment_target, min_tokens, max_tokens,
                tokenizer=self._tokenizer_target, accurate=True)
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
        # preprocess segments, then write to disk, the close file handle
        corpus_tune.close()
        corpus_eval.close()
        # logging
        logging.info("Training corpus: %s segments", corpus_train.get_size())
        logging.debug("Discarded %i segments because they were not well-formed" % self._num_malformed_segments)
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
                                    min_tokens, max_tokens, preprocess_external,
                                    process_xml):
        '''
        Pre-processes an external corpus into /corpus.

        @param basepath_external_corpus the basepath where the external corpus
            is located, without language code suffixes. Example: `/foo/bar/eval`
            will be expanded into `/foo/bar/eval.en` and `/foo/bar/eval.fr` in
            an EN to FR training
        @param basename the basename of the external corpus in the context of this
            training, e.g., `test`
        @param min_tokens minimal number of tokens in a segment
        @param max_tokens maximal number of tokens in a segment
        @param preprocess_external whether the segments should be preprocessed
        @param process_xml whether or not the XML processing strategy should be
            applied to the segments in external corpora
        '''
        corpus = ParallelCorpus(
            self._get_path_corpus(basename, self._src_lang),
            self._get_path_corpus(basename, self._trg_lang),
            max_size=None,
            preprocess=preprocess_external,
            tokenize=True, # todo: maybe do not hardcode this
            tokenizer_src=self._tokenizer_source,
            tokenizer_trg=self._tokenizer_target,
            mask=bool(self._masking_strategy),
            masker=self._masker if self._masking_strategy else None,
            process_xml=process_xml,
            xml_processor=self._xml_processor if self._xml_strategy else None
        )
        # pre-process segments
        corpus_source = open(basepath_external_corpus + "." + self._src_lang, 'r')
        corpus_target = open(basepath_external_corpus + "." + self._trg_lang, 'r')
        for segment_source, segment_target in zip(corpus_source, corpus_target):
            # tokenize, clean and mask segments (most importantly, remove trailing \n)
            segment_source = self._check_segment_length(segment_source, min_tokens, max_tokens,
                tokenizer=self._tokenizer_source, accurate=True)
            segment_target = self._check_segment_length(segment_target, min_tokens, max_tokens,
                tokenizer=self._tokenizer_target, accurate=True)
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
        Lowercases the training and tuning corpora.
        '''
        files_to_be_lowercased = []
        if self._casing_strategy == SELFCASING:
            files_to_be_lowercased.append(
                (BASENAME_TRAINING_CORPUS, self._src_lang)
            )
            if self._tuning:
                files_to_be_lowercased.append(
                    (BASENAME_TUNING_CORPUS, self._src_lang)
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
        elif self._casing_strategy == TRUECASING:
            # truecased input side, truecased output side
            add_suffix('train', 'src', SUFFIX_TRUECASED)
            add_suffix('train', 'trg', SUFFIX_TRUECASED)
            add_suffix('tune', 'src', SUFFIX_TRUECASED)
            add_suffix('tune', 'trg', SUFFIX_TRUECASED)
        elif self._casing_strategy == RECASING:
            # truecased input side, truecased output side
            add_suffix('train', 'src', SUFFIX_LOWERCASED)
            add_suffix('train', 'trg', SUFFIX_LOWERCASED)
            add_suffix('tune', 'src', SUFFIX_LOWERCASED)
            add_suffix('tune', 'trg', SUFFIX_LOWERCASED)
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
            '{script} -o {n} -S 30% -T "{path_temp_files}" < "{training_corpus}" > "{base_dir_lm}/{n}-grams.{trg_lang}.arpa"'.format(
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
        command_backward = '{threads_environment}{script} -i {corpus} -d -o -v -r > {corpus}.backward'.format(
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
        Tunes the system through Minimum Error Rate Trainign (MERT)
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
        commander.run(mert_command, "Tuning engine through Minimum Error Rate Training (MERT)")

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
