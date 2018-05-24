#!/usr/bin/env python3

from mtrain import constants as C
from mtrain import commander
from mtrain.preprocessing.external import ExternalProcessor

"""
Provides further processing and postprocessing steps in order to use neural network training and translation in backend nematus.
"""

class BytePairEncoderFile(object):
    '''
    Further preprocessing for nematus backend by byte-pair encoding the given parallel corpora.

    The encoding is limited to input that is already processed using 'truecased' casing strategy.

    Cf. https://gitlab.cl.uzh.ch/mt/mtrain/blob/nematus/README.md for list of references.
    '''
    def __init__(self, corpus_train_tc, corpus_tune_tc, bpe_model_path, bpe_operations, src_lang, trg_lang):
        '''
        @params corpus_train_tc location of truecased training corpus (no language ending)
        @params corpus_tune_tc location of truecased tuning corpus (no language ending)
        @params bpe_model_path path of byte-pair encoding model to be stored (no filename)
        @params bpe_operations number of n-grams to be created
        @params src_lang language identifier of source language
        @params trg_lang language identifier of target language
        '''
        self._corpus_train_tc=corpus_train_tc
        self._corpus_tune_tc=corpus_tune_tc
        self._bpe_model_path=bpe_model_path
        self._bpe_operations=bpe_operations
        self._src_lang=src_lang
        self._trg_lang=trg_lang

    def learn_bpe_model(self):
        '''
        Learn bpe model using truecased training corpus.
        Stores the bpe model in the basepath's subfolder 'engine/bpe/', model name SRC-TRG.bpe.

        Script reference https://github.com/rsennrich/subword-nmt/blob/master/learn_bpe.py:
            Rico Sennrich, Barry Haddow, and Alexandra Birch (2016): Neural Machine Translation of Rare Words with Subword Units.
            In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.

        Cf. https://gitlab.cl.uzh.ch/mt/mtrain/blob/nematus/README.md for list of references.
        '''
        commander.run(
            'cat {corpus}.{src} {corpus}.{trg} | {script} --s {bpe_ops} > {bpe_model}/{src}-{trg}.bpe'.format(
                corpus=self._corpus_train_tc,
                script=SUBWORD_NMT_LEARN,
                bpe_ops=self._bpe_operations,
                bpe_model=self._bpe_model_path,
                src=self._src_lang,
                trg=self._trg_lang
            ),
            "Learning BPE model: %s operations" % self._bpe_operations
        )

    def apply_bpe_model(self):
        '''
        Apply bpe model on truecased training corpus and truecased tuning corpus. Creates the files in the
        basepath's subfolder 'corpus', file names train.truecased.bpe.SRC|TRG and tune.truecased.bpe.SRC|TRG.

        Script reference https://github.com/rsennrich/subword-nmt/blob/master/apply_bpe.py:
            Rico Sennrich, Barry Haddow, and Alexandra Birch (2016): Neural Machine Translation of Rare Words with Subword Units.
            In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.

        Cf. https://gitlab.cl.uzh.ch/mt/mtrain/blob/nematus/README.md for list of references.
        '''
        def command(current_corpus, current_lang):
            return '{script} -c {bpe_model}/{src}-{trg}.bpe < {corpus}.{lang} > {corpus}.bpe.{lang}'.format(
            script=SUBWORD_NMT_APPLY,
            bpe_model=self._bpe_model_path,
            src=self._src_lang,
            trg=self._trg_lang,
            corpus=current_corpus,  # either truecased training or tuning corpus, depending on commander
            lang=current_lang       # either source or target language, depending on commander
            )
        commands = [
            command(self._corpus_train_tc, self._src_lang),
            command(self._corpus_train_tc, self._trg_lang),
            command(self._corpus_tune_tc, self._src_lang),
            command(self._corpus_tune_tc, self._trg_lang),
        ]
        commander.run_parallel(commands, "Applying BPE model")

    def build_bpe_dictionary(self):
        '''
        Build bpe dictionary (JSON files) for truecased training corpus.
        Note that the JSON files such as train.truecased.bpe.SRC.json and train.truecased.bpe.TRG.json
        are automatically stored at the location of the input files, which is the basepath's subfolder 'corpus'.

        Script reference https://github.com/EdinburghNLP/nematus/blob/master/data/build_dictionary.py:
            Rico Sennrich, Orhan Firat, Kyunghyun Cho, Alexandra Birch, Barry Haddow, Julian Hitschler, Marcin Junczys-Dowmunt, Samuel Läubli,
            Antonio Valerio Miceli Barone, Jozef Mokry, and Maria Nadejde (2017): Nematus: a Toolkit for Neural Machine Translation.
            In Proceedings of the Software Demonstrations of the 15th Conference of the European Chapter of the Association for Computational
            Linguistics (EACL 2017). Valencia, Spain, pp. 65-68.

        Cf. https://gitlab.cl.uzh.ch/mt/mtrain/blob/nematus/README.md for list of references.
        '''
        commander.run(
            '{script} {corpus}.bpe.{src} {corpus}.bpe.{trg}'.format(
                script=NEMATUS_BUILD_DICT,
                corpus=self._corpus_train_tc,
                src=self._src_lang,
                trg=self._trg_lang,
            ),
            "Building network dictionary from BPE model"
        )

class BytePairEncoderSegment(object):
    '''
    Creates a byte-pair encoder which encodes normalized, tokenized and truecased segments
    in order to enable translation in backend nematus.

    Cf. https://gitlab.cl.uzh.ch/mt/mtrain/blob/nematus/README.md for list of references.
    '''
    def __init__(self, bpe_model):
        '''
        @param bpe_model full path to byte-pair processing model trained in `mtrain`

        Script reference https://github.com/rsennrich/subword-nmt/blob/master/apply_bpe.py:
            Rico Sennrich, Barry Haddow, and Alexandra Birch (2016): Neural Machine Translation of Rare Words with Subword Units.
            In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.

        Cf. https://gitlab.cl.uzh.ch/mt/mtrain/blob/nematus/README.md for list of references.
        '''
        arguments = [
            '-c %s' % bpe_model
        ]
        # the subword script apply_bpe.py needs to be run in python 3 environment and thus,
        # respective constant used to make this explicit (workaround for environments that do not
        # clearly specify paths to python versions)
        self._processor = ExternalProcessor(
            command=" ".join([PYTHON3] + [SUBWORD_NMT_APPLY] + arguments),
            stream_stderr=False, # just used as positional argument
            trailing_output=False, # just used as positional argument
            shell=False # todo: call as python process (instead of shell subprocess), not enabled in external.py yet
        )

    def close(self):
        del self._processor

    def bpencode_segment(self, segment):
        '''
        Encodes a single @param segment by applying the byte-pair
        processing model trained in `mtrain`.
        '''
        encoded_segment = self._processor.process(segment)
        return encoded_segment

class BytePairDecoderSegment(object):
    '''
    Creates a byte-pair decoder which decodes a translated segment in backend nematus.

    No need for byte-pair processing model as only the remaining byte-pair markers are
    removed that could not be translated from source to target language. These unknown byte-pairs
    in the input to be translated are due to absence in the training corpus and thus, in the
    byte-pair model.

    Cf. https://gitlab.cl.uzh.ch/mt/mtrain/blob/nematus/README.md for list of references.
    '''

    def bpdecode_segment(self, segment):
        '''
        Decodes a single @param segment.

        Code directly applied from example shellscript https://github.com/rsennrich/wmt16-scripts/blob/master/sample/postprocess-test.sh:
            Rico Sennrich, Barry Haddow, and Alexandra Birch (2016): Edinburgh Neural Machine Translation Systems for WMT 16.
            In Proceedings of the First Conference on Machine Translation (WMT16). Berlin, Germany.

        Cf. https://gitlab.cl.uzh.ch/mt/mtrain/blob/nematus/README.md for list of references.
        '''
        decoded_segment = segment.replace("@@ ","")
        return decoded_segment
