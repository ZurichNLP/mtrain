#!/usr/bin/env python3

import os
import random

from mtrain.preprocessing import cleaner

class ParallelCorpus(object):
    '''
    A parallel corpus storing either a limited or unlimited number of
    bi-segments.
    '''

    def __init__(self, filepath_source, filepath_target, max_size=None, 
        preprocess=False, tokenize=True, tokenizer_src=None, tokenizer_trg=None,
        mask=False, masker=None, process_xml=False, xml_processor=None):
        '''
        Creates an empty corpus stored at @param filepath_source (source side)
        and @param filepath_target (target side). Existing files will be
        overwritten. Preprocesses segments before writing them to disk.

        @param max_size the maximum number of segments to be stored. If given,
            the `self.insert` method will remove and return a random segment
            that is already stored in this corpus.
        @param preprocess whether segments in this corpus should be preprocessed
            before they are written to disk
        @param tokenize whether segments should be tokenized
        @param tokenizer_src tokenizer object for the source language
        @param tokenizer_trg tokenizer object for the target language
        @param mask whether segments should be masked
        @param masker masking.Masker object
        @param process_xml whether XML should be dealt with
        @param xml_processor an xmlprocessing.XmlProcessor object
        '''
        # set up preprocessing attributes
        self._preprocess = preprocess
        self._tokenize = tokenize
        self._tokenizer_src = tokenizer_src
        self._tokenizer_trg = tokenizer_trg
        self._mask = mask
        self._masker = masker
        self._process_xml = process_xml
        self._xml_processor = xml_processor

        # set up file paths and handles
        self._filepath_source = filepath_source
        self._filepath_target = filepath_target
        self._bisegments = []
        self._num_bisegments = 0
        self._max_size = max_size
        self._flush_immediately = False if self._max_size else True
        self._closed = False # closed corpora have been flushed to disk and can't be manipulated anymore.
        file_buffer = 1 if self._flush_immediately else -1 # 1: line buffered, -1: use system default
        self._file_source = open(self._filepath_source, 'w', file_buffer)
        self._file_target = open(self._filepath_target, 'w', file_buffer)

    def insert(self, segment_source, segment_target):
        '''
        Inserts a bi-segment into this corpus. If the maximum number of entries
        is exhausted, a random segment already contained in this corpus will
        be returned.
        '''
        assert self._closed == False, "Can't manipulate a closed corpus."
        bisegment = (segment_source, segment_target)
        self._num_bisegments += 1
        if self._flush_immediately:
            self._write_bisegment(bisegment)
        else:
            self._bisegments.append(bisegment)
            if self._num_bisegments > self._max_size:
                return self._pop_random_bisegment()

    def close(self):
        '''
        Writes all segments in this corpus to disk and closes the file handles.
        '''
        assert self._closed == False, "Can't manipulate a closed corpus."
        if not self._flush_immediately:
            for bisegment in self._bisegments:
                self._write_bisegment(bisegment)
        self._file_source.close()
        self._file_target.close()
        self._closed = True

    def delete(self):
        '''
        Deletes this corpus on disk.
        '''
        filepath_source, filepath_target = self.get_filepaths()
        os.remove(filepath_source)
        os.remove(filepath_target)

    def get_filepaths(self):
        '''
        Returns the filepaths for the source and target side of the parallel
        corpus as tuple.
        '''
        return (self._filepath_source, self._filepath_target)

    def get_size(self):
        '''
        Returns the number of bi-segments in this corpus.
        '''
        return self._num_bisegments

    def _preprocess_segment(self, segment, tokenizer):
        '''
        Tokenizes a bisegment, escapes special characters, introduces mask tokens or
            processes markup found in the segment.
        @param segment the segment that should be preprocessed
        @param the tokenizer object that should be used for tokenization
        '''
        segment = segment.strip()
        if self._tokenize:
            segment = tokenizer.tokenize(segment, split=False)
        if self._process_xml:
            segment, _ = self._xml_processor.preprocess_markup(segment)
        if self._mask:
            segment, _ = self._masker.mask_segment(segment)
        return cleaner.clean(segment)

    def _preprocess_bisegment(self, bisegment):
        '''
        Preprocesses a bisegment.
        '''
        segment_source, segment_target = bisegment
        segment_source = self._preprocess_segment(segment_source, tokenizer=self._tokenizer_src)
        segment_target = self._preprocess_segment(segment_target, tokenizer=self._tokenizer_trg)

        return segment_source, segment_target

    def _write_bisegment(self, bisegment):
        '''
        Writes a bisegment to file.
        @param bisegment the (source, target) segment tuple to be written to
            file.
        '''
        if self._preprocess:
            segment_source, segment_target = self._preprocess_bisegment(bisegment)
        else:
            segment_source, segment_target = bisegment
            segment_source = segment_source.strip()
            segment_target = segment_target.strip()

        self._file_source.write(segment_source + '\n')
        self._file_target.write(segment_target + '\n')

    def _pop_random_bisegment(self):
        '''
        Removes and returns a random bi-segment from this corpus.
        '''
        self._num_bisegments -= 1
        #todo: this is slow (O(n)) and needs improvement
        i = random.randrange(len(self._bisegments))
        return self._bisegments.pop(i)
