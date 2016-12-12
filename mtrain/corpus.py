#!/usr/bin/env python3

import os
import random

class ParallelCorpus(object):
    '''
    A parallel corpus storing either a limited or unlimited number of
    bi-segments.
    '''

    def __init__(self, filepath_source, filepath_target, max_size=None):
        '''
        Creates an empty corpus stored at @param filepath_source (source side)
        and @param filepath_target (target side). Existing files will be
        overwritten.

        @param max_size the maximum number of segments to be stored. If given,
            the `self.insert` method will remove and return a random segment
            that is already stored in this corpus.
        '''
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

    def insert(self, segment_source, segment_target, tokenize=True,
        tokenizer_src=None, tokenizer_trg=None, mask=False, masker=None,
        process_xml=False, xml_processor=None):
        '''
        Inserts a bi-segment into this corpus. If the maximum number of entries
        is exhausted, a random segment already contained in this corpus will
        be returned.
        '''
        assert self._closed == False, "Can't manipulate a closed corpus."
        bisegment = (segment_source, segment_target)
        self._num_bisegments += 1
        if self._flush_immediately:
            self._write_bisegment(
                bisegment,
                tokenize=tokenize,
                tokenizer_src=tokenizer_src,
                tokenizer_trg=tokenizer_trg,
                mask=mask, masker=masker,
                process_xml=process_xml,
                xml_processor=xml_processor
            )
        else:
            self._bisegments.append(bisegment)
            if self._num_bisegments > self._max_size:
                return self._pop_random_bisegment()

    def close(self, tokenize=True, tokenizer_src=None, tokenizer_trg=None,
        mask=False, masker=None, process_xml=False, xml_processor=None):
        '''
        Writes all segments in this corpus to disk and closes the file handles.
        '''
        assert self._closed == False, "Can't manipulate a closed corpus."
        if not self._flush_immediately:
            for bisegment in self._bisegments:
                self._write_bisegment(
                    bisegment, tokenize=True, tokenizer_src=tokenizer_src,
                    tokenizer_trg=tokenizer_trg, mask=False,
                    masker=None, process_xml=False, xml_processor=None
                )
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

    def _preprocess_segment(self, segment, tokenize=True, tokenizer=None, mask=False,
        masker=None, process_xml=False, xml_processor=None):
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
            segment, _ = xml_processor.preprocess_markup(segment)
        if mask:
            segment, _ = masker.mask_segment(segment)
        # check length of segment after masking and xml processing, otherwise
        # the counts will not be meaningful
        tokens = [token for token in segment.split(" ") if token != '']
        if len(tokens) < min_tokens or len(tokens) > max_tokens:
            return None # means segment should be discarded
        segment = cleaner.clean(segment)

        return segment

    def _preprocess_bisegment(self, bisegment, tokenize=True, tokenizer_src=None,
        tokenizer_trg=None, mask=False, masker=None, process_xml=False, xml_processor=None):
        '''
        Preprocesses a bisegment.
        '''
        segment_source, segment_target = bisegment
        segment_source = self._preprocess_segment(segment_source, tokenize=tokenize,
            tokenizer=tokenizer_src, mask=mask, masker=masker, process_xml=process_xml)
        segment_target = self._preprocess_segment(segment_target, tokenize=tokenize,
            tokenizer=tokenizer_trg, mask=mask, masker=masker, process_xml=process_xml)

        return segment_source, segment_target

    def _write_bisegment(self, bisegment, tokenize=True, tokenizer_src=None,
        tokenizer_trg=None, mask=False, masker=None, process_xml=False,
        xml_processor=None):
        '''
        Writes a bi-segment to file.

        @param bisegment the (source, target) segment tuple to be written to
            file.
        @param preprocess whether the bisegment should be preprocessed before it
            is written to file
        '''
        segment_source, segment_target = self._preprocess_bisegment(
            bisegment, tokenize=True, tokenizer_src=None,
            tokenizer_trg=None, mask=False, masker=None,
            process_xml=False, xml_processor=None
            )

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
