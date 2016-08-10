#!/usr/bin/env python3

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

    def get_filepaths(self):
        '''
        Returns the filepaths for the source and target side of the parallel
        corpus as tuple.
        '''
        return (self._filepath_source, self._filepath_target)

    def _write_bisegment(self, bisegment):
        '''
        Writes a bi-segment to file.

        @param bi-segment the (source, target) segment tuple to be written to
            file.
        '''
        segment_source, segment_target = bisegment
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
