import re, string, random, glob, operator, heapq, codecs, sys, optparse, os, logging, math
from functools import reduce
from collections import defaultdict
from math import log10


def takeProb(entry):
    return entry[2]

class Segment:

    def __init__(self, Pw):
        self.Pw = Pw

    def segment(self, text):
        "Return a list of words that is the best segmentation of text."
        if not text: return []

        ## Initialize the heap ##  
        heap = []
        splitted_text = self.splits(text)

        for w,rem in splitted_text:
            #  Entry(word, start-position, log-probability, back-pointer)
            entry = (w, 0, math.log10(self.Pw(w)), None)
            heap.append(entry)

        heap.sort(key=takeProb)

        # # initialize chart: the dynamic programming table to store the argmax for every prefix of input
        # # indexed by character position in input
        chart = [None for i in range(len(text))]

        ## Iteratively fill in chart[i] for all i ##
        while heap:
            entry = heap.pop()
            entry_word = entry[0]
            entry_start = entry[1]
            entry_prob = entry[2]
            entry_backptr = entry[3]

            # get the endindex based on the length of the word in entry
            endindex = entry_start + len(entry_word)  - 1 

            if chart[endindex]:
                preventry = chart[endindex]
                preventry_word = preventry[0]
                preventry_start = preventry[1]
                preventry_prob = preventry[2]
                preventry_backptr = preventry[3]

                if entry_prob > preventry_prob:
                    # if entry has a higher probability than preventry
                    chart[endindex] = entry
                if entry_prob <= preventry_prob:
                     ## we have already found a good segmentation until endindex ##
                    continue
            else:
                chart[endindex] = entry
            
            splitted_text = self.splits(text[(endindex + 1):])

            for newword,rem in splitted_text:
                newentry = (newword, endindex + 1, entry_prob + math.log10(self.Pw(newword)), entry)
                if newentry not in heap:
                    heap.append(newentry)

            heap.sort(key=takeProb)

        # ## Get the best segmentation ##
        finalindex = len(text)
        finalentry = chart[finalindex - 1]
        segmentation = self.recursive_back(finalentry)

        return segmentation

    def recursive_back(self,entry):
        if entry:
            return self.recursive_back(entry[3]) + entry[0] + ' '
        else:       
            return ''

    def splits(self, text, L = 5):
        "Return a list of all possible (first, rem) pairs, len(first)<=L."
        return [(text[:i+1], text[i+1:]) 
                for i in range(min(len(text), L))]    

    def Pwords(self, words): 
        "The Naive Bayes probability of a sequence of words."
        return product(self.Pw(w) for w in words)


# bigram segment class
class Segment2:
    def __init__(self, Pw, P2w):
        self.Pw = Pw
        self.P2w = P2w
        self.table = {}

    def cPw(self, prev, word):
        "Return the conditional probability P(word | previous-word)."
        try:
            ml1 = float(Pw(word))
            ml2 = float(self.P2w[prev + ' ' + word])/float(self.Pw[prev])
            x = 0.9
            return x * ml2 + (1 - x) * ml1
        except KeyError:
            return self.Pw(word)

    def segment(self, text, prev = '<S>'):
        if (text, prev) in self.table:
            return self.table[(text, prev)]

        if not text: return 0.0, []

        segmentation = []
        
        splitted_text = self.splits(text)

        for first, rem in splitted_text:
            # recursively call segment
            restProb, rest = self.segment(rem,first)

            prob = log10(self.cPw(prev,first))
            segmentation.append((prob + restProb, [first] + rest))

        finalentry = segmentation[0]
        for index in range(len(segmentation)):
            if segmentation[index][0] > finalentry[0]:
                finalentry = segmentation[index]

        self.table[text, prev] = finalentry

        return finalentry


    def splits(self, text, L = 5):
        "Return a list of all possible (first, rem) pairs, len(first)<=L."
        return [(text[:i+1], text[i+1:]) 
                for i in range(min(len(text), L))]  


#### Support functions (p. 224)

def print_bigram(text):
    result = ""
    for w in text:
        result = result + w + " "
    return result
    

def product(nums):
    "Return the product of a sequence of numbers."
    return reduce(operator.mul, nums, 1)

def smooth(word, N):
    return 10000./(N * 1000**len(word))

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None, missingfn=None):
        for key,count in data:
            self[key] = self.get(key, 0) + int(count)
        self.N = float(N or sum(self.values()))
        self.missingfn = missingfn or (lambda k, N: 1./N)
    def __call__(self, key): 
        if key in self: return self[key]/self.N  
        else: return self.missingfn(key, self.N)


def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    with open(name) as fh:
        for line in fh:
            (key, value) = line.split(sep)
            yield (key, value)

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts [default: data/count_1w.txt]")
    optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts [default: data/count_2w.txt]")
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    Pw = Pdist(data=datafile(opts.counts1w),N = None, missingfn = smooth)
    P2w = Pdist(data=datafile(opts.counts2w))

    segmenter1 = Segment(Pw)
    segmenter2 = Segment2(Pw,P2w)
    with open(opts.input) as f:
        for line in f:
            #recursive bigram
            print(print_bigram(segmenter2.segment(line.strip())[1]))

            #unigram iterative segmenter
            # print(segmenter1.segment(line.strip()))





