import os, sys, optparse
import os.path
import tqdm
import pymagnitude
import subprocess


class LexSub:

    def __init__(self, wvec_file, topn=10):
        self.wvecs = pymagnitude.Magnitude(wvec_file)
        self.topn = topn

    def substitutes(self, index, sentence):
        "Return ten guesses that are appropriate lexical substitutions for the word at sentence[index]."
        return(list(map(lambda k: k[0], self.wvecs.most_similar(sentence[index], topn=self.topn))))

    def substitutes2(self, index, sentence):
        substitutability = {}
        target = sentence[index]
        sentential_context = sentence
        guesses = []
        neighbours = list(map(lambda k: k[0], self.wvecs.most_similar(target, topn = 15)))

        # s : lexical substitude (target word's neighbours)
        # t : target substitude word
        # c : words in sentential context 

        for s in neighbours:
            cos_s_t = self.wvecs.similarity(s, target)
            cos_s_c = 0
            for c in sentential_context:
                if c in self.wvecs:
                    cos_s_c += self.wvecs.similarity(s, c)
            substitutability[s] = ( cos_s_t + cos_s_c ) / (1 + len(sentential_context))


        for i in range(10):
            guesses.append(min(substitutability, key=substitutability.get))
            del substitutability[min(substitutability, key=substitutability.get)]


        return guesses  

    def retrofit(self, lexicon, T = 10):
        # initialize retrofit word vector equal to origin word vector
        newwv = {}
        for key, vector in self.wvecs:
            newwv[key] = vector

        loopVocab = []
        for key in lexicon:
            if key in self.wvecs:
                loopVocab.append(key)

        for i in range(T):
            for word in loopVocab:

                wordneighbours = []
                for neighbours in lexicon[word]:
                    if neighbours in self.wvecs:
                        wordneighbours.append(neighbours)
                numneighbours = len(wordneighbours)
                # pass if no neighbours (use original dataset)
                if numneighbours == 0:
                    continue
                # the weight(sum of bij and aij) of the data estimate is the number of neighbours
                newVec= numneighbours * self.wvecs.query(word)

                for qjword in wordneighbours:
                    newVec += newwv[qjword]

                newwv[word] = newVec / (2 * numneighbours)

        return newwv

def norm_word(word):
  if word.lower().isdigit():
    return '-number-'
  elif word.lower() == '':
    return '-punctuation-'
  else:
    return word.lower()

def read_lex(lex_file):
    lexicon = {}
    for line in open(lex_file, 'r'):
        words = line.lower().strip().split()
        lexicon[norm_word(words[0])] = [norm_word(word) for word in words[1:]]
    return lexicon

def write_wvtxt(wordVec, outFileName):
    sys.stderr.write('\nWriting retrofitted word vectors in '+outFileName+'\n')
    output = open(outFileName, 'w')  
    for word, values in wordVec.items():
        output.write(word+' ')
        for element in wordVec[word]:
            output.write('%.5f' %(element)+' ')
        output.write('\n')      
    output.close()


if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input', 'dev.txt'), help="file to segment")
    optparser.add_option("-w", "--wordvecfile", dest="wordvecfile", default=os.path.join('data', 'glove.6B.100d.magnitude'), help="word vectors file")
    optparser.add_option("-x", "--lexiconsfile", dest="lexicons", default=os.path.join('data', 'lexicons', 'wordnet-synonyms+.txt'), help="lexicons file")
    optparser.add_option("-n", "--topn", dest="topn", default=10, help="produce these many guesses")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    # generate retrofit word vector if it does not exit
    if not os.path.exists('data/glove.6B.100d.retrofit.magnitude'):
        lexsub = LexSub(opts.wordvecfile, int(opts.topn))
        lexicon = read_lex(opts.lexicons)
        sys.stderr.write('\nStart producing retrofitted word vectors\n')
        newWordVec = lexsub.retrofit(lexicon)
        write_wvtxt(newWordVec, "data/glove.6B.100d.retrofit.txt")
        sys.stderr.write('\nConvert txt retrofitted word vector file into pymagnitude format \n')
        subprocess.call('python3 -m pymagnitude.converter -i data/glove.6B.100d.retrofit.txt -o data/glove.6B.100d.retrofit.magnitude', shell = True)

    sys.stderr.write('\nUse retrofitted word vector to do lexsub\n')
    lexsub2 = LexSub('data/glove.6B.100d.retrofit.magnitude', int(opts.topn))

    num_lines = sum(1 for line in open(opts.input,'r'))
    with open(opts.input) as f:
        for line in tqdm.tqdm(f, total=num_lines):
            fields = line.strip().split('\t')
            print(" ".join(lexsub2.substitutes(int(fields[0].strip()), fields[1].strip().split())))
