# Code adapted from original code by Robert Guthrie

import os, sys, optparse, gzip, re, logging
import string  
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

def read_conll(handle, input_idx=0, label_idx=2):
    conll_data = []
    contents = re.sub(r'\n\s*\n', r'\n\n', handle.read())
    contents = contents.rstrip()
    for sent_string in contents.split('\n\n'):
        annotations = list(zip(*[ word_string.split() for word_string in sent_string.split('\n') ]))
        assert(input_idx < len(annotations))
        if label_idx < 0:
            conll_data.append( annotations[input_idx] )
            logging.info("CoNLL: {}".format( " ".join(annotations[input_idx])))
        else:
            assert(label_idx < len(annotations))
            conll_data.append(( annotations[input_idx], annotations[label_idx] ))
            logging.info("CoNLL: {} ||| {}".format( " ".join(annotations[input_idx]), " ".join(annotations[label_idx])))
    return conll_data

def prepare_sequence(seq, to_ix, unk):
    idxs = []
    if unk not in to_ix:
        idxs = [to_ix[w] for w in seq]
    else:
        idxs = [to_ix[w] for w in map(lambda w: unk if w not in to_ix else w, seq)]
    return torch.tensor(idxs, dtype=torch.long)

def prepare_initial_char_seq(seq, dic):
    idxs = []
    for w in seq:
        temp = torch.zeros([len(string.printable)], dtype=torch.float)
        temp[dic[w[0]]] = 1
        idxs.append(temp)
    result = torch.stack(idxs,0)
    return result


def prepare_last_char_seq(seq, dic):
    idxs = []
    for w in seq:
        temp = torch.zeros([len(string.printable)], dtype=torch.float)
        temp[dic[w[-1]]] = 1
        idxs.append(temp)
    result = torch.stack(idxs,0)
    return result

def prepare_mid_char_seq(seq, dic):
    idxs = []
    for w in seq:
        if len(w) >2:
            for char in w[1:-1]:
                temp = torch.zeros([len(string.printable)], dtype=torch.float)
                temp[dic[char]] += 1
            idxs.append(temp)
        else:
            idxs.append(torch.zeros([len(string.printable)], dtype=torch.float))
    result = torch.stack(idxs,0)
    return result




class LSTMTaggerModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        torch.manual_seed(1)
        super(LSTMTaggerModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=False)
        self.lstm2 = nn.LSTM(428, hidden_dim, bidirectional=False)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence, char):
        embeds = self.word_embeddings(sentence)
        embeds_t = embeds.t()
        char_t = char.t()
        embeds = torch.cat((embeds_t,char_t),0)
        embeds = embeds.t()

        lstm_out, _ = self.lstm2(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

class LSTMTagger:

    def __init__(self, trainfile, modelfile, modelsuffix, unk="[UNK]", epochs=10, embedding_dim=128, hidden_dim=64):
        self.unk = unk
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.modelfile = modelfile
        self.modelsuffix = modelsuffix
        self.training_data = []
        if trainfile[-3:] == '.gz':
            with gzip.open(trainfile, 'rt') as f:
                self.training_data = read_conll(f)
        else:
            with open(trainfile, 'r') as f:
                self.training_data = read_conll(f)

        self.word_to_ix = {} # replaces words with an index (one-hot vector)
        self.tag_to_ix = {} # replace output labels / tags with an index
        self.ix_to_tag = [] # during inference we produce tag indices so we have to map it back to a tag

        printable = string.printable
        self.diction = {}
        count = 0
        for char in printable:
            self.diction[char] = count
            count = count + 1

        for sent, tags in self.training_data:
            for word in sent:
                if word not in self.word_to_ix:
                    self.word_to_ix[word] = len(self.word_to_ix)
            for tag in tags:
                if tag not in self.tag_to_ix:
                    self.tag_to_ix[tag] = len(self.tag_to_ix)
                    self.ix_to_tag.append(tag)

        logging.info("word_to_ix:", self.word_to_ix)
        logging.info("tag_to_ix:", self.tag_to_ix)
        logging.info("ix_to_tag:", self.ix_to_tag)

        self.model = LSTMTaggerModel(self.embedding_dim, self.hidden_dim, len(self.word_to_ix), len(self.tag_to_ix))
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def argmax(self, seq):
        output = []
        with torch.no_grad():
            inputs = prepare_sequence(seq, self.word_to_ix, self.unk)

            v1 = prepare_initial_char_seq(seq, self.diction)
            v2 = prepare_mid_char_seq(seq, self.diction)
            v3 = prepare_last_char_seq(seq, self.diction)
            v1 = v1.t()
            v2 = v2.t()
            v3 = v3.t()
            char_in = torch.cat((v1,v2,v3),0)
            char_in = char_in.t()

            tag_scores = self.model(inputs,char_in)
            for i in range(len(inputs)):
                output.append(self.ix_to_tag[int(tag_scores[i].argmax(dim=0))])
        return output

    def train(self):
        loss_function = nn.NLLLoss()

        self.model.train()
        loss = float("inf")
        for epoch in range(self.epochs):
            for sentence, tags in tqdm.tqdm(self.training_data):
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                self.model.zero_grad()

                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                sentence_in = prepare_sequence(sentence, self.word_to_ix, self.unk)
                targets = prepare_sequence(tags, self.tag_to_ix, self.unk)

                v1 = prepare_initial_char_seq(sentence, self.diction)
                v2 = prepare_mid_char_seq(sentence, self.diction)
                v3 = prepare_last_char_seq(sentence, self.diction)
                v1 = v1.t()
                v2 = v2.t()
                v3 = v3.t()

                char_in = torch.cat((v1,v2,v3),0)
                char_in = char_in.t()

                # Step 3. Run our forward pass.
                tag_scores = self.model(sentence_in,char_in)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = loss_function(tag_scores, targets)
                loss.backward()
                self.optimizer.step()

            if epoch == self.epochs-1:
                epoch_str = '' # last epoch so do not use epoch number in model filename
            else:
                epoch_str = str(epoch)
            savefile = self.modelfile + epoch_str + self.modelsuffix
            print("saving model file: {}".format(savefile), file=sys.stderr)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                        'unk': self.unk,
                        'word_to_ix': self.word_to_ix,
                        'tag_to_ix': self.tag_to_ix,
                        'ix_to_tag': self.ix_to_tag,
                    }, savefile)

    def decode(self, inputfile):
        if inputfile[-3:] == '.gz':
            with gzip.open(inputfile, 'rt') as f:
                input_data = read_conll(f, input_idx=0, label_idx=-1)
        else:
            with open(inputfile, 'r') as f:
                input_data = read_conll(f, input_idx=0, label_idx=-1)

        if not os.path.isfile(self.modelfile + self.modelsuffix):
            raise IOError("Error: missing model file {}".format(self.modelfile + self.modelsuffix))

        saved_model = torch.load(self.modelfile + self.modelsuffix)
        self.model.load_state_dict(saved_model['model_state_dict'])
        self.optimizer.load_state_dict(saved_model['optimizer_state_dict'])
        epoch = saved_model['epoch']
        loss = saved_model['loss']
        self.unk = saved_model['unk']
        self.word_to_ix = saved_model['word_to_ix']
        self.tag_to_ix = saved_model['tag_to_ix']
        self.ix_to_tag = saved_model['ix_to_tag']
        self.model.eval()
        decoder_output = []
        for sent in tqdm.tqdm(input_data):
            decoder_output.append(self.argmax(sent))
        return decoder_output

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--inputfile", dest="inputfile", default=os.path.join('data', 'input', 'dev.txt'), help="produce chunking output for this input file")
    optparser.add_option("-t", "--trainfile", dest="trainfile", default=os.path.join('data', 'train.txt.gz'), help="training data for chunker")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join('data', 'chunker'), help="filename without suffix for model files")
    optparser.add_option("-s", "--modelsuffix", dest="modelsuffix", default='.tar', help="filename suffix for model files")
    optparser.add_option("-e", "--epochs", dest="epochs", default=5, help="number of epochs [fix at 5]")
    optparser.add_option("-u", "--unknowntoken", dest="unk", default='[UNK]', help="unknown word token")
    optparser.add_option("-f", "--force", dest="force", action="store_true", default=False, help="force training phase (warning: can be slow)")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="log file for debugging")
    (opts, _) = optparser.parse_args()

    if opts.logfile is not None:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.DEBUG)

    modelfile = opts.modelfile
    if opts.modelfile[-4:] == '.tar':
        modelfile = opts.modelfile[:-4]

    chunker = LSTMTagger(opts.trainfile, modelfile, opts.modelsuffix, opts.unk)

    # use the model file if available and opts.force is False
    if os.path.isfile(opts.modelfile + opts.modelsuffix) and not opts.force:
        decoder_output = chunker.decode(opts.inputfile)
    else:
        print("Warning: could not find modelfile {}. Starting training.".format(modelfile + opts.modelsuffix), file=sys.stderr)
        chunker.train()
        decoder_output = chunker.decode(opts.inputfile)

    print("\n\n".join([ "\n".join(output) for output in decoder_output ]))
