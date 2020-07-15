import argparse
import re
import string
import time
import torch
import torch.nn as nn

import data
import model

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--emsize', type=int, default=400,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1150,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=30,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=8000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=80, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=float, default=0.3,
                    help='dropout for rnn layers (0 = no dropout)')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
randomhash = ''.join(str(time.time()).split('.'))
parser.add_argument('--save', type=str,  default=randomhash+'.pt',
                    help='path to save the final model')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--resume', type=str,  default='',
                    help='path of model to resume')
parser.add_argument('--optimizer', type=str,  default='sgd',
                    help='optimizer to use (sgd, adam)')
parser.add_argument('--when', nargs="+", type=int, default=[-1],
                    help='When (which epochs) to divide the learning rate by 10 - accepts multiple')
args = parser.parse_args()
args.tied = True

import os
import hashlib
fn = 'corpus.{}.data'.format(hashlib.md5(args.data.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = data.Corpus(args.data)
    torch.save(corpus, fn)

ntokens = len(corpus.dictionary)
model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)

if args.cuda:
    model = model.cuda(3)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

def new_tokenize(text):
    words = text.split()
    ids = torch.LongTensor(len(words))
    token = 0
    for word in words:
        if word in corpus.dictionary.word2idx:
            ids[token] = corpus.dictionary.word2idx[word]
        else:
            ids[token] = corpus.dictionary.word2idx['<unk>']
        token += 1
    return ids

def play(text, batch_size=1):
    model.eval()
    text = text.lower()
    text = re.sub('\d+', 'N', text)
    punc = string.punctuation.replace(".", "’—“”")
    punc = punc.replace("'", "")
    text = text.translate(str.maketrans('', '', punc))
    text = text.replace("n't", " n't")
    text = text.replace("'s", " 's")
    text = text.replace("'ve", " 've")
    text = text.replace("'d", " 'd")
    text = text.replace("'ll", " 'll")
    data = new_tokenize(text).unsqueeze(1).cuda()
    hidden = model.init_hidden(batch_size)
    output, hidden = model(data, hidden)
    logits = model.decoder(output)
    logProba = nn.functional.log_softmax(logits, dim=1)
    unk_idx = corpus.dictionary.word2idx['<unk>']
    mini = torch.min(logProba)
    logProba[:,unk_idx] = mini
    pred_idxs = torch.argmax(logProba, dim=1)
    preds = [corpus.dictionary.idx2word[idx] for idx in pred_idxs]
    next_word = preds[-1]
    return next_word

# Load the best saved model.
model_load(args.save)

while True:
    text = input("Hey, enter part of a sentence here: ")
    next_word = play(text)
    for i in range(70):
        text = text + " " + next_word
        next_word = play(text)
    print("Here's what we got:\n:", text)
    again = input("Press enter to play again! ")
    if again != "":
        break