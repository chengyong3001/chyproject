#!/usr/bin/env python

import argparse
import cPickle
import logging
import pprint

import numpy

from groundhog.trainer.SGD_adadelta import SGD as SGD_adadelta
from groundhog.trainer.SGD import SGD as SGD
from groundhog.trainer.SGD_momentum import SGD as SGD_momentum
from groundhog.mainLoop import MainLoop
from experiments.nmt import RNNEncoderDecoder
from experiments.nmt import get_batch_iterator
from experiments.nmt import BiEncoderDecoder
from experiments.nmt import prototype_state, get_batch_iterator
import experiments.nmt

logger = logging.getLogger(__name__)

class RandomSamplePrinter(object):

    def __init__(self, state, model, train_iter):
        args = dict(locals())
        args.pop('self')
        self.__dict__.update(**args)

    def __call__(self):
        def cut_eol(words):
            for i, word in enumerate(words):
                if words[i] == '<eol>':
                    return words[:i + 1]
            raise Exception("No end-of-line found")

        sample_idx = 0
        while sample_idx < self.state['n_examples']:
            batch = self.train_iter.next(peek=True)
            xs, ys, xs_1, ys_1 = batch['x'], batch['y'], batch['x_1'], batch['y_1']
            for seq_idx in range(xs.shape[1]):
                if sample_idx == self.state['n_examples']:
                    break

                x, y = xs[:, seq_idx], ys[:, seq_idx]
                x_1, y_1 = xs_1[:, seq_idx], ys_1[:, seq_idx]

                x_words = cut_eol(map(lambda w_idx : self.model.word_indxs_src[w_idx], x))
                y_words = cut_eol(map(lambda w_idx : self.model.word_indxs[w_idx], y))


                x_words_1 = cut_eol(map(lambda w_idx : self.model.word_indxs_src_1[w_idx], x_1))
                y_words_1 = cut_eol(map(lambda w_idx : self.model.word_indxs_1[w_idx], y_1))
                
                if len(x_words) == 0:
                    continue

                print "Input: {}".format(" ".join(x_words))
                print "Target: {}".format(" ".join(y_words))


                print "Input: {}".format(" ".join(x_words_1))
                print "Target: {}".format(" ".join(y_words_1))
                #this n_samples not number of samples but temperature.
                self.model.get_samples_fwd(self.state['seqlen'] + 1, self.state['n_samples'], x[:len(x_words)])
                self.model.get_samples_inv(self.state['seqlen'] + 1, self.state['n_samples'], x_1[:len(y_words)])
                sample_idx += 1

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", help="State to use")
    parser.add_argument("--proto",  default="prototype_state",
        help="Prototype state to use for state")
    parser.add_argument("--skip-init", action="store_true",
        help="Skip parameter initilization")
    parser.add_argument("changes",  nargs="*", help="Changes to state", default="")
    return parser.parse_args()

def main():
    args = parse_args()

    state = getattr(experiments.nmt, args.proto)()
    if args.state:
        if args.state.endswith(".py"):
            state.update(eval(open(args.state).read()))
        else:
            with open(args.state) as src:
                state.update(cPickle.load(src))
    for change in args.changes:
        state.update(eval("dict({})".format(change)))

    logging.basicConfig(level=getattr(logging, state['level']), format="%(asctime)s: %(name)s: %(levelname)s: %(message)s")
    logger.debug("State:\n{}".format(pprint.pformat(state)))

    rng = numpy.random.RandomState(state['seed'])
    #True is added by chengyong
    #enc_dec = RNNEncoderDecoder(state, rng, args.skip_init, True)
    #modified by chengyong
    enc_dec = BiEncoderDecoder(state, rng, args.skip_init, False)
    enc_dec.build()
    lm_model = enc_dec.create_lm_model()

    logger.debug("Load data")
    train_data = get_batch_iterator(state)
    logger.debug("Compile trainer")
    algo = eval(state['algo'])(lm_model, state, train_data)
    logger.debug("Run training")
    main = MainLoop(train_data, None, None, lm_model, algo, state, None,
            reset=state['reset'],
            hooks=[RandomSamplePrinter(state, lm_model, train_data)]
                if state['hookFreq'] >= 0
                else None)
    if state['reload']:
        main.load()
    if state['loopIters'] > 0:
        main.main()

if __name__ == "__main__":
    main()
