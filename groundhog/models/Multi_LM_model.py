"""
Implementation of a language model class.


TODO: write more documentation
"""
__docformat__ = 'restructedtext en'
__authors__ = ("Razvan Pascanu "
               "KyungHyun Cho "
               "Caglar Gulcehre ")
__contact__ = "Razvan Pascanu <r.pascanu@gmail>"


import numpy
import itertools
import logging

import cPickle as pkl

import theano
import theano.tensor as TT
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from groundhog.utils import id_generator
from groundhog.layers.basic import Model
from groundhog.layers.basic import Container

logger = logging.getLogger(__name__)

#deualt sample_fn[0] corresponds to forward, inverse
#cost_layer forward, inverse, agree
class Multi_LM_Model(Container):
    def  __init__(self,
                  sum_layer = None,
                  cost_layer = None,
                  sample_fn = None,
                  valid_fn = None,
                  noise_fn = None,
                  clean_before_noise_fn = False,
                  clean_noise_validation=True,
                  weight_noise_amount = 0,
                  indx_word="/data/lisa/data/PennTreebankCorpus/dictionaries.npz",
                  need_inputs_for_generating_noise=False,
                  indx_word_src=None,
                  indx_word_1 = None,
                  indx_word_src_1 = None,
                  character_level = False,
                  exclude_params_for_norm=None,
                  rng = None):
        """
        Constructs a model, that respects the interface required by the
        trainer class.

        :type cost_layer: groundhog layer
        :param cost_layer: the cost (last) layer of the model

        :type sample_fn: function or None
        :param sample_fn: function used to sample from the model

        :type valid_fn: function or None
        :param valid_fn: function used to compute the validation error on a
            minibatch of examples

        :type noise_fn: function or None
        :param noise_fn: function called to corrupt an input (that
            potentially will be denoised by the model)

        :type clean_before_noise_fn: bool
        :param clean_before_noise_fn: If the weight noise should be removed
            before calling the `noise_fn` to corrupt some input

        :type clean_noise_validation: bool
        :param clean_noise_validation: If the weight noise should be removed
            before calling the validation function

        :type weight_noise_amount: float or theano scalar
        :param weight_noise_amount: weight noise scale (standard deviation
            of the Gaussian from which it is sampled)

        :type indx_word: string or None
        :param indx_word: path to the file describing how to match indices
            to words (or characters)

        :type need_inputs_for_generating_noise: bool
        :param need_inputs_for_generating_noise: flag saying if the shape of
            the inputs affect the shape of the weight noise that is generated at
            each step

        :type indx_word_src: string or None
        :param indx_word_src: similar to indx_word (but for the source
            language

        :type character_level: bool
        :param character_level: flag used when sampling, saying if we are
            running the model on characters or words

        :type excluding_params_for_norm: None or list of theano variables
        :param excluding_params_for_norm: list of parameters that should not
            be included when we compute the norm of the gradient (for norm
            clipping). Usually the output weights if the output layer is
            large

        :type rng: numpy random generator
        :param rng: numpy random generator

        """
        super(Multi_LM_Model, self).__init__()
        if rng == None:
            rng = numpy.random.RandomState(123)
        self.rng = rng
        self.trng = RandomStreams(rng.randint(1000)+1)
        assert type(sample_fn) in (list, tuple)
        assert len(sample_fn) > 1
        self.sample_fn_fwd = sample_fn[0]
        self.sample_fn_inv = sample_fn[1]
        self.indx_word = indx_word
        self.indx_word_src = indx_word_src
        self.indx_word_1 = indx_word_1
        self.indx_word_src_1 = indx_word_src_1
        self.param_grads = sum_layer.grads
        self.params = sum_layer.params
        #updates length is zero
        self.updates = sum_layer.updates
        #noise params is zero
        self.noise_params = sum_layer.noise_params
        self.noise_params_shape_fn = sum_layer.noise_params_shape_fn
        #inputs exist
        self.inputs = sum_layer.inputs
        self.params_grad_scale = sum_layer.params_grad_scale
        self.train_cost = sum_layer.cost
        #self.out = output_layer.out
        #schedules is zero
        self.schedules = sum_layer.schedules
        #self.output_layer = output_layer
        #Properties is zero
        self.properties = sum_layer.properties
        
        self._get_samples_fwd = cost_layer[0]._get_samples
        self._get_samples_inv = cost_layer[1]._get_samples


        if exclude_params_for_norm is None:
            self.exclude_params_for_norm = []
        else:
            self.exclude_params_for_norm = exclude_params_for_norm
        self.need_inputs_for_generating_noise=need_inputs_for_generating_noise
        
        # is an array
        self.cost_layer = cost_layer
        
        self.validate_step = valid_fn
        self.clean_noise_validation = clean_noise_validation
        self.noise_fn = noise_fn
        self.clean_before = clean_before_noise_fn
        self.weight_noise_amount = weight_noise_amount
        self.character_level = character_level

        self.valid_costs = ['cost','ppl']
        # Assume a single cost
        # We need to merge these lists

        #forward
        fwd_state_below = self.cost_layer[0].state_below
        if hasattr(self.cost_layer[0], 'mask') and self.cost_layer[0].mask:
            num_words_fwd = TT.sum(self.cost_layer[0].mask)
        else:
            num_words_fwd = TT.cast(fwd_state_below.shape[0], 'float32')
        scale_fwd = getattr(self.cost_layer[0], 'cost_scale', numpy.float32(1))
        if not scale_fwd:
            scale_fwd = numpy.float32(1)
        scale_fwd *= numpy.float32(numpy.log(2))


        grad_norm = TT.sqrt(sum(TT.sum(x**2)
            for x,p in zip(self.param_grads, self.params) if p not in
                self.exclude_params_for_norm))
        fwd_new_properties = [
                ('grad_norm', grad_norm),
                ('fwd_train_cost', self.cost_layer[0].cost),
                ('fwd_log2_p_word', self.cost_layer[0].cost / num_words_fwd / scale_fwd),
                ('fwd_log2_p_expl', self.cost_layer[0].cost_per_sample.mean() / scale_fwd)]
        self.properties += fwd_new_properties

        #inverse
        inv_state_below = self.cost_layer[1].state_below
        if hasattr(self.cost_layer[1], 'mask') and self.cost_layer[1].mask:
            num_words_inv = TT.sum(self.cost_layer[1].mask)
        else:
            num_words_inv = TT.cast(inv_state_below.shape[1], 'float32')
        scale_inv = getattr(self.cost_layer[1], 'cost_scale', numpy.float32(1))
        if not scale_inv:
            scale_inv = numpy.float32(1)
        scale_inv *= numpy.float32(numpy.log(2))

        inv_new_properties = [
                ('inv_train_cost', self.cost_layer[1].cost),
                ('inv_log2_p_word', self.cost_layer[1].cost / num_words_inv / scale_inv),
                ('inv_log2_p_expl', self.cost_layer[1].cost_per_sample.mean() / scale_inv)]
        self.properties += inv_new_properties

        #self.properties += [('agree_cost', self.cost_layer[2].cost)]


        if len(self.noise_params) >0 and weight_noise_amount:
            if self.need_inputs_for_generating_noise:
                inps = self.inputs
            else:
                inps = []
            self.add_noise = theano.function(inps,[],
                                             name='add_noise',
                                             updates = [(p,
                                                 self.trng.normal(shp_fn(self.inputs),
                                                     avg =0,
                                                     std=weight_noise_amount,
                                                     dtype=p.dtype))
                                                 for p, shp_fn in
                                                        zip(self.noise_params,
                                                         self.noise_params_shape_fn)],
                                            on_unused_input='ignore')
            self.del_noise = theano.function(inps,[],
                                             name='del_noise',
                                             updates=[(p,
                                                       TT.zeros(shp_fn(self.inputs),
                                                                p.dtype))
                                                      for p, shp_fn in
                                                      zip(self.noise_params,
                                                          self.noise_params_shape_fn)],
                                            on_unused_input='ignore')
        else:
            self.add_noise = None
            self.del_noise = None


    def validate(self, data_iterator, train=False):
        cost = 0
        n_batches = 0
        n_steps = 0
        if self.del_noise and self.clean_noise_validation:
            if self.need_inputs_for_generating_noise:
                self.del_noise(**vals)
            else:
                self.del_noise()

        for vals in data_iterator:
            n_batches += 1

            if isinstance(vals, dict):
                val = vals.values()[0]
                if val.ndim ==3:
                    n_steps += val.shape[0]*val.shape[1]
                else:
                    n_steps += val.shape[0]

                _rvals = self.validate_step( **vals)
                cost += _rvals
            else:
                # not dict
                if vals[0].ndim ==3:
                    n_steps += vals[0].shape[0]*vals[1].shape[1]
                else:
                    n_steps += vals[0].shape[0]
                if self.del_noise and self.clean_noise_validation:
                    if self.need_inputs_for_generating_noise:
                        self.del_noise(*vals)
                    else:
                        self.del_noise()
                inps = list(vals)
                _rvals = self.validate_step(*inps)
                _cost += _rvals

        n_steps = numpy.log(2.)*n_steps
        cost = cost / n_steps

        entropy = cost# (numpy.log(2.))
        ppl = 10**(numpy.log(2)*cost/numpy.log(10))
        return [('cost',entropy), ('ppl',ppl)]


    def load_dict(self, opts):
        """
        Loading the dictionary that goes from indices to actual words
        """

        if self.indx_word and '.pkl' in self.indx_word[-4:]:
            data_dict = pkl.load(open(self.indx_word, "r"))
            self.word_indxs = data_dict
            self.word_indxs[opts['null_sym_target']] = '<eol>'
            self.word_indxs[opts['unk_sym_target']] = opts['oov']
        elif self.indx_word and '.np' in self.indx_word[-4:]:
            self.word_indxs = numpy.load(self.indx_word)['unique_words']

        if self.indx_word_src and '.pkl' in self.indx_word_src[-4:]:
            data_dict = pkl.load(open(self.indx_word_src, "r"))
            self.word_indxs_src = data_dict
            self.word_indxs_src[opts['null_sym_source']] = '<eol>'
            self.word_indxs_src[opts['unk_sym_source']] = opts['oov']
        elif self.indx_word_src and '.np' in self.indx_word_src[-4:]:
            self.word_indxs_src = numpy.load(self.indx_word_src)['unique_words']


        if self.indx_word_1 and '.pkl' in self.indx_word_1[-4:]:
            data_dict = pkl.load(open(self.indx_word_1, "r"))
            self.word_indxs_1 = data_dict
            self.word_indxs_1[opts['null_sym_target']] = '<eol>'
            self.word_indxs_1[opts['unk_sym_target']] = opts['oov']
        elif self.indx_word_1 and '.np' in self.indx_word_1[-4:]:
            self.word_indxs_1 = numpy.load(self.indx_word_1)['unique_words']

        if self.indx_word_src_1 and '.pkl' in self.indx_word_src_1[-4:]:
            data_dict = pkl.load(open(self.indx_word_src_1, "r"))
            self.word_indxs_src_1 = data_dict
            self.word_indxs_src_1[opts['null_sym_source']] = '<eol>'
            self.word_indxs_src_1[opts['unk_sym_source']] = opts['oov']
        elif self.indx_word_src_1 and '.np' in self.indx_word_src_1[-4:]:
            self.word_indxs_src_1 = numpy.load(self.indx_word_src_1)['unique_words']
    
    #added by chengyong
    def _get_samples(self, word_indxs_src, word_indxs, sample_fn = None, length=30, temp=1, *inps):
        """
        See parent class
        """
        assert hasattr(self, 'word_indxs_src')

        character_level = False
        if hasattr(self, 'character_level'):
            character_level = self.character_level
        if self.del_noise:
            self.del_noise()
        [values, probs] = sample_fn(length, temp, *inps)
        #print 'Generated sample is:'
        #print
        if values.ndim > 1:
            for d in xrange(2):
                print '%d-th sentence' % d
                print 'Input: ',
                if character_level:
                    sen = []
                    for k in xrange(inps[0].shape[0]):
                        if word_indxs_src[inps[0][k][d]] == '<eol>':
                            break
                        sen.append(word_indxs_src[inps[0][k][d]])
                    print "".join(sen),
                else:
                    for k in xrange(inps[0].shape[0]):
                        print word_indxs_src[inps[0][k][d]],
                        if word_indxs_src[inps[0][k][d]] == '<eol>':
                            break
                print ''
                print 'Output: ',
                if character_level:
                    sen = []
                    for k in xrange(values.shape[0]):
                        if word_indxs[values[k][d]] == '<eol>':
                            break
                        sen.append(word_indxs[values[k][d]])
                    print "".join(sen),
                else:
                    for k in xrange(values.shape[0]):
                        print word_indxs[values[k][d]],
                        if word_indxs[values[k][d]] == '<eol>':
                            break
                print
                print
        else:
            print 'Input:  ',
            if character_level:
                sen = []
                for k in xrange(inps[0].shape[0]):
                    if word_indxs_src[inps[0][k]] == '<eol>':
                        break
                    sen.append(word_indxs_src[inps[0][k]])
                print "".join(sen),
            else:
                for k in xrange(inps[0].shape[0]):
                    print word_indxs_src[inps[0][k]],
                    if word_indxs_src[inps[0][k]] == '<eol>':
                        break
            print ''
            print 'Output: ',
            if character_level:
                sen = []
                for k in xrange(values.shape[0]):
                    if word_indxs[values[k]] == '<eol>':
                        break
                    sen.append(word_indxs[values[k]])
                print "".join(sen),
            else:
                for k in xrange(values.shape[0]):
                    print word_indxs[values[k]],
                    if word_indxs[values[k]] == '<eol>':
                        break
            print
            print
    #added by chengyong, defautly, every
    def get_samples_fwd(self, length = 30, temp=1, *inps):
        #self._get_samples_fwd(self, length, temp, *inps)
        self._get_samples(self.word_indxs_src, self.word_indxs, self.sample_fn_fwd,
             length, temp, *inps)
    #added by chengyong
    def get_samples_inv(self, length = 30, temp=1, *inps):
        self._get_samples(self.word_indxs_src_1, self.word_indxs_1, self.sample_fn_inv,
             length, temp, *inps)

    def perturb(self, *args, **kwargs):
        if args:
            inps = args
            assert not kwargs
        if kwargs:
            inps = kwargs
            assert not args

        if self.noise_fn:
            if self.clean_before and self.del_noise:
                if self.need_inputs_for_generating_noise:
                    self.del_noise(*args, **kwargs)
                else:
                    self.del_noise()
            inps = self.noise_fn(*args, **kwargs)
        if self.add_noise:
            if self.need_inputs_for_generating_noise:
                self.add_noise(*args, **kwargs)
            else:
                self.add_noise()
        return inps

    def get_schedules(self):
        return self.schedules
