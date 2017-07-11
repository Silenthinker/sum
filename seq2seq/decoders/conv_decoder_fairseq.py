# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Base class for sequence decoders.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
from collections import namedtuple
from pydoc import locate

import six
import tensorflow as tf
from tensorflow.python.util import nest  # pylint: disable=E0611
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops

from seq2seq import graph_utils
from seq2seq.graph_module import GraphModule
from seq2seq.configurable import Configurable
from seq2seq.contrib.seq2seq.decoder import Decoder, dynamic_decode
from seq2seq.contrib.seq2seq.decoder import _transpose_batch_time
#from seq2seq.encoders.pooling_encoder import _create_position_embedding, position_encoding
from seq2seq.encoders.conv_encoder_utils import *
from seq2seq.inference import beam_search  
from tensorflow.python.util import nest
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops
from seq2seq.encoders.encoder import EncoderOutput

class ConvDecoderOutput(
    #namedtuple("ConvDecoderOutput", ["logits", "predicted_ids", "cell_output", "attention_scores", "attention_context"])):
    namedtuple("ConvDecoderOutput", ["logits", "predicted_ids"])): 
    pass


@six.add_metaclass(abc.ABCMeta)
class ConvDecoderFairseq(Decoder, GraphModule, Configurable):
  """An RNN Decoder that uses attention over an input sequence.

  Args:
    cell: An instance of ` tf.contrib.rnn.RNNCell`
    helper: An instance of `tf.contrib.seq2seq.Helper` to assist decoding
    initial_state: A tensor or tuple of tensors used as the initial cell
      state.
    vocab_size: Output vocabulary size, i.e. number of units
      in the softmax layer
    attention_keys: The sequence used to calculate attention scores.
      A tensor of shape `[B, T, ...]`.
    attention_values: The sequence to attend over.
      A tensor of shape `[B, T, input_dim]`.
    attention_values_length: Sequence length of the attention values.
      An int32 Tensor of shape `[B]`.
    attention_fn: The attention function to use. This function map from
      `(state, inputs)` to `(attention_scores, attention_context)`.
      For an example, see `seq2seq.decoder.attention.AttentionLayer`.
    reverse_scores: Optional, an array of sequence length. If set,
      reverse the attention scores in the output. This is used for when
      a reversed source sequence is fed as an input but you want to
      return the scores in non-reversed order.
  """

  def __init__(self,
               params,
               mode,
               vocab_size,
               config,
               target_embedding,
               pos_embedding,
               start_tokens,
               name="conv_decoder_fairseq"):
    GraphModule.__init__(self, name)
    Configurable.__init__(self, params, mode)
    
    self.vocab_size = vocab_size
    self.config=config
    self.target_embedding=target_embedding 
    self.start_tokens=start_tokens
    self._combiner_fn = locate(self.params["position_embeddings.combiner_fn"])
    self.pos_embed = pos_embedding
    self.current_inputs = None
    self.initial_state = None
    self.batch_size = 1
    if self.config.beam_width is not None and self.config.beam_width > 0:
      self.batch_size = self.config.beam_width

  @staticmethod
  def default_params():
    return {
        "cnn.layers": 3,
        "cnn.nhids": "256,256,256",
        "cnn.kwidths": "3,3,3",
        "cnn.nhid_default": 256,
        "cnn.kwidth_default": 3,
        "embedding_dropout_keep_prob": 0.9,
        "nhid_dropout_keep_prob": 0.9,
        "out_dropout_keep_prob": 0.9,
        "position_embeddings.enable": True,
        "position_embeddings.combiner_fn": "tensorflow.add",
        "max_decode_length": 49,
        "nout_embed": 256,
    }
  '''
  @property
  def batch_size(self):
    return self.config.beam_width
  '''

  @property
  def output_size(self):
    return ConvDecoderOutput(
        logits=self.vocab_size,   # need pay attention
        predicted_ids=tf.TensorShape([]))

  @property
  def output_dtype(self):
    return ConvDecoderOutput(
        logits=tf.float32,
        predicted_ids=tf.int32)

  def print_shape(self, name, tensor):
    print(name, tensor.get_shape().as_list()) 
 
  def _setup(self, initial_state, helper=None):
    self.initial_state = initial_state
  
  def initialize(self, batch_size, name=None):
    
    finished = tf.tile([False], [batch_size])
    
    start_tokens_batch = tf.fill([batch_size], self.start_tokens)
    first_inputs = tf.nn.embedding_lookup(self.target_embedding, start_tokens_batch)
    first_inputs = tf.expand_dims(first_inputs, 1)
    zeros_padding = tf.zeros([batch_size, self.params['max_decode_length']-1, self.target_embedding.get_shape().as_list()[-1]])
    first_inputs = tf.concat([first_inputs, zeros_padding], axis=1)

    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      outputs = tf.tile(self.initial_state.outputs, [batch_size,1,1]) 
      attention_values = tf.tile(self.initial_state.attention_values, [batch_size,1,1]) 
    else:
      outputs = self.initial_state.outputs
      attention_values = self.initial_state.attention_values

    enc_output = EncoderOutput(
        outputs=outputs,
        final_state=self.initial_state.final_state,
        attention_values=attention_values,
        attention_values_length=self.initial_state.attention_values_length)
    
    return finished, first_inputs, enc_output
  
  def finalize(self, outputs, final_state):
 
    return outputs, final_state
   
  def next_inputs(self, sample_ids, batch_size, name=None):
    finished = math_ops.equal(sample_ids, self.config.eos_token)
    all_finished = math_ops.reduce_all(finished)
    next_inputs = control_flow_ops.cond(
        all_finished,
        # If we're finished, the next_inputs value doesn't matter
        lambda:  tf.nn.embedding_lookup(self.target_embedding, tf.tile([self.config.eos_token], [batch_size])),
        lambda: tf.nn.embedding_lookup(self.target_embedding, sample_ids))
    return all_finished, next_inputs

  def _create_position_embedding(self, lengths, maxlen):

    # Slice to size of current sequence
    pe_slice = self.pos_embed[2:maxlen+2, :] # [T, embed_size]
    # Replicate encodings for each element in the batch
    batch_size = tf.shape(lengths)[0] # B
    pe_batch = tf.tile([pe_slice], [batch_size, 1, 1]) # [B, T, embed_size]

    # Mask out positions that are padded
    positions_mask = tf.sequence_mask(
        lengths=lengths, maxlen=maxlen, dtype=tf.float32) # [B, T]
    positions_embed = pe_batch * tf.expand_dims(positions_mask, 2) # [B, T, 1] * [B, T, embed_size]

    return positions_embed # [B, T, embed_size]
  
  def add_position_embedding(self, inputs, time, batch_size):
    seq_pos_embed = self.pos_embed[2:time+1+2,:]  
    seq_pos_embed = tf.expand_dims(seq_pos_embed, axis=0) 
    seq_pos_embed_batch = tf.tile(seq_pos_embed, [batch_size, 1, 1])
    
    return self._combiner_fn(inputs, seq_pos_embed_batch)

  def step(self, time, inputs, state, batch_size, name=None, sample=False):
    '''
    Args:
      sample: True to generate by sampling; otherwise greedily
    '''
   
    cur_inputs = inputs[:,0:time+1,:] 
    zeros_padding = inputs[:,time+2:,:] 
    cur_inputs_pos = self.add_position_embedding(cur_inputs, time, batch_size)
    
    enc_output = state 
    logits = self.infer_conv_block(enc_output, cur_inputs_pos, is_train=False) # [B, V]
    
    
    softmax = tf.nn.softmax(logits, dim=-1, name=None) # [B, self.V]
    log_softmax = tf.log(tf.clip_by_value(softmax, 1e-20, 1.0))
    if sample:
      sample_ids = tf.multinomial(log_softmax, 1) # [None, 1]
      sample_ids = tf.cast(tf.reshape(sample_ids, [-1]), dtypes.int32) # [B]
    else:
      sample_ids = tf.cast(tf.argmax(logits, axis=-1), dtypes.int32) # greedy... [B]
    
    one_hot = tf.one_hot(sample_ids, log_softmax.get_shape().as_list()[1], axis=-1) # [B, V]
    log_prob = tf.reduce_sum(tf.multiply(one_hot, log_softmax), axis=1) # [B, 1] # compute log prob of sampling the word
    log_prob.set_shape([batch_size])
    
    finished, next_inputs = self.next_inputs(sample_ids=sample_ids, batch_size=batch_size)
    next_inputs = tf.reshape(next_inputs, [batch_size, 1, inputs.get_shape().as_list()[-1]])
    next_inputs = tf.concat([cur_inputs, next_inputs], axis=1)
    next_inputs = tf.concat([next_inputs, zeros_padding], axis=1)
    next_inputs.set_shape([batch_size, self.params['max_decode_length'], inputs.get_shape().as_list()[-1]])
    outputs = ConvDecoderOutput(
        logits=logits,
        predicted_ids=sample_ids)
    return outputs, enc_output, next_inputs, finished, log_prob


    

  def infer_conv_block(self, enc_output, input_embed, is_train=None):
    # Apply dropout to embeddings
    if is_train is None:
      is_train = self.mode == tf.contrib.learn.ModeKeys.TRAIN

    input_embed = tf.contrib.layers.dropout(
        inputs=input_embed,
        keep_prob=self.params["embedding_dropout_keep_prob"],
        is_training=is_train) # tf.contrib.learn.ModeKeys.INFER
     
    next_layer = self.conv_block(enc_output, input_embed, is_train)
    shape = next_layer.get_shape().as_list()  
    
    logits = tf.reshape(next_layer, [-1,shape[-1]])   
    return logits

  def conv_block(self, enc_output, input_embed, is_train=True):
    if is_train:
      mode = tf.contrib.learn.ModeKeys.TRAIN
    else:
      mode = tf.contrib.learn.ModeKeys.INFER
    with tf.variable_scope("decoder_cnn"):    
      next_layer = input_embed
      if self.params["cnn.layers"] > 0:
        nhids_list = parse_list_or_default(self.params["cnn.nhids"], self.params["cnn.layers"], self.params["cnn.nhid_default"])
        kwidths_list = parse_list_or_default(self.params["cnn.kwidths"], self.params["cnn.layers"], self.params["cnn.kwidth_default"])
        
        # mapping emb dim to hid dim
        next_layer = linear_mapping_weightnorm(next_layer, nhids_list[0], dropout=self.params["embedding_dropout_keep_prob"], var_scope_name="linear_mapping_before_cnn")      
         
        next_layer = conv_decoder_stack(input_embed, enc_output, next_layer, nhids_list, kwidths_list, {'src':self.params["embedding_dropout_keep_prob"], 'hid': self.params["nhid_dropout_keep_prob"]}, mode=mode)
    
    with tf.variable_scope("softmax"):
      if is_train:
        next_layer = linear_mapping_weightnorm(next_layer, self.params["nout_embed"], var_scope_name="linear_mapping_after_cnn")
      else:         
        next_layer = linear_mapping_weightnorm(next_layer[:,-1:,:], self.params["nout_embed"], var_scope_name="linear_mapping_after_cnn")
      next_layer = tf.contrib.layers.dropout(
        inputs=next_layer,
        keep_prob=self.params["out_dropout_keep_prob"],
        is_training=is_train)
     
      next_layer = linear_mapping_weightnorm(next_layer, self.vocab_size, in_dim=self.params["nout_embed"], dropout=self.params["out_dropout_keep_prob"], var_scope_name="logits_before_softmax")
      
    return next_layer 
 
  def init_params_in_loop(self, batch_size):
    with tf.variable_scope("decoder"):
      initial_finished, initial_inputs, initial_state = self.initialize(batch_size)
      enc_output = initial_state
      logits = self.infer_conv_block(enc_output, initial_inputs, is_train=False)
      

  def print_tensor_shape(self, tensor, name):
    print(name, tensor.get_shape().as_list()) 
  
  def conv_decoder_infer(self):
    maximum_iterations = self.params["max_decode_length"]
    batch_size = self.batch_size
    with tf.variable_scope("decoder"):
      initial_finished, initial_inputs, initial_state = self.initialize(batch_size)
      enc_output = initial_state
      logits = self.infer_conv_block(enc_output, initial_inputs, is_train=False)
    conv_dec_dict = {"initial_inputs": initial_inputs, "enc_output": initial_state, "logits": logits}
    graph_utils.add_dict_to_collection(conv_dec_dict, "conv_dec_dict")
    tf.get_variable_scope().reuse_variables()    
    outputs, final_state, _ = dynamic_decode(
        decoder=self,
        output_time_major=True,
        impute_finished=False,
        maximum_iterations=maximum_iterations)
    
    return outputs, final_state

  def conv_decoder_train_infer(self, enc_output, sequence_length, sample=False):
    '''
    Infer during training, return greedy and sampled generation
    in this mode, sentences are generated one by one (batch_size = 1)
    total number of sentences is batch_size
    Returns:
      outputs: complicated structure, elements follow the shape [T, B]
      log_pro_sum: [B]
    '''
    maximum_iterations = self.params["max_decode_length"] - 1
    batch_size = enc_output.attention_values_length.get_shape().as_list()[0]
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
      enc_output = EncoderOutput(
        outputs=enc_output.outputs,
        final_state=enc_output.final_state,
        attention_values=enc_output.attention_values,
        attention_values_length=enc_output.attention_values_length)
      self._setup(initial_state=enc_output)
      with tf.variable_scope("decoder"):
        initial_finished, initial_inputs, initial_state = self.initialize(batch_size)
        logits = self.infer_conv_block(initial_state, initial_inputs, is_train=False)
        
      outputs, final_state, log_prob_sum = dynamic_decode(
          decoder=self,
          output_time_major=True,
          impute_finished=False,
          maximum_iterations=maximum_iterations,
          sample=sample,
          batch_size=batch_size)
    # graph_utils.add_dict_to_collection({"log_prob_sum": log_prob_sum}, "probs")
    return {"outputs": outputs, "log_prob_sum": log_prob_sum}
    

  def conv_decoder_train(self, enc_output, labels, sequence_length):
    '''
    If sample is set True, returns two ConvDecoderOutput: greedy and sampled, respectively;
    otherwise, returns greedy only, the sampled is None
    '''
    embed_size = labels.get_shape().as_list()[-1]
    if self.params["position_embeddings.enable"]:
      positions_embed = self._create_position_embedding(
          lengths=sequence_length,
          maxlen=tf.shape(labels)[1])
      labels = self._combiner_fn(labels, positions_embed)
     
    # Apply dropout to embeddings
    inputs = tf.contrib.layers.dropout(
        inputs=labels,
        keep_prob=self.params["embedding_dropout_keep_prob"],
        is_training=self.mode == tf.contrib.learn.ModeKeys.TRAIN)
    
    next_layer = self.conv_block(enc_output, inputs, True)
      
       
    logits = _transpose_batch_time(next_layer) 
    
    sample_ids = tf.cast(tf.argmax(logits, axis=-1), tf.int32) # greedy...
    greedy_output = ConvDecoderOutput(logits=logits, predicted_ids=sample_ids) 
    return {"outputs": greedy_output}

  def _build(self, enc_output, labels=None, sequence_length=None, rl=True):
    
    if not self.initial_state:
      self._setup(initial_state=enc_output)

    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      outputs, states = self.conv_decoder_infer()
      return self.finalize(outputs, states)
    else:
      with tf.variable_scope("decoder"):  # when infer, dynamic decode will add decoder scope, so we add here to keep it the same  
        outputs = self.conv_decoder_train(enc_output=enc_output, labels=labels, sequence_length=sequence_length)
        states = None
      if rl:
        outputs_greedy = self.conv_decoder_train_infer(enc_output=enc_output, sequence_length=sequence_length, sample=False)
        outputs_sampled = self.conv_decoder_train_infer(enc_output=enc_output, sequence_length=sequence_length, sample=True)
        return outputs, outputs_greedy, outputs_sampled
      else:
        return outputs
