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
Base class for decoder with Conv layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import abc
from collections import namedtuple

import six
import tensorflow as tf
from tensorflow.python.util import nest  # pylint: disable=E0611
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_ops

from seq2seq.graph_module import GraphModule
from seq2seq.configurable import Configurable
from seq2seq.contrib.seq2seq.decoder import Decoder
from seq2seq.training import utils as training_utils


class DecoderOutput(
    namedtuple("DecoderOutput", ["logits", "predicted_ids"])):
  """Output of an Conv decoder.

  Note that we output both the logits and predictions because during
  dynamic decoding the predictions may not correspond to max(logits).
  For example, we may be sampling from the logits instead.
  """
  pass


@six.add_metaclass(abc.ABCMeta)
class ConvDecoder(Decoder, GraphModule, Configurable):
  """Base class for RNN decoders.

  Args:
    cell: An instance of ` tf.contrib.rnn.RNNCell`
    helper: An instance of `tf.contrib.seq2seq.Helper` to assist decoding
    initial_state: A tensor or tuple of tensors used as the initial cell
      state.
    name: A name for this module
  """

  def __init__(self, params, mode, vocab_size, name="conv_deocder"):
    GraphModule.__init__(self, name)
    Configurable.__init__(self, params, mode)
    self.vocab_size = vocab_size
    # self.params["rnn_cell"] = _toggle_dropout(self.params["rnn_cell"], mode)
    # self.cell = training_utils.get_rnn_cell(**self.params["rnn_cell"])
    # Not initialized yet

  def initialize(self, name=None):
    """ Do nothing
    """
    return
    

  def step(self, name=None):
    """ Not an RNN decoder, no step function needed.
    """
    raise NotImplementedError

  @property
  def batch_size(self):
    # return tf.shape(nest.flatten([self.initial_state])[0])[0]
    raise NotImplementedError

  def _setup(self, initial_state, helper):
    """Sets the initial state and helper for the decoder.
    """
    raise NotImplementedError

  @staticmethod
  def default_params():
    return {
        "max_input_seq_length": 64,
        "input_embeding_dim": 64,
        "output_embeding_dim": 64,
        "init_scale": 0.04,
    }

  def _build(self, inputs, gt_embedding):
    scope = tf.get_variable_scope()
    scope.set_initializer(tf.random_uniform_initializer(
        -self.params["init_scale"],
        self.params["init_scale"]))

    
    max_input_seq_len = self.params["max_input_seq_length"]
    input_embeding_dim = self.params["input_embeding_dim"]
    batch_size = tf.shape(inputs)[0]

    # Patch or trim dynamic input to fixed size.
    # inputs data is RNN result in shape of [batch, seq_len, embedding_dim]
    # Patch with 0 or trim extra length and make the tensor shape to be
    # [batch, max_input_seq_len, embedding_dim]
    target_size = ops.convert_to_tensor(max_input_seq_len, dtype=dtypes.int32,
                                        name="max_input_seq_len")

    def _pad_fn(inputs, max_input_seq_len):
      target_pad = max_input_seq_len - tf.shape(inputs)[1]
      inputs = tf.Print(inputs, [tf.shape(inputs), tf.transpose(inputs, [0, 2, 1])], message="input before pad: ", summarize = 1000)
      padded_inputs = tf.pad(inputs, [[0, 0], [0, target_pad], [0, 0]], "CONSTANT")
      return padded_inputs
      
    
    pad_inputs = tf.cond(tf.shape(inputs)[1] < max_input_seq_len,
                         lambda: _pad_fn(inputs, max_input_seq_len),
	                 lambda: tf.identity(inputs),
                         name="maybe_padding")

    def _trim_fn(inputs, max_input_seq_len):
      return inputs
    
    final_inputs = tf.cond(tf.shape(pad_inputs)[1] > max_input_seq_len,
                           lambda: _trim_fn(pad_inputs, max_input_seq_len),
                           lambda: tf.identity(pad_inputs),
                           name="maybe_trimming")
    
    # Stacks FC layers.
    print (inputs)
    print (batch_size)
    print (input_embeding_dim)
    pad_inputs = tf.reshape(pad_inputs, [batch_size, max_input_seq_len, input_embeding_dim])
    pad_inputs = tf.transpose(pad_inputs, [0, 2, 1])
    # FC layers in sequence.
    y = tf.contrib.layers.stack(pad_inputs,
                                tf.contrib.layers.fully_connected,
                                [128, 64, 32, self.vocab_size],
                                scope='Decoder/FC')
    y = tf.reshape(y, [batch_size, input_embeding_dim * self.vocab_size])
    logits = tf.contrib.layers.fully_connected(inputs=y, num_outputs=self.vocab_size, activation_fn=None)
    # calculate logics.
    predictions = math_ops.cast(
        math_ops.argmax(logits, axis=-1), dtypes.int32)
    return DecoderOutput(
        logits=logits, predicted_ids=predictions), None
