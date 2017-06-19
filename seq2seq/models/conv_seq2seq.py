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
Definition of a basic seq2seq model
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections

from pydoc import locate
import tensorflow as tf

from seq2seq import graph_utils
from seq2seq import losses as seq2seq_losses
from seq2seq.models.model_base import _flatten_dict
from seq2seq.models.seq2seq_model import Seq2SeqModel
from seq2seq.graph_utils import templatemethod


class ConvSeq2Seq(Seq2SeqModel):
  """A customiaze Sequence2Sequence model with a dynamic rnn encoder and convolution
  decoder.

  Args:
    source_vocab_info: An instance of `VocabInfo`
      for the source vocabulary
    target_vocab_info: An instance of `VocabInfo`
      for the target vocabulary
    params: A dictionary of hyperparameters
  """

  def __init__(self, params, mode, name="conv_seq2seq"):
    super(ConvSeq2Seq, self).__init__(params, mode, name)
    self.encoder_class = locate(self.params["encoder.class"])
    self.decoder_class = locate(self.params["decoder.class"])

  @staticmethod
  def default_params():
    params = Seq2SeqModel.default_params().copy()
    params.update({
        "encoder.class": "seq2seq.encoders.UnidirectionalRNNEncoder",
        "encoder.params": {},  # Arbitrary parameters for the encoder
        "decoder.class": "seq2seq.decoders.ConvDecoder",
        "decoder.params": {},  # Arbitrary parameters for the decoder
    })
    return params

  @templatemethod("create_predictions")
  def create_predictions(self, decoder_output, features, labels, losses=None):
    """Creates the dictionary of predictions that is returned by the model.
    """
    predictions = {}

    # Add features and, if available, labels to predictions
    predictions.update(_flatten_dict({"features": features}))
    if labels is not None:
      predictions.update(_flatten_dict({"labels": labels}))

    if losses is not None:
      predictions["losses"] = losses

    # Decoders returns output in time-major form [T, B, ...]
    # Here we transpose everything back to batch-major for the user
    output_dict = collections.OrderedDict(
        zip(decoder_output._fields, decoder_output))
    decoder_output_flat = _flatten_dict(output_dict)
    predictions.update(decoder_output_flat)

    # If we predict the ids also map them back into the vocab and process them
    if "predicted_ids" in predictions.keys():
      predicted_tk_id = predictions["predicted_ids"]
      vocab_tables = graph_utils.get_dict_from_collection("vocab_tables")
      target_id_to_vocab = vocab_tables["target_id_to_vocab"]
      predicted_tokens = target_id_to_vocab.lookup(
          tf.to_int64(predicted_tk_id))
      # Raw predicted tokens
      # Force to reshape to [batch_size, 1, 1] so that it has same shape as regular seq2seq,
      # and all post process code can be reused.
      #predicted_tokens = tf.reshape(predicted_tokens, [tf.shape(predicted_tokens)[0], 1, 1])
      predictions["predicted_tokens"] = predicted_tokens

    return predictions

  @templatemethod("compute_loss")
  def compute_loss(self, decoder_output, _features, labels):
    """Computes the loss for this model.

    Returns a tuple `(losses, loss)`, where `losses` are the per-batch
    losses and loss is a single scalar tensor to minimize.
    """
    #pylint: disable=R0201
    # Calculate loss per example-timestep of shape [B, T]
    print (decoder_output)
    print (labels)
    logits = decoder_output.logits
    # logits = tf.Print(logits, [tf.shape(logits), logits], message="compute_loss, logits:", summarize=18)
    ll = labels["target_class"]
    # ll = tf.Print(ll, [tf.shape(ll), ll], message="compute_loss, target_class:", summarize=18)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=ll)
    print (decoder_output.logits)
    print (labels["target_onehot"])
    print (losses)

    return losses, tf.reduce_sum(losses)

  def _decode_train(self, decoder, encoder_output, _features, labels):
    """Runs decoding in training mode"""
    target_embedded = tf.nn.embedding_lookup(self.target_embedding,
                                             labels["target_ids"])
    """
    decoder_inputs = tf.Print(encoder_output.outputs,
                               [tf.shape(_features["source_ids"]),
                                tf.shape(_features["source_tokens"]),
                                tf.shape(labels["target_class"]),
                                tf.shape(labels["target_onehot"]),
                                labels["target_class"],
                                labels["target_onehot"], labels["target_tokens"]],
                               message="encoder input: ", summarize=1000)
    """
    return decoder(encoder_output.outputs, target_embedded[:, 1:-1])

  def _decode_infer(self, decoder, encoder_output, features, labels):
    """Runs decoding in inference mode"""
    decoder_inputs = tf.Print(encoder_output.outputs,
                               [tf.shape(features["source_ids"]),
                                tf.shape(features["source_tokens"]),
                                features["source_ids"], features["source_tokens"]],
                               message="encoder input: ", summarize=1000)
    return decoder(decoder_inputs, None)

  @templatemethod("encode")
  def encode(self, features, labels):
    source_embedded = tf.nn.embedding_lookup(self.source_embedding,
                                             features["source_ids"])
    """
    source_embedded = tf.Print(source_embedded,
                               [tf.shape(features["source_ids"]),
                                tf.shape(features["source_tokens"]),
                                tf.shape(features["target_class"]),
                                tf.shape(features["target_onehot"]),
                                tf.shape(features["target_onehot"]), features["source_tokens"]],
                               message="encoder input: ", summarize=1000)
    """
    encoder_fn = self.encoder_class(self.params["encoder.params"], self.mode)
    return encoder_fn(source_embedded, features["source_len"])

  @templatemethod("decode")
  def decode(self, encoder_output, features, labels):
    decoder = self.decoder_class(
        params=self.params["decoder.params"],
        mode=self.mode,
        vocab_size=self.target_vocab_info.total_size)

    if self.mode == tf.contrib.learn.ModeKeys.INFER:
      return self._decode_infer(decoder, encoder_output, features,
                                labels)
    else:
      return self._decode_train(decoder, encoder_output, features,
                                labels)
