#! /usr/bin/env python
# -*- coding: utf-8 -*-
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

""" Generates model predictions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys 

from pydoc import locate

import yaml
from six import string_types

import tensorflow as tf
from tensorflow import gfile

from seq2seq import tasks, models
from seq2seq.configurable import _maybe_load_yaml, _deep_merge_dict
from seq2seq.data import input_pipeline
from seq2seq.inference import create_inference_graph
from seq2seq.training import utils as training_utils

import BaseHTTPServer
import cgi

from BaseHTTPServer import BaseHTTPRequestHandler
from BaseHTTPServer import HTTPServer

tf.flags.DEFINE_string("tasks", "[{'class': 'DecodeText'}]",
                        "List of inference tasks to run.")
tf.flags.DEFINE_string("model_params", "{}", """Optionally overwrite model
                        parameters for inference""")

tf.flags.DEFINE_string("config_path", None,
                       """Path to a YAML configuration file defining FLAG
                       values and hyperparameters. Refer to the documentation
                       for more details.""")

tf.flags.DEFINE_string("input_pipeline", None,
                       """Defines how input data should be loaded.
                       A YAML string.""")

tf.flags.DEFINE_string("model_dir",
                       "/home/chunfengh/nmt_data/toy_reverse/model/",
                       "directory to load model from")
tf.flags.DEFINE_string("checkpoint_path", None,
                       """Full path to the checkpoint to be loaded. If None,
                       the latest checkpoint in the model dir is used.""")
tf.flags.DEFINE_integer("batch_size", 32, "the train/dev batch size")

FLAGS = tf.flags.FLAGS


class Seq2SeqInferer:
  def __init__(self, _argv):
    if isinstance(FLAGS.tasks, string_types):
      FLAGS.tasks = _maybe_load_yaml(FLAGS.tasks)

    # Load saved training options
    train_options = training_utils.TrainOptions.load(FLAGS.model_dir)

    # Create the model
    model_cls = locate(train_options.model_class) or \
      getattr(models, train_options.model_class)
    model_params = train_options.model_params
    model_params = _deep_merge_dict(
        model_params, _maybe_load_yaml(FLAGS.model_params))
    self.model = model_cls(
        params=model_params,
        mode=tf.contrib.learn.ModeKeys.INFER)

    # Load inference tasks
    hooks = []
    for tdict in FLAGS.tasks:
      if not "params" in tdict:
        tdict["params"] = {}
      task_cls = locate(tdict["class"]) or getattr(tasks, tdict["class"])
      task = task_cls(tdict["params"])
      hooks.append(task)

    # Create the graph used for inference
    self.input_data = tf.placeholder(tf.string)
    self.tokens = tf.string_split(self.input_data).values
    self.length = tf.size(self.tokens)
    self.features = {'source_tokens': tf.expand_dims(self.tokens, 0),
                'source_len':    tf.expand_dims(self.length, 0)}
    self.predictions, _, _ = self.model(features=self.features,
                                        labels=None, params=None)
    saver = tf.train.Saver()
    checkpoint_path = FLAGS.checkpoint_path
    if not checkpoint_path:
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.model_dir)

    def session_init_op(_scaffold, sess):
      saver.restore(sess, checkpoint_path)
      tf.logging.info("Restored model from %s", checkpoint_path)

    scaffold = tf.train.Scaffold(init_fn=session_init_op)
    session_creator = tf.train.ChiefSessionCreator(scaffold=scaffold)
    self.sess = tf.train.MonitoredSession(session_creator=session_creator,
                                          hooks=hooks)
    print ('Model initialized')

  def Infer(self, input_string):
    data = unicode(' ').join(
        [input_string.encode('utf-8'), u' SEQUENCE_END']).encode('utf-8')
    output = self.sess.run([self.predictions[u'predicted_tokens']],
                           feed_dict={self.input_data: [data]})
    # remove last token 'SEQUENCE_END'
    str_list = output[0][0][:-1]
    return unicode(' ').join(str_list).encode('utf-8')


PORT_NUMBER = 8888

PAGE = u'''
<!DOCTYPE html>
<html>
<body>
<form action="/parse" method="POST">
请输入一个完整句子：<br>
<input type="text" name="sentence" size=100>
<br/>
<input type="submit">
</form>
<br/>
输入句子： <br>
<pre>
{0}
</pre>
<br>
转换结果：<br/>
<pre>
{1}
</pre>
</body>
</html>
'''

class MyHandler(BaseHTTPServer.BaseHTTPRequestHandler):
  def do_HEAD(s):
    s.send_response(200)
    s.send_header("Content-type", "text/html; charset=utf-8")
    s.end_headers()

  def do_GET(s):
    """Respond to a GET request."""
    s.send_response(200)
    s.send_header("Content-type", "text/html; charset=utf-8")
    s.end_headers()
    s.wfile.write(PAGE.format(' ', ' ').encode("utf-8"))  
  
  def do_POST(s):
    form = cgi.FieldStorage(fp=s.rfile,
                            headers=s.headers,
                            environ={"REQUEST_METHOD": "POST"})
    input_text = ''
    for item in form.list:
      #print "begin: %s = %s" % (item.name, item.value)
      if item.name == 'sentence':
        input_text = item.value
    target_text = ''
    if input_text:
      target_text = s.inferer.Infer(input_text.encode('utf-8'))
    s.send_response(200)
    s.send_header("Content-type", "text/html; charset=utf-8")
    s.end_headers()
    s.wfile.write(PAGE.encode("utf-8").format(input_text, target_text))


if __name__ == '__main__':
    server_class = BaseHTTPServer.HTTPServer


def main(_argv):
  reload(sys)  
  sys.setdefaultencoding('utf8')
  inferer = Seq2SeqInferer(_argv)
  MyHandler.inferer = inferer

  server_class = BaseHTTPServer.HTTPServer
  httpd = server_class(('', PORT_NUMBER), MyHandler)
  try:
    httpd.serve_forever()
  except KeyboardInterrupt:
    pass
  httpd.server_close()


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
