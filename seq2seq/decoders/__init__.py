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
"""Collection of decoders and decoder-related functions.
"""

from seq2seq.decoders.rnn_decoder import *
from seq2seq.decoders.attention import *
from seq2seq.decoders.basic_decoder import *
from seq2seq.decoders.attention_decoder import *
###from seq2seq.decoders.conv_decoder_fairseq import *
###from seq2seq.decoders.conv_decoder_fairseq_bs import *
from seq2seq.decoders.conv_decoder_fairseq_topic import *
from seq2seq.decoders.conv_decoder_fairseq_bs_topic import *
