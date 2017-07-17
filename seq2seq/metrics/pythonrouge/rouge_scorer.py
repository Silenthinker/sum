# -*- coding: utf-8 -*-
from __future__ import print_function

import sys
import os
import argparse

from pythonrouge.pythonrouge import Pythonrouge

root = os.path.dirname(os.path.abspath(__file__))
ROUGE_path = os.path.join(root, "pythonrouge/RELEASE-1.5.5/ROUGE-1.5.5.pl")
data_path = os.path.join(root, "pythonrouge/RELEASE-1.5.5/data")


def evaluate(reference, summary, files=False):
  '''
  Args:
    if files == False:
      # summary: double list
        summary = [[summaryA_sent1, summaryA_sent2], [summaryB_sent1, summaryB_sent2]]
      # reference: triple list
        reference = [[[summaryA_ref1_sent1, summaryA_ref1_sent2], [summaryA_ref2_sent1, summaryA_ref2_sent2]],
                     [[summaryB_ref1_sent1, summaryB_ref1_sent2], [summaryB_ref2_sent1, summaryB_ref2_sent2]]
    else:
      # summary: summary_path 
      # reference: reference_path
        # Directory format sample
        1 system summary and 4 reference summaries.
        - system summary
        ./summary_path/summaryA.txt
        - reference summary
        ./reference_path/summaryA.1.txt
        ./reference_path/summaryA.2.txt
        ./reference_path/summaryA.3.txt
        ./reference_path/summaryA.4.txt
        In first N strings, reference summaries should have same file name as the system output file.
        delete: If True, the rouge setting file(setting.xml) is deleted.
                If False, rouge setting file is saved in current directory.
  '''



  # setting rouge options
  rouge = Pythonrouge(n_gram=2, ROUGE_SU4=True, ROUGE_L=True, stemming=True, stopwords=True, word_level=True, length_limit=True, length=50, use_cf=False, cf=95, scoring_formula="average", resampling=False, samples=1000, favor=True, p=0.5)
  if files:
    setting_file = rouge.setting(files=True, summary_path=summary, reference_path=reference)  
  else:
    setting_file = rouge.setting(files=False, summary=summary, reference=reference)
  res = rouge.eval_rouge(setting_file, f_measure_only=True, ROUGE_path=ROUGE_path, data_path=data_path)
  return res
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-ref_dir", type=str, required=True, help="reference directory")
  parser.add_argument("-sum_dir", type=str, required=True, help="summary directory")
  args = parser.parse_args()

  print(evaluate(args.ref_dir, args.sum_dir, files=True))
  '''
  summary = [["Great location, very good selection of food for breakfast buffet.",
              "Stunning food, amazing service.",
              "The food is excellent and the service great."],
              ["The keyboard, more than 90% standard size, is just large enough .",
              "Surprisingly readable screen for the size .",
              "Smaller size videos   play even smoother ."]]
  reference = [[["Food was excellent with a wide range of choices and good services.", "It was a bit expensive though."],
             ["Food can be a little bit overpriced, but is good for a hotel."],
             ["The food in the hotel was a little over priced but excellent in taste and choice.",
             "There were also many choices to eat in the near vicinity of the hotel."],
             ["The food is good, the service great.",
             "Very good selection of food for breakfast buffet."]
             ],
             [
             ["The size is great and allows for excellent portability.",
             "Makes it exceptionally easy to tote around, and the keyboard is fairly big considering the size of this netbook."],
             ["Size is small and manageable.",
             "Perfect size and weight.",
             "Great size for travel."],
             ["The keyboard is a decent size, a bit smaller then average but good.",
             "The laptop itself is small but big enough do do things on it."],
             ["In spite of being small it is still comfortable.",
             "The screen and keyboard are well sized for use"]
             ]
             ]
  evaluate(reference, summary)
  '''