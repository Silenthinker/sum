# -*- coding: utf-8 -*-
from __future__ import print_function
import sys
from pythonrouge.pythonrouge import Pythonrouge

ROUGE_path = "./pythonrouge/RELEASE-1.5.5/ROUGE-1.5.5.pl"
data_path = "./pythonrouge/RELEASE-1.5.5/data"

def evaluate(reference, summary):
  '''
  Args:
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
  '''



  # setting rouge options
  rouge = Pythonrouge(n_gram=2, ROUGE_SU4=True, ROUGE_L=True, stemming=True, stopwords=True, word_level=True, length_limit=True, length=50, use_cf=False, cf=95, scoring_formula="average", resampling=False, samples=1000, favor=True, p=0.5)
  setting_file = rouge.setting(files=False, summary=summary, reference=reference)
  res = rouge.eval_rouge(setting_file, f_measure_only=True, ROUGE_path=ROUGE_path, data_path=data_path)
  print(res)
  
if __name__ == '__main__':
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
