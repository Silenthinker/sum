Tools
---
#### `downsampling.sh`
Downsample original datasets. For example, to downsample the GIGA dataset, specify sampling numbers for training and validation, respectively. Also, original file path and output path should be specified. The downsampling process makes sure that source and target in output files correspond to each other through a random seed.
```bash
# Make sure that the user has permission
# If permission is denied
sudo chmod 775
# then simply type:
./downsampling.sh
```

#### `preprocess_art_sum.sh`
Preprocess original article-summary datasets such as the GIGA dataset. The preprocessing includes:

* Use `mosesdecoder` to tokenize data
* and to clean all corpora
* Create character vocabulary on tokenized data
* and on non-tokenized data
* Create vocabulary for article and summary data (non-bpe)
* Apply BPE transformation to data by `subword-nmt`
* Create BPE vocabulary

This script will automatically download tools if they do not exist locally; it then applies preprocessing to all files with extensions `.art` and `.sum`.

#### `rouge_scorer.py`
A `Python` wrapper for ROUGE scorer in `Perl`.  It only contains one function `evaluate` which accepts both files and `list` and computes ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-SU4.

