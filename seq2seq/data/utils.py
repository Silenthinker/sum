from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

def decode_tokens_for_blue(tokens, delimiter):
  '''
  Args:
    tokens: np.array
  Returns:
    mask: list of mask
    decoded: list of sequences

  '''
  if tokens.ndim == 1:
    T = tokens.shape[0]
    N = 1
  else:
    N, T = tokens.shape

  decoded = []
  masks = []
  for i in range(N):
    words = []
    mask = []
    for t in range(T):
      if tokens.ndim == 1:
        word = tokens[t].decode("utf-8")
      else:
        word = tokens[i, t].decode("utf-8")
      if word == 'SEQUENCE_END':
        break
      if word != '':
        words.append(word)
        mask.append(1)
    decoded.append(delimiter.join(words))
    mask.extend([0]*(T-len(mask)))
    masks.append(mask)
  return masks, decoded

def convertDict(result_dict):
  # Convert dict of lists to list of dicts
  result_dicts = [
      dict(zip(result_dict, t)) for t in zip(*result_dict.values())
  ]
  return result_dicts

def flattenList(l):
  '''
  Flatten l: [[1],[2], [3]] to [1,2,3]
  '''
  return [x[0] for x in l]