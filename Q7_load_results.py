import nltk
import pandas as pd
import numpy as np 
import pdb
from sklearn.feature_extraction import text
from nltk.corpus import stopwords
from nltk import pos_tag
from string import punctuation

stop_words_skt = text.ENGLISH_STOP_WORDS
stop_words_eng = stopwords.words('english')
combined_stopwords = set.union(set(stop_words_eng), set(punctuation), 
                               set(stop_words_skt))

wnl = nltk.wordnet.WordNetLemmatizer()
analyzer = text.CountVectorizer().build_analyzer()

def penn2morphy(penntag):
  """Convertes Penn Treebank tags to WordNet"""
  morphy_tag = {'NN':'n', 'JJ':'a', 'VB':'v', 'RB':'r'}
  try:
    return morphy_tag[penntag[:2]]
  except:
    return 'n'

def lemmatize(list_word):
  return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag))
          for word, tag in pos_tag(list_word)]

def stem_remove_punc(text):
  return (word for word in lemmatize(analyzer(text))
          if word not in combined_stopwords and not word.isdigit())

def row(scores, params):
  d = {
     'min_score': min(scores),
     'max_score': max(scores),
     'mean_score': np.mean(scores),
     'std_score': np.std(scores),
  }
  return pd.Series({**params,**d})

#-------------------------------------------------------------------
# Question 7 output
#-------------------------------------------------------------------
if __name__ == '__main__':
  # Dataframe is merged into one
  # index 0 - 31: data w/o headers & footers
  #       32- 64: data w headers & footers
  df = pd.read_pickle("./final_result.pkl")
  rows = []
  scores = []
  params = df['params']
  for i in range(5):
    key = "split{}_test_score".format(i)
    r = df[key]        
    scores.append(r.values.reshape(len(params),1))
  all_scores = np.hstack(scores)
  for p, s in zip(params,all_scores):
    rows.append((row(s, p)))

  print("Grid Search Best Score Clf:")
  print(df[df['rank_test_score'] == 1])
  df_summary = pd.concat(rows, axis=1).T.sort_values(['mean_score'], ascending=False)
  print(df_summary.loc[df['rank_test_score'] == 1])
