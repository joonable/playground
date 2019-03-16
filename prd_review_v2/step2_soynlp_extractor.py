from soynlp.noun import LRNounExtractor_v2
from soynlp.word import WordExtractor
import pandas as pd
import re
from collections import Counter
from itertools import chain

WORDS_THRESHOLD = 2
NOUNS_THRESHOLD = 0.9
TOKENS_THRESHOLD = 0.15
USERS_THRESHOLD = 0.15

df = pd.read_pickle('./data/df_raw.pkl')

corpus = df.text.tolist()

print(''' nouns extractor ''')
noun_extractor = LRNounExtractor_v2(
    verbose=True,
    extract_compound=True
)

noun_extractor.train(corpus)
nouns = noun_extractor.extract(min_noun_frequency=100, min_noun_score=0.3)
nouns_list = list()
for k, v in nouns.items():
    word = k
    score = v.score
    freq = v.frequency

    temp = dict()
    temp['noun'] = word.lower()
    temp['score'] = score
    temp['freq'] = freq

    nouns_list.append(temp)

df_nouns = pd.DataFrame(nouns_list)
df_nouns = df_nouns.sort_values(by=['score'], ascending=False)
nouns_candidates_list = df_nouns.loc[df.score > NOUNS_THRESHOLD].noun.tolist()
print('nouns_candidates_list : {}\n'.format(len(nouns_candidates_list)))

print(''' words extractor ''')
word_extractor = WordExtractor(
    min_frequency=100,
    min_cohesion_forward=0.05,
    min_right_branching_entropy=0.0
)

word_extractor.train(corpus)
words = word_extractor.extract()
words = {k: v for k, v in words.items() if len(k) > 1}
words_list = list()
for k, v in words.items():
    temp = dict()
    cohesion = v.cohesion_forward
    branching_entropy = v.left_branching_entropy
    left_freq = v.leftside_frequency
    right_freq = v.rightside_frequency
    score = cohesion * branching_entropy

    temp['word'] = k.lower()
    temp['cohesion'] = cohesion
    temp['branching_entropy'] = branching_entropy
    temp['left_freq'] = left_freq
    temp['right_freq'] = right_freq
    temp['score'] = cohesion * branching_entropy
    words_list.append(temp)

df_words = pd.DataFrame(words_list)
df_words = df_words.sort_values(by=['score'], ascending=False)
words_candidates_list = df_words.loc[df_words.score > WORDS_THRESHOLD].word.tolist()
print('words_candidates_list : {}\n'.format(len(words_candidates_list)))

print(''' tokens extractor ''')
# corpus_list = corpus.split()xx
corpus_list = [text.split() for text in corpus]
# corpus_list = [text.split() for text in corpus]
tokens_list = []

subfix = list('을를이가는은에도와과로라')
subfix_2 = ['하게', '하고', '해서', '이고', '처럼', '하면', '하며', '으로', '에서', '까지', '이라', '인데', '라서', '이나']
for token in chain(*corpus_list):
    if len(token) == 1 or token[-1] in ['요', '다']:
        continue
    elif token[-1] in subfix and len(token) > 2:
        tokens_list.append(token[:-1])
    elif token[-2:] in subfix_2 and len(token) > 3:
        tokens_list.append(token[:-2])
    else:
        tokens_list.append(token)
del corpus_list

# tokens_list = [token for token in tokens_list if token[-1] not in ['요', '다']]
tokens_count = Counter(tokens_list)
tokens_candidates_list = [tup[0].lower() for tup in tokens_count.most_common( int(TOKENS_THRESHOLD * len(tokens_count)) )]
print('tokens_candidates_list : {}\n'.format(len(tokens_candidates_list)))


print(''' users extractor ''')
# user_dict : parsing 결과만 고려
users_list = pd.read_pickle('./data/sr_dict_parsed_result.pkl').tolist()
users_list = [user for user in users_list]
users_count = Counter(users_list)
users_candidates_list = [tup[0].lower() for tup in users_count.most_common( int(USERS_THRESHOLD * len(users_count)) )]
print('users_candidates_list : {}\n'.format(len(users_candidates_list)))


print(''' terms extractor ''')
term_list = df.term_vectors.tolist()
term_list = [term for term in chain(*term_list)]
print('term_list : {}\n'.format(len(term_list)))

candidates_list = nouns_candidates_list + words_candidates_list + tokens_candidates_list + users_candidates_list
candidates_list = list(set(candidates_list) - set(term_list))
print('candidates_list : {}\n'.format(len(candidates_list)))

sr_candidates = pd.Series(candidates_list).sort_values()
sr_candidates.to_csv('./data/feature_candidates.csv', index=False)
