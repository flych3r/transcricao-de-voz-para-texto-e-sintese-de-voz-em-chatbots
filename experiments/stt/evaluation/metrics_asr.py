import re
import jiwer
import nltk
from nltk.translate import bleu_score, meteor_score

from gensim import corpora
from gensim.matutils import softcossim
from gensim.models import KeyedVectors


nltk.download('wordnet')
nltk.download('rslp')

print('loading embedding models')
emb_models = {
    'word2vec_cbow_s50': KeyedVectors.load_word2vec_format('word2vec/cbow_s50.txt'),
    'word2vec_skip_s50': KeyedVectors.load_word2vec_format('word2vec/skip_s50.txt')
}


def clean_str(x):
    return re.sub('\W', ' ', x).lower()


def cosine_similarity(reference, hypothesis, model):
    reference = reference.split()
    hypothesis = hypothesis.split()
    documents = [hypothesis, reference]
    dictionary = corpora.Dictionary(documents)

    similarity_matrix = emb_models[model].similarity_matrix(dictionary)

    hypothesis = dictionary.doc2bow(hypothesis)
    reference = dictionary.doc2bow(reference)

    return softcossim(hypothesis, reference, similarity_matrix)


def bleu(reference, hypothesis):
    references = [reference.split()]
    hypothesis = hypothesis.split()

    if len(references[0]) < 4:
        weights=(1.0, 0.0, 0.0, 0.0)
    elif len(references[0]) < 8:
        weights=(0.7, 0.2, 0.1, 0.0)
    elif len(references[0]) < 16:
        weights=(0.4, 0.3, 0.2, 0.1)
    else:
        weights=(0.25, 0.25, 0.25, 0.25)

    chencherry = bleu_score.SmoothingFunction()
    return bleu_score.sentence_bleu(
        references, hypothesis,
        smoothing_function=chencherry.method1,
        weights=weights
    )


pt_stemmer = nltk.stem.RSLPStemmer()

def meteor(reference, hypothesis):
    references = [reference]
    hypothesis = hypothesis
    return meteor_score.meteor_score(references, hypothesis, stemmer=pt_stemmer)


def wer(reference, hypothesis):
    return jiwer.wer(reference, hypothesis)
