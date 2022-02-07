import time
import nltk

import numpy as np
import pickle as pkl

from gensim.models import KeyedVectors
from collections import Counter
from sklearn.decomposition import PCA
from scipy import stats
from tqdm import tqdm

from utils import *

import ipdb


rg65 = {
    ('cord', 'smile'): 0.02, 
    ('rooster', 'voyage'): 0.04, 
    ('noon', 'string'): 0.04, 
    ('fruit', 'furnace'): 0.05, 
    ('autograph', 'shore'): 0.06, 
    ('automobile', 'wizard'): 0.11, 
    ('mound', 'stove'): 0.14, 
    ('grin', 'implement'): 0.18, 
    ('asylum', 'fruit'): 0.19, 
    ('asylum', 'monk'): 0.39, 
    ('graveyard', 'madhouse'): 0.42, 
    ('glass', 'magician'): 0.44, 
    ('boy', 'rooster'): 0.44, 
    ('cushion', 'jewel'): 0.45, 
    ('monk', 'slave'): 0.57, 
    ('asylum', 'cemetery'): 0.79, 
    ('coast', 'forest'): 0.85, 
    ('grin', 'lad'): 0.88, 
    ('shore', 'woodland'): 0.9, 
    ('monk', 'oracle'): 0.91, 
    ('boy', 'sage'): 0.96, 
    ('automobile', 'cushion'): 0.97, 
    ('mound', 'shore'): 0.97, 
    ('lad', 'wizard'): 0.99, 
    ('forest', 'graveyard'): 1., 
    ('food', 'rooster'): 1.09, 
    ('cemetery', 'woodland'): 1.18, 
    ('shore', 'voyage'): 1.22, 
    ('bird', 'woodland'): 1.24, 
    ('coast', 'hill'): 1.26, 
    ('furnace', 'implement'): 1.37, 
    ('crane', 'rooster'): 1.41, 
    ('hill', 'woodland'): 1.48, 
    ('car', 'journey'): 1.55, 
    ('cemetery', 'mound'): 1.69, 
    ('glass', 'jewel'): 1.78, 
    ('magician', 'oracle'): 1.82, 
    ('crane', 'implement'): 2.37, 
    ('brother', 'lad'): 2.41, 
    ('sage', 'wizard'): 2.46, 
    ('oracle', 'sage'): 2.61, 
    ('bird', 'crane'): 2.63, 
    ('bird', 'cock'): 2.63, 
    ('food', 'fruit'): 2.69, 
    ('brother', 'monk'): 2.74, 
    ('asylum', 'madhouse'): 3.04, 
    ('furnace', 'stove'): 3.11, 
    ('magician', 'wizard'): 3.21, 
    ('hill', 'mound'): 3.29, 
    ('cord', 'string'): 3.41, 
    ('glass', 'tumbler'): 3.45, 
    ('grin', 'smile'): 3.46, 
    ('serf', 'slave'): 3.46, 
    ('journey', 'voyage'): 3.58, 
    ('autograph', 'signature'): 3.59, 
    ('coast', 'shore'): 3.6, 
    ('forest', 'woodland'): 3.65, 
    ('implement', 'tool'): 3.66, 
    ('cock', 'rooster'): 3.68, 
    ('boy', 'lad'): 3.82, 
    ('cushion', 'pillow'): 3.84, 
    ('cemetery', 'graveyard'): 3.88, 
    ('automobile', 'car'): 3.92, 
    ('midday', 'noon'): 3.94, 
    ('gem', 'jewel'): 3.94
}


def compute_cosine_similarity_rg65(fn):
    sim = []

    for k, v in tqdm(rg65.items(), 'Computing cosine similarity'):
        a, b = k
        s_a = fn(a)
        s_b = fn(b)
        if s_a is not None and s_b is not None:
            cos = cosine_similarity(s_a, s_b)
        else: 
            cos = 0.

        sim.append(cos)

    return sim


def get_human_sim():
    human_sim = [v for k, v in rg65.items()]
    return human_sim


def construct_dictionary(corpus_words, N):
    # construct dictionary
    W = Counter(corpus_words).most_common(N)
    W = [w[0].lower() for w in W]

    for k, v in rg65.items():
        a, b = k

        if a.lower() not in W:
            W.append(a.lower())
        
        if b.lower() not in W:
            W.append(b.lower())
    
    return W
            


def q2():

    # norm change
    def __norm(a, b, p):
        norm_change = np.linalg.norm(a - b, p, axis=-1)
        return norm_change

    def __neg_cosine(a, b):
        cos_change = (a * b).sum(-1) / (np.linalg.norm(a, 2, axis=-1)*np.linalg.norm(b, 2, axis=-1))
        return -cos_change


    def __report_top20_bot20(embeddings, fn):

        a = np.zeros_like(embeddings[:, 0, :])
        fill_mask = np.zeros(embeddings.shape[0], dtype=bool)
        for i in range(embeddings.shape[1]):
            mask = np.linalg.norm(embeddings[:, i, :], 2, -1) > 0
            a[mask & ~fill_mask] = embeddings[mask & ~fill_mask, i, :]
            fill_mask = mask | fill_mask
            if np.all(fill_mask):
                break

        b = embeddings[:, -1, :]

        change = fn(a, b)
        idx = np.argsort(change)
        most_chaning = idx[::-1][:20]
        least_chaning = idx[:20]
        return idx, most_chaning, least_chaning


    def _step2(data, words, embeddings):

        print('l1-norm')
        l1_idx, most_chaning, least_chaning = __report_top20_bot20(embeddings, fn=lambda a, b: __norm(a, b, 1))
        print(', '.join(words[most_chaning]))
        print(', '.join(words[least_chaning]))

        print('l-\infty norm')
        linf_idx, most_chaning, least_chaning = __report_top20_bot20(embeddings, fn=lambda a, b: __norm(a, b, np.inf))
        print(', '.join(words[most_chaning]))
        print(', '.join(words[least_chaning]))

        print('neg cosine')
        neg_cos_idx, most_chaning, least_chaning = __report_top20_bot20(embeddings, fn=__neg_cosine)
        print(', '.join(words[most_chaning]))
        print(', '.join(words[least_chaning]))

        print(f'Pearson correlation between l1-norm and linf-norm: {stats.pearsonr(l1_idx, linf_idx)}')
        print(f'Pearson correlation between l1-norm and neg-cos-norm: {stats.pearsonr(l1_idx, neg_cos_idx)}')
        print(f'Pearson correlation between l1-norm and neg-cos-norm: {stats.pearsonr(linf_idx, neg_cos_idx)}')
        print(f'Pearson correlation between l1-norm and neg-cos-norm: {stats.pearsonr(neg_cos_idx, neg_cos_idx)}')

        
    def _step3(data, words, embeddings):
        
        l1_idx, _, _ = __report_top20_bot20(embeddings, fn=lambda a, b: __norm(a, b, 1))
        linf_idx, _, _ = __report_top20_bot20(embeddings, fn=lambda a, b: __norm(a, b, np.inf))
        neg_cos_idx, _, _ = __report_top20_bot20(embeddings, fn=__neg_cosine)

        stable_words = ['hundreds', 'thousands', 'millions', 
                'january', 'february', 'april', 'june', 'july', 'august', 'september', 'october', 'november', 'december']

        print(f'Stable words: {", ".join(stable_words)}')
        stable_words_mask = np.any(words.reshape(-1, 1) == np.array(stable_words).reshape(1, -1), 1)

        stable_words_idx = np.where(stable_words_mask)[0]

        print(f'Average rank of stable words with l1-norm: {l1_idx[stable_words_idx].mean():.3f}')
        print(f'Average rank of stable words with linf-norm: {linf_idx[stable_words_idx].mean():.3f}')
        print(f'Average rank of stable words with neg_cos-norm: {neg_cos_idx[stable_words_idx].mean():.3f}')

    def _step4(data, words, embeddings):
        neg_cos_idx, _, _ = __report_top20_bot20(embeddings, fn=__neg_cosine)
        top3_changing = neg_cos_idx[::-1][:3]
        for i in top3_changing:
            print(words[i])
            changes = [__neg_cosine(embeddings[i, t, :], embeddings[i, t+1, :]) for t in range(embeddings.shape[1]-1)]
            print(changes)
                


    data = pkl.load(open('data/embeddings/data.pkl', 'rb'))
    words = np.array(data['w'])
    decades = data['d']
    embeddings = np.array(data['E'])

    #_step2(data, words, embeddings)
    #_step3(data, words, embeddings)
    _step4(data, words, embeddings)


    
def q1():

    def step3():

        ## Word2vec
        model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)
        word2vec_sim = compute_cosine_similarity_rg65(fn=lambda k: model[k])

        pearsonr = stats.pearsonr(word2vec_sim, get_human_sim())
        print(f'The Pearson correlation between word2vec-based and human similarity: {pearsonr[0]}')


        brown_words = get_brown_words()
        W = construct_dictionary(brown_words, 5000)
        print(f'Dictionary size: {len(W)}')

        ## word-context model
        word_context_model = construct_weighted_word_context_model(W, window=1)
        word_conetxt_model_sim = compute_cosine_similarity_rg65(fn=lambda k: word_context_model[W.index(k.lower()), :].toarray().reshape(-1) if k.lower() in W else None)
        pearsonr = stats.pearsonr(word_conetxt_model_sim, get_human_sim())
        print(f'The Pearson correlation between word context model-based and human similarity: {pearsonr[0]}')

        ## positive pmi model
        positive_pmi_model = construct_ppmi_model(word_context_model, W)
        ppmi_sim = compute_cosine_similarity_rg65(fn=lambda k: positive_pmi_model[W.index(k.lower()), :].toarray().reshape(-1) if k.lower() in W else None)
        pearsonr = stats.pearsonr(ppmi_sim, get_human_sim())
        print(f'The Pearson correlation between positive point-wise mutual information-based and human similarity: {pearsonr[0]}')

        ## latent semantic model
        for n_components in [10, 100, 300]:
            print('-'*30)
            print(f'PCA components: {n_components}')
            
            t1 = time.time()
            pca = PCA(n_components=n_components, random_state=123)
            pca.fit(positive_pmi_model.toarray())
            print(f'PCA fit costs {time.time() - t1} sec')

            PCA_positive_pmi_model = pca.transform(positive_pmi_model.toarray())
            pca_ppmi_sim = compute_cosine_similarity_rg65(fn=lambda k: PCA_positive_pmi_model[W.index(k.lower()), :].reshape(-1) if k.lower() in W else None)

            pearsonr = stats.pearsonr(pca_ppmi_sim, get_human_sim())
            print(f'The Pearson correlation between pca-ppmi ({n_components}) and human similarity: {pearsonr[0]}')

    def step4():
        analogy_data = parse_analogy_data('data/word-test.v1.txt')

        word2vec_model = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

        brown_words = get_brown_words()
        W = construct_dictionary(brown_words, 5000)
        print(f'Dictionary size: {len(W)}')


        # filter questions
        analogy_data = filter_analogy_data(analogy_data, fn=lambda a, b, c, ans: a in W and b in W and c in W and ans in W and a in word2vec_model and b in word2vec_model and c in word2vec_model and ans in word2vec_model)
        

        print('Random model')
        compute_analogy_test(analogy_data, fn=lambda k: np.random.rand(2))

        print('Word2vec model')
        compute_analogy_test(analogy_data, fn=lambda k: word2vec_model[k] if k in word2vec_model else None)


        #word_context_model = construct_word_context_model(W)
        word_context_model = construct_weighted_word_context_model(W, window=1)
        compute_analogy_test(analogy_data, fn=lambda k: word_context_model[W.index(k)].toarray() if k in W else None)

        positive_pmi_model = construct_ppmi_model(word_context_model, W)
        compute_analogy_test(analogy_data, fn=lambda k: positive_pmi_model[W.index(k)].toarray() if k in W else None)

        pca = PCA(n_components=300, random_state=123)
        pca.fit(positive_pmi_model.toarray())
        PCA_positive_pmi_model = pca.transform(positive_pmi_model.toarray())

        print('LSA(300) model')
        compute_analogy_test(analogy_data, fn=lambda k: PCA_positive_pmi_model[W.index(k)] if k in W else None)
        
    step3()
    step4()


if __name__ == '__main__':
    #q1()
    q2()

