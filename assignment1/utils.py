import time
import nltk

import numpy as np

from copy import deepcopy
from collections import defaultdict
from scipy.sparse import csr_matrix
from tqdm import tqdm

import ipdb


def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a, 2)
    norm_b = np.linalg.norm(b, 2)

    if norm_a > 0  and norm_b > 0:
        return np.sum(a * b) / (norm_a * norm_b)
    else: 
        return 0.


def parse_analogy_data(p):
    
    with open(p) as f:
        cont = f.read()
        lines = cont.split('\n')
        lines.pop(-1)

    tasks = defaultdict(lambda: {'questions': [], 'answers': []})
    cur_task = None
    for l in lines[1:]:
        if ':' in l:
            cur_task = l.split(' ')[1]
        else:
        
            words = l.split(' ')
            words = list(map(lambda k: k.lower(), words))
            words = list(map(lambda k: k.replace('\t', ''), words))
            
            
            tasks[cur_task]['questions'].append(words[:3])
            tasks[cur_task]['answers'].append(words[3])

    print(f'Parsed analogy data (n_tasks: {len(tasks)})')
    return tasks


def get_brown_words():
    from nltk.corpus import brown
    brown_words = []
    for s in tqdm(brown.sents(), 'Get words from Brown Corpus'):
        for w in s:
            brown_words.append(w)
    
    return brown_words


'''
def construct_word_context_model(W):
    from nltk.corpus import brown
    
    # construct word context model
    sparse_m_data = defaultdict(lambda: 0)
    for s in tqdm(brown.sents(), 'Constructing word-context vector model'):
        l = len(s)

        if l > 1:
            for i in range(l-1):
                w = s[i].lower()

        if l > 0:
            for i in range(1, l):
                preceding_w, w = s[i-1].lower(), s[i].lower()
                if preceding_w in W and w in W:
                    sparse_m_data[(preceding_w, w)] += 1

    data = []
    rows = []
    cols = []
    for k, v in sparse_m_data.items():
        data.append(v)
        rows.append(W.index(k[0]))
        cols.append(W.index(k[1]))

    word_context_model = csr_matrix((data, (rows, cols)), shape=(len(W), len(W)))

    return word_context_model
'''


def construct_ppmi_model(word_context_model, W):
    # compute positive mutual information matrix
    data = []
    denominator = word_context_model.sum()
    rows, cols = word_context_model.nonzero()
    if len(rows) / len(W)**2 > 0.01:
        word_context_model = word_context_model.toarray()
        mask = word_context_model > 0
        joint_p = word_context_model / denominator
        marginal_product = (word_context_model.sum(1, keepdims=True) / denominator) * (word_context_model.sum(0, keepdims=True) / denominator)
        pmi = np.log(1e-8 + joint_p / (marginal_product+1e-8))
        

        data = pmi[mask]
        positive_pmi_model = csr_matrix((data, (rows, cols)), shape=(len(W), len(W)))
    else:
        for i, j in tqdm(zip(rows, cols), 'Constructing positive pmi model'):
            #TODO: does the value in the matrix mean the joint or conditional?
            joint_p = word_context_model[i,j] / denominator
            marginal_product = (word_context_model[i,:].sum() / denominator) * (word_context_model[:,j].sum() / denominator)
            pmi = np.log(joint_p / marginal_product)
            data.append(max(pmi, 0.))

        positive_pmi_model = csr_matrix((data, (rows, cols)), shape=(len(W), len(W)))

    return positive_pmi_model


def compute_analogy_test(analogy_data, fn, verbose=False):
    
    topK_acc = []
    for task_name, task_data in analogy_data.items():
        if verbose:
            print(f'Task: {task_name}')

        questions = task_data['questions']
        answers = task_data['answers']

        answers = np.unique(answers).tolist()
        answers_emb = np.array([fn(a).reshape(-1) for a in answers])
        

        task_topK_acc = []
        for que, ans in zip(task_data['questions'], task_data['answers']):
        
            a, b, c = que
            
            emb_a = fn(a)
            emb_b = fn(b)
            emb_c = fn(c)

            target_d = emb_c + emb_c - emb_a

            assert np.all(np.linalg.norm(answers_emb, 2, axis=1) > 0) and np.all(np.linalg.norm(target_d, 2) > 0), ipdb.set_trace()
            cosine_sim = (answers_emb * target_d.reshape(1, -1)).sum(1) / (np.linalg.norm(answers_emb, 2, axis=1) * np.linalg.norm(target_d, 2))
            pred = cosine_sim.argsort()[::-1]
            gt = answers.index(ans)

            task_topK_acc.append([gt in pred[:k] for k in [1, 5, 10]])
            
        
        task_topK_acc = np.array(task_topK_acc)
        if verbose:
            for i, k in enumerate([1, 5, 10]):
                acc = task_topK_acc[:, i].mean()
                print(f'Top-{k} Acc.: {acc:.3f}')

        topK_acc.append(task_topK_acc)


    print('All Task')
    topK_acc = np.concatenate(topK_acc, 0)
    for i, k in enumerate([1, 5, 10]):
        acc = topK_acc[:, i].mean()
        print(f'Top-{k} Acc.: {acc:.3f}')


def filter_analogy_data(analogy_data, fn):
    new_analogy_data = deepcopy(analogy_data)

    for task_name, task_data in tqdm(analogy_data.items(), 'Filter out out-of-dictionary tests'):
        new_task_data = {
            'questions': [], 
            'answers': []
        }
        for que, ans in zip(task_data['questions'], task_data['answers']):
            a, b, c = que

            if fn(a, b, c, ans):
                new_task_data['questions'].append([a, b, c])
                new_task_data['answers'].append(ans)


        new_analogy_data.pop(task_name)
        if len(new_task_data["questions"]) > 0:
            new_analogy_data[task_name]['questions'].extend(new_task_data['questions'])
            new_analogy_data[task_name]['answers'].extend(new_task_data['answers'])
        
    return new_analogy_data



def construct_weighted_word_context_model(W, window, direction='left'):
    
    assert window >= 1
    from nltk.corpus import brown
    
    # construct word context model
    sparse_m_data = defaultdict(lambda: 0)
    for s in tqdm(brown.sents(), 'Constructing word-context vector model'):
        l = len(s)

        if l > 1:
            for i in range(l):
                w = s[i].lower()
                if w in W:

                    if direction == 'left':
                        for j in range(window):
                            if j+i+1 < l:
                                next_w = s[i+j+1].lower()
                                if next_w in W:
                                    sparse_m_data[w, next_w] += 1
                    elif direction == 'left+right':
                        for j in range(window // 2 + window % 1):
                            if i+(j+1) < l:
                                next_w = s[i+(j+1)].lower()
                                if w in W and next_w in W:
                                    sparse_m_data[w, next_w] += 1

                        for j in range(window // 2):
                            if i-(j+1) > 0:
                                preceding_w = s[i-(j+1)].lower()
                                if preceding_w in W:
                                    sparse_m_data[preceding_w, w] += 1


    data = []
    rows = []
    cols = []
    for k, v in sparse_m_data.items():
        data.append(v)
        rows.append(W.index(k[0]))
        cols.append(W.index(k[1]))

    word_context_model = csr_matrix((data, (rows, cols)), shape=(len(W), len(W)))

    return word_context_model
