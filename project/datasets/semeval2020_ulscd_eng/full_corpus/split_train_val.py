import numpy as np
from pathlib import Path

import ipdb


seed = 123
np.random.seed(seed)


def split_train_val(train_ratio=0.8):
    dataset_dir = Path('/home/andliao/workspace/course/csc2611-course-semantic-change/project/datasets/semeval2020_ulscd_eng/full_corpus')
    
    corpus = []
    source = []
    for p in dataset_dir.glob('*.txt'):
        lines = p.read_text().split('\n')
        lowercase_lines = map(lambda l: l.lower(), lines)
        corpus.extend(lowercase_lines)
        source.extend(len(lines) * [p.name])

        
    train_mask = np.zeros(len(corpus), dtype=bool)
    train_idx = np.random.choice(len(corpus), int(train_ratio * len(corpus)), replace=False)
    train_mask[train_idx] = True
    
    train_corpus = [l for m, l, s in zip(train_mask, corpus, source) if m and len(l) > 0]
    with (dataset_dir / 'train' / 'train.txt').open('w') as f:
        for l in train_corpus:
            f.writelines(l + '\n')
        
    train_source = [s for m, l, s in zip(train_mask, corpus, source) if m and len(l) > 0]
    with (dataset_dir / 'train' / 'train_source.txt').open('w') as f:
        for l in train_source:
            f.writelines(l + '\n')
    
    
    val_corpus = [l for m, l, s in zip(train_mask, corpus, source) if not m and len(l) > 0]
    with (dataset_dir / 'val' / 'val.txt').open('w') as f:
        for l in val_corpus:
            f.writelines(l + '\n')
    
    val_source = [s for m, l, s in zip(train_mask, corpus, source) if not m and len(l) > 0]
    with (dataset_dir / 'val' / 'val_source.txt').open('w') as f:
        for l in val_source:
            f.writelines(l + '\n')
    
    
if __name__ == '__main__':
    split_train_val()
    
