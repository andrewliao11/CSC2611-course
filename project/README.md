# Instructions


## Dataset preparation
Put `ccoha1.txt` and `ccoha2.txt` from `corpus1/lemma/ccoha1.txt` and `corpus2/lemma/ccoha2.txt` to `datasets/semeval2020_ulscd_eng/full_corpus/`
```bash
$ cd datasets/semeval2020_ulscd_eng/full_corpus/
$ python split_train_val.py
```


## Train BERT on `ccoha1.txt` and `ccoha2.txt`

```bash
$ python src/train_bert.py data_dir=datasets/semeval2020_ulscd_eng vocab_size=3000 max_length=128 batch_size=256 num_train_epochs=60
```
Under the output directory, you can find `model-*` storing the checkpoints.


## Perform influence functions on the trained model

```bash
$ python src/eval_bert.py data_dir=datasets/semeval2020_ulscd_eng snapshot_dir=<OUTPUT_DIR> hessian_approx.scale=5000 hessian_approx.recursion_depth=10000
```

It will output a json file under `<OUTPUT_DIR>` with name `scale_5000-recursion_depth_1000.json`

## Generate figures in the report

```bash
$ cd src
```
Run `eval_semantic_change.ipynb`, it will generate figures: `influences_binary.pdf`, `influences_by_corpus.pdf`, `influences_grade_scatter.pdf`