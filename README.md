## CoIn_DialogRE
## 0. Package Description
```
├─ data/: raw data and preprocessed data
    ├─ train.json
    ├─ dev.json
    ├─ test.json
    ├─ entity_type_id.json
    ├─ speaker_vocab_id.json
    ├─ vocab.txt: bert vocab file, we add the new-introduced special tokens
├─ logs/: save the log files
├─ model/: save the optimal model file and prediction results
├─ src/: source codes
    ├─ attention.py 
    ├─ data_utils.py: utils for processing data
    ├─ dataset.py
    ├─ embeddings.py: generate entity type/ utterance embedding
    ├─ model.py
    ├─ main.py: main file to run the model
├─ readme.md
```

## 1. Environments
We conducted experiments on a sever with two GeForce GTX 1080Ti GPU.
- python      (3.6.5)  
- cuda        (11.0)  
- CentOS Linux release 7.8.2003  (Core)

## 2. Dependencies
- torch                    (1.2.0)
- transformers             (2.0.0)
- pytorch-transformers     (1.2.0)
- numpy                    (1.19.2)

## 3. Preparation
### 3.1 Download the pre-trained language models.
- Download the bert-base-uncase model.
### 3.2 Add the special token id into the vocab.txt
- Inspired by the resource paper, we add the newly-introduced special tokens to indicate the speakers. (Replacing [unused1]..[unsued10] with speaker1..speaker10).
- You can replace the original vocab.txt with our file (in './data/vocab.txt')

## 4. Training
If you want to reproduce our results, please follow our hyper-parameter settings and run the code with the following command.
```
CUDA_VISIBLE_DEVICES=0,1 nohup python -m torch.distributed.launch --nproc_per_node=2 main.py --bert_path {your_bert_path}
```

## 5. Evaluating
You also can evaluate our model without training. Please download the released model. [model](https://drive.google.com/u/0/uc?id=1B4FgMQjJAg5i-6R3l1XBt55xhlDjdaK1&export=download)
```
python evaluate.py --bert_path {your_bert_path} --optimal_model_path {released_model_path}
```

## Citation
Thank you for your interests in our paper, if you have any problem, please feel free to contact me. (longxinwei19@mails.ucas.ac.cn)
```
@inproceedings{DBLP:conf/ijcai/LongNL21,
  author    = {Xinwei Long and
               Shuzi Niu and
               Yucheng Li},
  title     = {Consistent Inference for Dialogue Relation Extraction},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on Artificial
               Intelligence, {IJCAI} 2021, Virtual Event / Montreal, Canada, 19-27
               August 2021},
  pages     = {3885--3891},
  year      = {2021},
  url       = {https://doi.org/10.24963/ijcai.2021/535}
}
```
