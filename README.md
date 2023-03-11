# REVIVE: Regional Visual Representation Matters in Knowledge-Based Visual Question Answering (NeurIPS 2022 Accepted)

![REVIVE](https://github.com/yzleroy/REVIVE/blob/main/figures/framework.png)


**REVIVE** is a method for knowledge-based VQA task, as described in ([link to paper](https://arxiv.org/abs/2206.01201)).
>This paper revisits visual representation in knowledge-based visual question answering (VQA) and demonstrates that using regional information in a better way can significantly improve the performance. While visual representation is extensively studied in traditional VQA, it is under-explored in knowledge-based VQA even though these two tasks share the common spirit, i.e., rely on visual input to answer the question. Specifically, we observe that in most state-of-the-art knowledge-based VQA methods: 1) visual features are extracted either from the whole image or in a sliding window manner for retrieving knowledge, and the important relationship within/among object regions is neglected; 2) visual features are not well utilized in the final answering model, which is counter-intuitive to some extent. Based on these observations, we propose a new knowledge-based VQA method REVIVE, which tries to utilize the explicit information of object regions not only in the knowledge retrieval stage but also in the answering model. The key motivation is that object regions and inherent relationship are important for knowledge-based VQA. We perform extensive experiments on the standard OK-VQA dataset and achieve new state-of-the-art performance, i.e., 58.0% accuracy, surpassing previous state-of-the-art method by a large margin (+3.6%). We also conduct detailed analysis and show the necessity of regional information in different framework components for knowledge-based VQA.

Please kindly give us a <img src="https://github.com/yzleroy/REVIVE/blob/main/figures/3.png" width="32" height="32"> and cite our paper, if you think our project is helpful for your research :)

## Citation

```
@article{lin2022revive,
  title={Revive: Regional visual representation matters in knowledge-based visual question answering},
  author={Lin, Yuanze and Xie, Yujia and Chen, Dongdong and Xu, Yichong and Zhu, Chenguang and Yuan, Lu},
  journal={arXiv preprint arXiv:2206.01201},
  year={2022}
}
```

## Getting Started

### Installation
```
git clone https://github.com/yzleroy/REVIVE.git
conda create --name revive python=3.7
conda activate revive
pip install -r requirements.txt
```
### Download data
We provide the pre-processed data, it contains visual features,  implicit/explicit knowledge, 
bounding boxes, caption, tags for each sample.

Download the pre-processed data, which contains two files ("train.pkl" and "test.pkl").
```
cd REVIVE
gdown https://drive.google.com/uc?id=1kP_xeuUCAS5wqWQwuwVItDgRTAbEjUeM&export=download
unzip processed_data.zip
```
It will create a folder data:
```
REVIVE
├── ...
├── processed_data
│   ├── train.pkl
│   ├── test.pkl
└── ...
```

### Pre-trained model
|Model |Description|Accuracy(%)|Weight|Log
|  ----  | ----  | ----  | ---- | ---- | 
|REVIVE (Single)|large size and trained with visual features, explicit and implicit knowledge| 56.6 |[model.zip](https://drive.google.com/file/d/1yCEgGaxz-GNR4WS89d8ndvuB9bZmMBy_/view?usp=sharing)|[run.log](https://drive.google.com/file/d/1JaSigxV7UoVN5GvYZe0qdyfzLIczTmo7/view?usp=sharing)|

As for **model ensembling**, you can train three models with different seeds, and for each sample, 
you can get the final result with the **highest occurence frequency** among the three models' predictions,
please refer to **ensemble.py**.

### Prediction results
The prediction results of **"single"** and **"ensemble"** versions are shared:

|Model |Accuracy(%)|Download|
|  ----  | ----  | ---- |  
|REVIVE (Single)| 56.6 |[prediction_acc56.6.json](https://drive.google.com/file/d/1KjMa-XjWjLIwQBg6JhCoLUtJQ9rIMON-/view?usp=sharing)|
|REVIVE (Ensemble)| 58.1 |[prediction_acc58.1.json](https://drive.google.com/file/d/1rvIP74bfGP5aLr9x2yMn03_f0KrnG0OH/view?usp=sharing)|


### Train the model
Run the following command to start training (the A5000 training example):
```
export NGPU=4;
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 10847 train.py \
          --train_data processed_data/train.pkl \
          --eval_data processed_data/test.pkl \
          --use_checkpoint \
          --lr 0.000075 \
          --model_size large \
          --num_workers 16 \
          --optim adamw \
          --box_number 36 \
          --scheduler linear \
          --weight_decay 0.01 \
          --save_freq 2000 \
          --eval_freq 500 \
          --print_freq 100 \
          --text_maxlength 256 \
          --seed 833 \
          --name exp \
          --checkpoint_dir ./checkpoints \
          --per_gpu_batch_size 1 \
          --n_block 9 \
          --n_tags 30 \
          --n_im_context 5 \
          --n_ex_context 40 \
          --total_step 10000 \
          --warmup_step 1000
```
The whole training time is about 18 hours with 4 X A5000 GPUs.

### Test the trained model
Run the following command to start evaluation:
```
CUDA_VISIBLE_DEVICES=0 python test.py --eval_data processed_data/test.pkl \
          --model_size large \
          --per_gpu_batch_size 8 \
          --num_workers 4 \
          --box_number 36 \
          --text_maxlength 256 \
          --n_im_context 5 \
          --n_ex_context 40 \
          --name eval \
          --model_path checkpoints/exp/checkpoint/best_dev/ \
          --n_block 9 \
          --n_tags 30 \
          --write_results
```

### Test with json file
If your prediction json file is named as: "prediction.json".

Run the following command to start evaluation with json files:
```
python leaderboard_evaluation.py --pred_path prediction.json \
          --gt_path eval/mscoco_val2014_annotations.json
```

## Experimental Results

### Comparison with previous methods

![comparison](https://github.com/yzleroy/REVIVE/blob/main/figures/1.png)

### Example visualization

![visualization](https://github.com/yzleroy/REVIVE/blob/main/figures/2.png)


## Acknowledgements
Our code is built on [FiD](https://github.com/facebookresearch/FiD) which is under the [LICENSE](https://github.com/facebookresearch/FiD/blob/main/LICENSE).
