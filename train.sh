export NGPU=4;
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port 10847 train.py \
	--train_data processed_data/train.pkl \
	--eval_data processed_data/test.pkl \
	--use_checkpoint \
	--lr 0.000075 \
	--model_size large \
	--num_workers 8 \
	--optim adamw \
	--box_number 36 \
	--scheduler linear \
	--weight_decay 0.01 \
	--save_freq 2000 \
	--eval_freq 1000 \
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

