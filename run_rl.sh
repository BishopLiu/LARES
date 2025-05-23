

GPU_ID=3
DATASET=Instruments
MODEL=LARES
DATA_PATH=./dataset


python rl_train.py \
  -d $DATASET \
  -m $MODEL \
  --gpu_id=$GPU_ID \
  --data_path=$DATA_PATH \
  --pretrain_model_path=./saved/LARES-SL-Instruments.pth \
  --learning_rate=0.0005 \
  --n_pre_layers=2 \
  --n_core_layers=2 \
  --attn_dropout_prob=0.1 \
  --hidden_dropout_prob=0.1 \
  --mean_recurrence=4 \
  --train_batch_size=400 \
  --group_num=4 \
  --beta=1.0 \
  --k=10 \
  --reward_metric=Recall \
  --state_std=1.0 \
  --state_scale=3

