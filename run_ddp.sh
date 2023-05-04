CUDA_VISIBLE_DEVICES=0,1,2,3 
torchrun \
    --standalone\
    --nnodes 1 \
    --nproc_per_node 4 \
    train_parallel_smart.py  \
    --model_name 'hfl/chinese-macbert-large' --batch_size 8 --epochs 50 --lr 2e-5 --seed 1024 \
    --hidden_size 1024 --question 'SA' --experiment_name 'smart_ls' --early_stop 100 --dropout 0.1
