CUDA_VISIBLE_DEVICES=1 \
python train_smart.py --model_name 'hfl/chinese-macbert-large' --batch_size 8 --epochs 50 --lr 2e-5 --seed 1024 \
       --hidden_size 1024 --question 'SA' --experiment_name '1_smart' --early_stop 100 --dropout 0 --warmup 3