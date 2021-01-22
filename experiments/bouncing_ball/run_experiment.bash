#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2
for seed in {1..5}
do
  echo "Performing experiment with seed: ${seed}"
  python bouncing_ball.py --log_dir rssm_experiment/exp_$seed --posterior_samples 5 \
   --model_learning_rate 5e-5 --model_name RSSM --seed $seed
done
echo Done!



