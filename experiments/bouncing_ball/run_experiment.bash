#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2
  echo "Performing experiment with seed: 1"
  python bouncing_ball.py --log_dir rssm_experiment/exp_1 --posterior_samples 5 \
   --model_learning_rate 5e-5 --model_name RSSM --seed 1

   echo "Performing experiment with seed: 2"
  python bouncing_ball.py --log_dir rssm_experiment/exp_2 --posterior_samples 5 \
   --model_learning_rate 5e-5 --model_name RSSM --seed 2

   echo "Performing experiment with seed: 3"
  python bouncing_ball.py --log_dir rssm_experiment/exp_3 --posterior_samples 5 \
   --model_learning_rate 5e-5 --model_name RSSM --seed 3

   echo "Performing experiment with seed: 4"
  python bouncing_ball.py --log_dir rssm_experiment/exp_4 --posterior_samples 5 \
   --model_learning_rate 5e-5 --model_name RSSM --seed 4

   echo "Performing experiment with seed: 5"
  python bouncing_ball.py --log_dir rssm_experiment/exp_5 --posterior_samples 5 \
   --model_learning_rate 5e-5 --model_name RSSM --seed 5
echo Done!



