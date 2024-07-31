export CUDA_VISIBLE_DEVICES=2
python -u run_hyper.py  --per_slot --thresh 0.1   --i_weights 10000 --eval_lpips_alex --eval_lpips_vgg  --render_train --config configs/inward-facing/chicken.py
#--eval_ssim