export CUDA_VISIBLE_DEVICES=3
python -u run_hyper.py     --eval_ssim --i_weights 30000 --eval_lpips_alex --eval_lpips_vgg  --render_train --render_test --config configs/inward-facing/chicken.py 