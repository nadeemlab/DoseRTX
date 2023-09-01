set -ex
python -m beam_angle_optim.run_bayes_optim --dataroot ./datasets/dose3d-boo --netG unet_128 --phase test --eval --model doseprediction3d --input_nc 8 --output_nc 1 --direction AtoB --dataset_mode dosepred3d --norm batch --bayes_iter 500
