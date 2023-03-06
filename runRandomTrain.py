import os
if __name__ == '__main__':
    for i in range(1):

        # planName = 'manualBeamMomentLossWeight0_001_run{}'.format(i+1)
        planName = 'echoBeamMomLossW0_1'
        # os.system('python3 train.py --dataroot ./datasets/msk-echo-3d-dvh-beamlet-sparse-separate-ptv --netG unet_128 --name {} --model doseprediction3d --direction AtoB --lambda_L1 1 --dataset_mode dosepred3d --norm batch --batch_size 1 --pool_size 0 --display_port 8097 --lr 0.0002 --input_nc 8 --output_nc 1 --display_freq 10 --print_freq 1 --gpu_ids 1'.format(planName))
        # os.system('python3 test.py --dataroot ./datasets/msk-echo-3d-dvh-beamlet-sparse-separate-ptv --netG unet_128 --name {} --phase test --mode eval --model doseprediction3d --input_nc 8 --output_nc 1 --direction AtoB --dataset_mode dosepred3d --norm batch'.format(planName))
        # os.system('python3 ./openkbp-stats/dvh-stats-open-kbp.py --planName {}'.format(planName))

        # os.system('python3 train.py --dataroot ./datasets/msk-manual-3d-dvh-beamlet-dense-separate-ptv --netG unet_128 --name {} --model doseprediction3d --direction AtoB --lambda_L1 1 --dataset_mode dosepred3d --norm batch --batch_size 1 --pool_size 0 --display_port 8097 --lr 0.0002 --input_nc 8 --output_nc 1 --display_freq 10 --print_freq 1 --gpu_ids 1'.format(planName))
        # os.system('python3 test.py --dataroot ./datasets/msk-manual-3d-dvh-beamlet-dense-separate-ptv --netG unet_128 --name {} --phase test --mode eval --model doseprediction3d --input_nc 8 --output_nc 1 --direction AtoB --dataset_mode dosepred3d --norm batch'.format(planName))
        os.system('python3 ./openkbp-stats/dvh-stats-open-kbp.py --planName {}'.format(planName))
        # os.system('python3 ./openkbp-stats/extra-stats-for-paper.py --planName {}'.format(planName))

        # os.system('python3 test.py --dataroot /nadeem_lab/Gourav/datasets/anonymous --netG unet_128 --name {} --phase test --mode eval --model doseprediction3d --input_nc 8 --output_nc 1 --direction AtoB --dataset_mode dosepred3d --norm batch'.format(planName))
        # os.system('python3 ./openkbp-stats/dvh-stats-open-kbp.py --planName {}'.format(planName))
        # os.system('python3 ./statistics/compute_dvh_stats.py --planName {}'.format(planName))
