python process_data.py --n_c 33 --r_type pca
python train.py --ep 32 --batch 512 --hid 100

python val_eval_comb.py --batch 512 --hid 100 --r_type pcal1 --th 0.045
python submission.py --batch 512 --hid 100 --r_type pcal1 --th 0.045

python val_eval_comb.py --batch 512 --hid 100 --r_type pcal2 --th 0.045 --r_w 3
python submission.py --batch 512 --hid 100 --r_type pcal2 --th 0.045 --r_w 3
