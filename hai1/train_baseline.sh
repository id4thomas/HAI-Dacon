#!/bin/bash
python train.py --ep 32 --batch 512 --hid 100 --m_name baseline
python eval.py --batch 512 --hid 100 --m_name baseline
# python plot_latents.py