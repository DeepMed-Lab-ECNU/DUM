import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os
import nibabel as nib

import argparse
import time

from monai.transforms import Resize
from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance


from model.Universal_Model import Debiased_Universal_Medical_Segmenation_Model
from dataset.dataloader import get_inference_loader_from_folder
from utils.utils import dice_score, threshold_organ, merge_label_simple, get_key
from utils.utils import TEMPLATE, ORGAN_NAME, NUM_CLASS
from utils.utils import organ_post_process
from utils.load_pretrain import load_checkpoint_partial

torch.multiprocessing.set_sharing_strategy('file_system')

def inference(model, alpha, test_loader, args):

    model.eval()


    #################### start inference ######################
    for batch_data in test_loader:
        image = batch_data["image"].unsqueeze(0).cuda()
        name = batch_data["name"]
        print(f"processing CT image {name}...")
        # affine = batch_data["image_meta_dict"]["affine"]  # shape (4, 4), numpy array
        original_affine = batch_data["image_meta_dict"]["original_affine"]  # 原始 affine
        original_shape = batch_data["image_meta_dict"]["spatial_shape"]    # 原始 (D, H, W)

        with torch.no_grad():
            predIX = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.5, mode='gaussian',
                                            tasks_id=['99_32cls'], bias=True, sw_device="cuda", device="cuda")
            predIX = predIX.detach().cpu()#
            predX = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.5, mode='gaussian',
                                             tasks_id=['15_3DIRCADb'], bias=False, sw_device="cuda", device="cuda")
            predX = predX.detach().cpu()
            pred = (1 + alpha) * F.sigmoid(predIX) - alpha * F.sigmoid(predX)
            pred_sigmoid = torch.clip(pred, 0.0, 1.0) 

        pred_hard = threshold_organ(pred_sigmoid, threshold=args.threshold)
        pred_hard = pred_hard.cpu()
        pred_hard_post = organ_post_process(pred_hard.numpy(), TEMPLATE['99'], args.save_folder, args)
        pred_hard_post = torch.tensor(pred_hard_post)

        split_pred = merge_label_simple(pred_hard_post, ['99'])
        split_pred = split_pred.cpu().numpy()[0]

        resizer = Resize(spatial_size=original_shape, mode="nearest")
        pred_resized = resizer(split_pred)  # 输入需为 (C, D, H, W)
        pred_resized = pred_resized[0].astype(np.uint8)   # (D, H, W)

        nib_img = nib.Nifti1Image(pred_resized, affine=original_affine)
        nib.save(nib_img, os.path.join(args.save_folder, f"{name}_pred.nii.gz"))

        # save to nii.gz
        # nib_img = nib.Nifti1Image(split_pred.astype(np.uint8), affine=affine)
        # nib.save(nib_img, os.path.join(args.save_folder, f"{name}_pred.nii.gz"))


def main():
    parser = argparse.ArgumentParser()
    ## logging
    parser.add_argument('--log_name', default='./out/', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--resume1', default='./out/xxx/epoch_999.pth', help='The path resume from checkpoint')
    ## hyperparameter
    
    parser.add_argument('--input_folder', default='./demo/img', help='data root folder')
    parser.add_argument('--save_folder', default='./demo/pred', help='data root folder')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--word_embedding_path', type=str, default='./explicit_prompt.pt', help='Name of Experiment')

    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')
    parser.add_argument('--num_context', default=4, type=int, help='context prompt number in each organ')
    parser.add_argument('--threshold', default=0.5, nargs='+', type=float)

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--word_prompt_number', default=384, type=int, help='text prompt number in each organ')
    parser.add_argument('--trans_num_layers', default=4, type=int, help='transformer layer number in cross attention')

    args = parser.parse_args()

    model = Debiased_Universal_Medical_Segmenation_Model(
                    out_channels=NUM_CLASS,
                    word_prompt_path=args.word_embedding_path,
                    num_context=args.num_context,
                    word_prompt_number=args.word_prompt_number,
                    trans_num_layers=args.trans_num_layers,)

    checkpoint_main = load_checkpoint_partial(
        model=model,
        checkpoint_path=args.resume1,
        exclude_prefixes=('text_encoder',))

    print('Success load pretrained model !')
    os.makedirs(args.save_folder, exist_ok=True)
    print(f'Results will be saved in {args.save_folder}!')

    model.cuda()
    torch.backends.cudnn.benchmark = True

    test_loader = get_inference_loader_from_folder(args)
    inference(model, args.alpha, test_loader, args)

if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES=0 python -W ignore quick_inference.py \
--resume1 ..checkpoint.pth \
--input_folder ./demo/img --save_folder ./demo/pred

"""
