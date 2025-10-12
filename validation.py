import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import glob
import os

import argparse
import time
import nibabel as nib

from monai.inferers import sliding_window_inference

from model.Universal_Model import Debiased_Universal_Medical_Segmenation_Model
from dataset.dataloader import get_loader
from utils.utils import dice_score, threshold_organ, merge_label, get_key
from utils.utils import TEMPLATE, ORGAN_NAME, NUM_CLASS
from utils.utils import organ_post_process
from utils.load_pretrain import load_checkpoint_partial

torch.multiprocessing.set_sharing_strategy('file_system')
 

def validation(model, alpha, ValLoader, val_transforms, args):
    dataset_list_strs = '_'.join([str(l) for l in args.dataset_list])
    save_dir = args.log_name + f'/val_result'


    model.eval()
    dice_list = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS)) 

    columns = ['name'] + [str(i) for i in range(32)]
    
    #################### start inference ######################
    for index, batch in enumerate(tqdm(ValLoader)):
        image, label, name = batch["image"].cuda(), batch["post_label"], batch["name"]
        with torch.no_grad():
            predIX = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.25, mode='gaussian',
                                            tasks_id=name, bias=True, sw_device="cuda", device="cuda")
            predIX = predIX.detach().cpu()
            predX = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.25, mode='gaussian',
                                            tasks_id=name, bias=False, sw_device="cuda", device="cuda")
            predX = predX.detach().cpu()

            pred = (1 + alpha) * F.sigmoid(predIX) - alpha * F.sigmoid(predX)
            pred_sigmoid = torch.clip(pred, 0.0, 1.0)

        pred_hard = threshold_organ(pred_sigmoid, threshold=args.threshold)
        pred_hard = pred_hard.cpu()

        B = pred_hard.shape[0]
        for b in range(B):
            content = 'case%s| '%(name[b])
            template_key = get_key(name[b])
            organ_list = TEMPLATE[template_key]
            pred_hard_post = organ_post_process(pred_hard.numpy(), organ_list, args.log_name+'/'+name[0].split('/')[0]+'/'+name[0].split('/')[-1],args)
            pred_hard_post = torch.tensor(pred_hard_post)

            for organ in organ_list:
                if torch.sum(label[b,organ-1,:,:,:].cuda()) != 0:
                    dice_organ, recall, precision, specificity, iou = dice_score(pred_hard_post[b,organ-1,:,:,:].cuda(), label[b,organ-1,:,:,:].cuda(), spe_sen=True)
                    dice_list[template_key][0][organ-1] += dice_organ.item()
                    dice_list[template_key][1][organ-1] += 1
                    content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice_organ.item())

    #################### merge all cases' metric results ######################
    ave_organ_dice = np.zeros((2, NUM_CLASS))

    with open(f"{save_dir}/val_result_epoch{args.epoch}.txt", 'w') as f:
        for key in TEMPLATE.keys():
            organ_list = TEMPLATE[key]
            content = 'Dice_Task%s| ' % (key)

            for organ in organ_list:
                dice = dice_list[key][0][organ - 1] / dice_list[key][1][organ - 1]
                content += '%s: %.4f, ' % (ORGAN_NAME[organ - 1], dice)
                ave_organ_dice[0][organ - 1] += dice_list[key][0][organ - 1]
                ave_organ_dice[1][organ - 1] += dice_list[key][1][organ - 1]

            f.write(content)
            f.write('\n')

        content_dice = 'Average Dice | '

        for i in range(NUM_CLASS):
            content_dice += '%s: %.4f, ' % (ORGAN_NAME[i], ave_organ_dice[0][i] / ave_organ_dice[1][i])

        f.write(content_dice)
        f.write('\n')
        f.write('%s: %.4f, ' % ('average', np.mean(ave_organ_dice[0] / ave_organ_dice[1])))
        f.write('\n')
        
        mean_dice = np.mean(ave_organ_dice[0] / ave_organ_dice[1])
        return mean_dice

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='./out/', help='The path resume from checkpoint')
    ## model load
    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')

    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_10_inner', 'PAOT_123457891213']) # 'PAOT', 'felix'
    ### please check this argment carefull
    ### PAOT: include PAOT_123457891213 and PAOT_10
    ### PAOT_123457891213: include 1 2 3 4 5 7 8 9 12 13
    ### PAOT_10_inner
    ### external15
    ### external16_totalseg
    ### external17
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
    parser.add_argument('--num_samples', default=1, type=int, help='sample number in each ct')
    parser.add_argument('--word_embedding_path', type=str, default='./explicit_prompt.pt', help='Name of Experiment')
    parser.add_argument('--phase', default='test', help='train or validation or test')

    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0., type=float, help='The percentage of cached data in total')
    parser.add_argument('--num_context', default=4, type=int, help='context prompt number in each organ')
    parser.add_argument('--threshold_organ', default=None, nargs='+', type=str)
    parser.add_argument('--threshold', default=0.5, nargs='+', type=float)

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--word_prompt_number', default=384, type=int, help='text prompt number in each organ')
    parser.add_argument('--trans_num_layers', default=4, type=int, help='transformer layer number in cross attention')

    parser.add_argument('--start_epoch', default=500, type=int, help='Number of start epoches')
    parser.add_argument('--end_epoch', default=1000, type=int, help='Number of end epoches')

    args = parser.parse_args()

    model = Debiased_Universal_Medical_Segmenation_Model(
                    out_channels=NUM_CLASS,
                    word_prompt_path=args.word_embedding_path,
                    num_context=args.num_context,
                    word_prompt_number=args.word_prompt_number,
                    trans_num_layers=args.trans_num_layers,).cuda()

    store_path_root = os.path.join(args.log_name, 'epoch_***.pth')
    df_dice = pd.DataFrame(columns=['epoch', 'val_dice']) # save each epoch dice results
    save_dir = args.log_name + f'/val_result'
    os.makedirs(save_dir, exist_ok=True)
    
    for index, store_path in enumerate(sorted(glob.glob(store_path_root), key=lambda x: int(x.split('/')[-1].split('_')[-1][:-4]))):
        current_epoch = int(store_path.split('/')[-1].split('_')[-1][:-4])
        if current_epoch >= args.start_epoch and current_epoch <= args.end_epoch:
            print(f"start validation epoch {current_epoch}!!!")
            checkpoint_main = load_checkpoint_partial(
                model=model,
                checkpoint_path=store_path,
                exclude_prefixes=('text_encoder',))
            print('Success load pretrained model !')
            args.epoch = store_path.split('_')[-1][:-4]

            torch.backends.cudnn.benchmark = True
            test_loader, val_transforms = get_loader(args)
            mean_dice = validation(model, args.alpha, test_loader, val_transforms, args)

            df_dice.loc[index, 'epoch'] = current_epoch
            df_dice.loc[index, 'val_dice'] = mean_dice
            df_dice.to_csv(f"{save_dir}/validation_dice.csv")

    max_epoch = df_dice.loc[df_dice['val_dice'].idxmax(), 'epoch']
    print(f"The best checkpoint on the validation set is at epoch {max_epoch} !")

if __name__ == "__main__":
    main()

"""
CUDA_VISIBLE_DEVICES=0 python -W ignore validation.py \
--log_name ./out/xxx --start_epoch 500 --end_epoch 1000 \
--data_root_path ../Datasets/ --dataset_list PAOT_10_inner \
--dataset_list PAOT_10_inner --num_workers 8
"""