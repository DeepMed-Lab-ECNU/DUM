import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os

import argparse
import time
import nibabel as nib

from monai.inferers import sliding_window_inference
from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance

from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance

from model.Universal_Model import Debiased_Universal_Medical_Segmenation_Model
from dataset.dataloader import get_loader
from utils.utils import dice_score, threshold_organ, merge_label, get_key
from utils.utils import TEMPLATE, ORGAN_NAME, NUM_CLASS
from utils.utils import organ_post_process
from utils.load_pretrain import load_checkpoint_partial

torch.multiprocessing.set_sharing_strategy('file_system')
 
def save_nii(batch_dict, save_path):
    img = batch_dict['image'].numpy()[0][0]
    label = batch_dict['one_channel_label_v2'].numpy()[0][0]
    gt = batch_dict['label'].numpy()[0][0]

    affine = batch_dict['image_meta_dict']['affine'][0].numpy()
    nib.save(nib.Nifti1Image(img, affine), f"{save_path}_img.nii.gz")
    nib.save(nib.Nifti1Image(label, affine), f"{save_path}_label.nii.gz")
    nib.save(nib.Nifti1Image(gt, affine), f"{save_path}_gt.nii.gz")


def validation(model, alpha, ValLoader, val_transforms, args):
    dataset_list_strs = '_'.join([str(l) for l in args.dataset_list])
    save_dir = args.log_name + f'/test_result_epoch{args.epoch}'

    os.makedirs(save_dir, exist_ok=True)
    if args.store_result:
        os.makedirs(save_dir + '/predict', exist_ok=True)

    model.eval()
    dice_list, hausdorff_list, recall_list, specificity_list, precision_list, iou_list = {}, {}, {}, {}, {}, {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS)) 
        hausdorff_list[key] = np.zeros((2, NUM_CLASS)) 
        recall_list[key] = np.zeros((2, NUM_CLASS)) 
        specificity_list[key] = np.zeros((2, NUM_CLASS)) 
        precision_list[key] = np.zeros((2, NUM_CLASS)) 
        iou_list[key] = np.zeros((2, NUM_CLASS)) 

    columns = ['name'] + [str(i) for i in range(32)]
    df_test = pd.DataFrame(columns=columns)
    
    #################### start inference ######################
    for index, batch in enumerate(tqdm(ValLoader)):
        image, label, name = batch["image"].cuda(), batch["post_label"], batch["name"]
        with torch.no_grad():
            predIX = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.5, mode='gaussian',
                                            tasks_id=name, bias=True, sw_device="cuda", device="cuda")
            predIX = predIX.detach().cpu()
            predX = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model, overlap=0.5, mode='gaussian',
                                            tasks_id=name, bias=False, sw_device="cuda", device="cuda")
            predX = predX.detach().cpu()

            pred = (1 + alpha) * F.sigmoid(predIX) - alpha * F.sigmoid(predX)
            pred_sigmoid = torch.clip(pred, 0.0, 1.0)

        
        pred_hard = threshold_organ(pred_sigmoid, threshold=args.threshold)
        pred_hard = pred_hard.cpu()
        torch.cuda.empty_cache()

        B = pred_hard.shape[0]
        df_test.loc[index, 'name'] = name[0]
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
                    recall_list[template_key][0][organ-1] += recall.item()
                    recall_list[template_key][1][organ-1] += 1
                    specificity_list[template_key][0][organ-1] += specificity.item()
                    specificity_list[template_key][1][organ-1] += 1
                    precision_list[template_key][0][organ-1] += precision.item()
                    precision_list[template_key][1][organ-1] += 1
                    iou_list[template_key][0][organ-1] += iou.item()
                    iou_list[template_key][1][organ-1] += 1
                    sf = compute_surface_distances(label[b, organ - 1, :, :, :].numpy().astype(np.bool_),
                                                   pred_hard_post[b, organ - 1, :, :, :].numpy().astype(np.bool_),
                                                   spacing_mm=(1.5, 1.5, 1.5))
                    nsd = compute_surface_dice_at_tolerance(sf, tolerance_mm=1.)
                    hausdorff_list[template_key][0][organ - 1] += nsd
                    hausdorff_list[template_key][1][organ - 1] += 1
 
                    df_test.loc[index, f'{organ}'] = dice_organ.item()
                    content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice_organ.item())
                    print('%s: dice %.4f, recall %.4f, precision %.4f, specificity %.4f, hausdorff_distance %.4f.'%(ORGAN_NAME[organ-1],
                                                                                           dice_organ.item(), recall.item(),
                                                                                           precision.item(), specificity.item(),
                                                                                           nsd))
        df_test.to_csv(f"{save_dir}/testset_allcase_dice_{dataset_list_strs}.csv", index=False)

        #################### save prediction mask ######################
        if args.store_result:
            pred_sigmoid_store = (pred_sigmoid.cpu().numpy() * 255).astype(np.uint8)
            label_store = (label.numpy()).astype(np.uint8)
            np.savez_compressed(save_dir + '/predict/' + name[0].split('/')[0] + name[0].split('/')[-1],
                                pred=pred_sigmoid_store, label=label_store)
            one_channel_label_v1, one_channel_label_v2 = merge_label(pred_hard_post, name)
            batch['one_channel_label_v1'] = one_channel_label_v1.cpu()
            batch['one_channel_label_v2'] = one_channel_label_v2.cpu()
            _, split_label = merge_label(batch["post_label"], name)
            batch['split_label'] = split_label.cpu()
            save_nii(batch, save_dir + '/predict/' + name[0].split('/')[0]+ name[0].split('/')[-1])

        torch.cuda.empty_cache()

    #################### merge all cases' metric results ######################
    ave_organ_dice = np.zeros((2, NUM_CLASS))
    ave_organ_hausdorff_distance = np.zeros((2, NUM_CLASS))
    ave_organ_recall = np.zeros((2, NUM_CLASS))
    ave_organ_specificity = np.zeros((2, NUM_CLASS))
    ave_organ_precision = np.zeros((2, NUM_CLASS))
    ave_organ_iou = np.zeros((2, NUM_CLASS))

    with open(f"{save_dir}/test_result.txt", 'w') as f:
        for key in TEMPLATE.keys():
            organ_list = TEMPLATE[key]
            content = 'Dice_Task%s| ' % (key)
            content_hausdorff_distance = 'Hausdorff_distance_Task%s| ' % (key)
            content_recall = 'Recall_Task%s| ' % (key)
            content_specificity = 'Specificity_Task%s| ' % (key)
            content_precision = 'Precision_Task%s| ' % (key)
            content_iou = 'IoU_Task%s| ' % (key)

            for organ in organ_list:
                dice = dice_list[key][0][organ - 1] / dice_list[key][1][organ - 1]
                hausdorff_distance = hausdorff_list[key][0][organ - 1] / hausdorff_list[key][1][organ - 1]
                recall = recall_list[key][0][organ - 1] / recall_list[key][1][organ - 1]
                specificity = specificity_list[key][0][organ - 1] / specificity_list[key][1][organ - 1]
                precision = precision_list[key][0][organ - 1] / precision_list[key][1][organ - 1]
                iou = iou_list[key][0][organ - 1] / iou_list[key][1][organ - 1]

                content += '%s: %.4f, ' % (ORGAN_NAME[organ - 1], dice)
                content_hausdorff_distance += '%s: %.4f, ' % (ORGAN_NAME[organ - 1], hausdorff_distance)
                content_recall += '%s: %.4f, ' % (ORGAN_NAME[organ - 1], recall)
                content_specificity += '%s: %.4f, ' % (ORGAN_NAME[organ - 1], specificity)
                content_precision += '%s: %.4f, ' % (ORGAN_NAME[organ - 1], precision)
                content_iou += '%s: %.4f, ' % (ORGAN_NAME[organ - 1], iou)

                ave_organ_dice[0][organ - 1] += dice_list[key][0][organ - 1]
                ave_organ_dice[1][organ - 1] += dice_list[key][1][organ - 1]

                ave_organ_hausdorff_distance[0][organ - 1] += hausdorff_list[key][0][organ - 1]
                ave_organ_hausdorff_distance[1][organ - 1] += hausdorff_list[key][1][organ - 1]

                ave_organ_recall[0][organ - 1] += recall_list[key][0][organ - 1]
                ave_organ_recall[1][organ - 1] += recall_list[key][1][organ - 1]

                ave_organ_specificity[0][organ - 1] += specificity_list[key][0][organ - 1]
                ave_organ_specificity[1][organ - 1] += specificity_list[key][1][organ - 1]

                ave_organ_precision[0][organ - 1] += precision_list[key][0][organ - 1]
                ave_organ_precision[1][organ - 1] += precision_list[key][1][organ - 1]

                ave_organ_iou[0][organ - 1] += iou_list[key][0][organ - 1]
                ave_organ_iou[1][organ - 1] += iou_list[key][1][organ - 1]


            f.write(content)
            f.write(content_hausdorff_distance)
            f.write(content_recall)
            f.write(content_specificity)
            f.write(content_precision)
            f.write(content_iou)
            f.write('\n')

        content_dice = 'Average Dice | '
        content_hausdorff_distance = 'Average Hausdorff_distance | '
        content_recall = 'Average Recall | '
        content_specificity = 'Average Specificity | '
        content_precision = 'Average Recall | '
        content_iou = 'Average Specificity | '

        for i in range(NUM_CLASS):
            content_dice += '%s: %.4f, ' % (ORGAN_NAME[i], ave_organ_dice[0][i] / ave_organ_dice[1][i])
            content_hausdorff_distance += '%s: %.4f, ' % (
            ORGAN_NAME[i], ave_organ_hausdorff_distance[0][i] / ave_organ_hausdorff_distance[1][i])
            content_recall += '%s: %.4f, ' % (ORGAN_NAME[i], ave_organ_recall[0][i] / ave_organ_recall[1][i])
            content_specificity += '%s: %.4f, ' % (
            ORGAN_NAME[i], ave_organ_specificity[0][i] / ave_organ_specificity[1][i])
            content_precision += '%s: %.4f, ' % (ORGAN_NAME[i], ave_organ_precision[0][i] / ave_organ_precision[1][i])
            content_iou += '%s: %.4f, ' % (
            ORGAN_NAME[i], ave_organ_iou[0][i] / ave_organ_iou[1][i])

        print(content_dice)
        f.write(content_dice)
        f.write(content_hausdorff_distance)
        f.write(content_recall)
        f.write(content_specificity)
        f.write(content_precision)
        f.write(content_iou)
        f.write('\n')
        f.write('%s: %.4f, ' % ('average', np.mean(ave_organ_dice[0] / ave_organ_dice[1])))
        f.write('\n')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--log_name', default='./out/', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--resume1', default='./out/Nvidia/old_fold0/aepoch_500.pth', help='The path resume from checkpoint')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=1000, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=10, type=int, help='Store model how often')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight Decay')
    parser.add_argument('--backbone', default='unet', help='backbone [swinunetr or unet]')


    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_10_inner', 'PAOT_123457891213']) # 'PAOT', 'felix'
    ### please check this argment carefull
    ### PAOT: include PAOT_123457891213 and PAOT_10
    ### PAOT_123457891213: include 1 2 3 4 5 7 8 9 12 13
    ### PAOT_10_inner
    ### external15
    ### external16_totalseg
    ### external17
    parser.add_argument('--data_root_path', default='/home/jliu288/data/whole_organ/', help='data root path')
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
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
    parser.add_argument('--store_result', action="store_true", default=False, help='whether save prediction result')
    parser.add_argument('--cache_rate', default=0., type=float, help='The percentage of cached data in total')
    parser.add_argument('--num_context', default=4, type=int, help='context prompt number in each organ')
    parser.add_argument('--threshold_organ', default=None, nargs='+', type=str)
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
    args.epoch = args.resume1.split('_')[-1][:-4]
    args.log_name = '/'.join(args.resume1.split('/')[:-1])
    print(f'Results will be saved in {args.log_name}!')

    model.cuda()
    torch.backends.cudnn.benchmark = True

    test_loader, val_transforms = get_loader(args)
    validation(model, args.alpha, test_loader, val_transforms, args)

if __name__ == "__main__":
    main()
