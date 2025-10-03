import gc
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import time

import warnings
warnings.filterwarnings("ignore")

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter

from monai.losses import DiceCELoss
from monai.data import load_decathlon_datalist, decollate_batch, DistributedSampler
from monai.metrics import DiceMetric
from utils.seed_everything import seed_reproducer
from model.Universal_Model import Debiased_Universal_Medical_Segmenation_Model

from dataset.dataloader import get_loader_orignal
from utils import loss
from utils.utils import dice_score, check_data, TEMPLATE, get_key, NUM_CLASS
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
# from torch.cuda.amp import GradScaler, autocast
import logging
import nibabel as nib
from utils.load_pretrain import save_checkpoint, load_checkpoint_partial

logging.getLogger("nibabel").setLevel(logging.ERROR)

torch.multiprocessing.set_sharing_strategy('file_system')

@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        if not param.requires_grad:# ignore text encoder weight
            continue
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)

def train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE, model_ema_noncontext):
    alpha = args.alpha
    print(f"alpha is {alpha}!")
    loss_bce_ave = 0
    loss_dice_ave = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):

        x, y, name = batch["image"].to(args.device), batch["post_label"].float().to(args.device), batch['name']
        optimizer.zero_grad()

        with_context_logit = model(x, tasks_id=name, bias=True)
        with torch.no_grad():
            wo_context_logit = model_ema_noncontext(x, tasks_id=name, bias=False).detach()

        logit_map = (1+alpha) * torch.sigmoid(with_context_logit) - alpha * torch.sigmoid(wo_context_logit)

        logit_map = torch.clamp(logit_map, min=1e-7, max=1 - 1e-7)
        term_seg_Dice = loss_seg_DICE.forward(logit_map, y, name, TEMPLATE, False)

        term_seg_BCE = loss_seg_CE.forward(logit_map, y, name, TEMPLATE)
        loss = term_seg_BCE + term_seg_Dice 

        loss.backward()
        optimizer.step()

        update_ema_variables(model, model_ema_noncontext, args.alpha_ema)

        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, bce_loss=%2.5f)" % (
                args.epoch, step, len(train_loader), term_seg_Dice.item(), term_seg_BCE.item())
        )
        loss_bce_ave += term_seg_BCE.item()
        loss_dice_ave += term_seg_Dice.item()
        # torch.cuda.empty_cache()
    print('Epoch=%d: ave_dice_loss=%2.5f, ave_bce_loss=%2.5f' % (args.epoch, loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator)))
    
    return loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator)

def process(args):

    if args.dist:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)
    seed_reproducer(2024)


    # prepare the 3D model
    model = Debiased_Universal_Medical_Segmenation_Model(
                    out_channels=NUM_CLASS,
                    word_prompt_path=args.word_embedding_path,
                    num_context=args.num_context,
                    word_prompt_number=args.word_prompt_number,
                    trans_num_layers=args.trans_num_layers,)
    model_ema_noncontext = Debiased_Universal_Medical_Segmenation_Model(
                    out_channels=NUM_CLASS,
                    word_prompt_path=args.word_embedding_path,
                    num_context=args.num_context,
                    word_prompt_number=args.word_prompt_number,
                    trans_num_layers=args.trans_num_layers,)
    
    model.to(args.device)
    model.train()
    model_ema_noncontext.to(args.device)
    model_ema_noncontext.eval()

    for (name, param) in model.named_parameters():
        if "text_encoder" in name:
            param.requires_grad = False
            print(f"freeze text_encoder {name}")
        else:
            pass
        
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    # criterion and optimizer
    loss_seg_DICE = loss.DiceLoss(num_classes=NUM_CLASS)#.to(args.device)
    loss_seg_CE = loss.Multi_BCELoss(num_classes=NUM_CLASS, withlogit=False)#.to(args.device)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)
        
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[args.device], find_unused_parameters=True)
        model_ema_noncontext = DistributedDataParallel(model_ema_noncontext, device_ids=[args.device], find_unused_parameters=True)
    
    # resume model and ema_model weights
    if args.resume:
        checkpoint_main = load_checkpoint_partial(
            model=model,
            checkpoint_path=args.root_log_name + '/' + args.resume,
            exclude_prefixes=('text_encoder',)
        )
        load_checkpoint_partial(
            model=model_ema_noncontext,
            checkpoint_path=args.root_log_name + '/' + args.resume.replace('epoch', 'ema_epoch'),
            exclude_prefixes=('text_encoder',)
        )

        optimizer.load_state_dict(checkpoint_main['optimizer'])
        scheduler.load_state_dict(checkpoint_main['scheduler'])
        args.epoch = checkpoint_main['epoch']
        print(f"Resumed training from epoch {args.epoch}")

    torch.backends.cudnn.benchmark = True
    train_loader, train_sampler = get_loader_orignal(args)
    
    if rank == 0:
        writer = SummaryWriter(log_dir=f'{args.root_log_name}/' + args.log_name)
        print('Writing Tensorboard logs to ', f'{args.root_log_name}/' + args.log_name)

    # train and save model
    while args.epoch < args.max_epoch:
        if args.dist:
            dist.barrier()
            train_sampler.set_epoch(args.epoch)
        scheduler.step()

        loss_dice, loss_bce = train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE,
                                    model_ema_noncontext)
        if rank == 0:
            writer.add_scalar('train_dice_loss', loss_dice, args.epoch)
            writer.add_scalar('train_bce_loss', loss_bce, args.epoch)
            writer.add_scalar('lr', scheduler.get_lr(), args.epoch)

        if ((args.epoch+1) % args.store_num == 0 and args.epoch != 0) and rank == 0:
            save_dir = f'{args.root_log_name}/{args.log_name}'
            os.makedirs(save_dir, exist_ok=True)
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=args.epoch,
                path=f'{save_dir}/epoch_{args.epoch}.pth',
                exclude_prefixes=('text_encoder',)
            )

            save_checkpoint(
                model=model_ema_noncontext,
                optimizer=optimizer, 
                scheduler=scheduler,
                epoch=args.epoch,
                path=f'{save_dir}/ema_epoch_{args.epoch}.pth',
                exclude_prefixes=('text_encoder',)
            )
            print(f"Saved checkpoints at epoch {args.epoch}")

        args.epoch += 1

    dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local-rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    ## logging
    parser.add_argument('--root_log_name', default='out', help='The root path resume from checkpoint')
    parser.add_argument('--log_name', default='unet', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--resume', default=None, help='The path resume from checkpoint')
    parser.add_argument('--resume_ema', default=None, help='The path resume from checkpoint')
    # parser.add_argument('--pretrain', default=None,  #swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
    #                     help='The path of pretrain model. Eg, ./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt')
    # parser.add_argument('--trans_encoding', default='word_embedding', 
    #                     help='the type of encoding: rand_embedding or word_embedding')
    # parser.add_argument('--word_embedding', default='./pretrained_weights/txt_encoding.pth', 
    #                     help='The path of word embedding')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=500, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=5, type=int, help='Store model how often')
    parser.add_argument('--warmup_epoch', default=10, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    ## dataset
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0.25, type=float, help='The percentage of cached data in total')
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_123457891213', 'PAOT_10_inner']) # 'PAOT', 'felix', 'PAOT_10_inner'
    ### please check this argment carefully
    ### PAOT: include PAOT_123457891213 and PAOT_10
    ### PAOT_123457891213: include 1 2 3 4 5 7 8 9 12 13
    ### PAOT_10_inner: same with NVIDIA for comparison
    ### PAOT_10: original division
    parser.add_argument('--data_root_path', default='./UniversalModel/Datasets/', help='data root path')
    parser.add_argument('--data_txt_path', default='./dataset/dataset_list/', help='data txt path')
    parser.add_argument('--batch_size', default=1, help='batch size', type=int)
    parser.add_argument('--num_workers', default=8, type=int, help='workers numebr for DataLoader')
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

    parser.add_argument('--phase', default='train', help='train or validation or test')
    parser.add_argument('--uniform_sample', action="store_true", default=False, help='whether utilize uniform sample strategy')

    parser.add_argument('--datasetkey', nargs='+', default=['01', '02', '03', '04', '05',
                                            '07', '08', '09', '12', '13', '10_03',
                                            '10_06', '10_07', '10_08', '10_09', '10_10'],
                                            help='the content for ')
    parser.add_argument('--word_embedding_path', type=str, default='./explicit_prompt.pt', help='Name of Experiment')
    parser.add_argument('--num_context', default=4, type=int, help='context prompt number in each organ')
    parser.add_argument('--alpha', default=0.5, type=float, help='')
    parser.add_argument('--word_prompt_number', default=384, type=int, help='text prompt number in each organ')
    parser.add_argument('--trans_num_layers', default=4, type=int, help='transformer layer number in cross attention')
    parser.add_argument('--alpha_ema', default=0.99, type=float, help='')

    # ./Fix_Standard_Debias_V2_alpha05_0430/epoch_499.pth
    args = parser.parse_args()
    process(args=args)

if __name__ == "__main__":
    main()


