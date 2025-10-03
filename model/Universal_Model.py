from typing import Sequence, Tuple, Type, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from collections import OrderedDict
from torch.nn import LayerNorm

from model.Unet import DownTransition, UpTransition, OutputTransition
from copy import deepcopy
from .transformer_decoder import TransformerDecoder, TransformerDecoderLayer
from einops import rearrange, repeat

from utils.utils import TEMPLATE
from transformers import AutoTokenizer, AutoModel




class PromptLearner_Coco(nn.Module):
    def __init__(self, n_class, clip_model, ctx_dim, vis_dim, n_ctx):
        super().__init__()
        n_cls = n_class
        n_ctx = n_ctx #cfg.TRAINER.COCOOP.N_CTX
        ctx_init = False #cfg.TRAINER.COCOOP.CTX_INIT
        # dtype = clip_model.dtype
        self.n_cls = n_cls
        self.n_ctx = n_ctx

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        # print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))

    def forward(self, im_features, suffix=None): # suffix : X, length, dim
        # prefix = self.token_prefix
        # suffix = self.token_suffix
        bs = im_features.shape[0]
        if suffix is not None:
            class_num = suffix.shape[0]
        else:
            class_num = 1

        ctx = self.ctx  # (n_ctx, ctx_dim)
        # bias = self.meta_net(im_features)  # (batch, ctx_dim)
        # bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = torch.repeat_interleave(ctx, bs, 0)
        # ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)
        ctx_shifted = torch.repeat_interleave(ctx_shifted.unsqueeze(1), class_num, 1) #b, X, n_ctx, ctx_dim
        if suffix is not None:
            suffix = torch.repeat_interleave(suffix.unsqueeze(0), bs, 0) #b, X, length, dim

        # Use instance-conditioned context tokens for all classes
        # prompts = []
        # for ctx_shifted_i in ctx_shifted:
        #     ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)
        #     pts_i = self.construct_prompts(ctx_i, prefix, suffix)  # (n_cls, n_tkn, ctx_dim)
        #     prompts.append(pts_i)
        # prompts = torch.stack(prompts)
            prompts = self.construct_prompts(ctx_shifted, suffix)
        else:
            prompts = ctx_shifted

        return prompts # #b, (X+n_ctx), length, dim

class PromptLearner(nn.Module):
    def __init__(self, n_class, clip_model, ctx_dim, vis_dim, n_ctx):
        super().__init__()
        n_cls = n_class
        n_ctx = n_ctx #cfg.TRAINER.COCOOP.N_CTX
        ctx_init = False #cfg.TRAINER.COCOOP.CTX_INIT
        self.n_cls = n_cls
        self.n_ctx = n_ctx

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = "this ct is {}"#
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        # print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)

        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, vis_dim // 16)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(vis_dim // 16, ctx_dim))
        ]))



    def construct_prompts(self, ctx, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            # prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                ctx,  # (b, dim0, n_ctx, dim)
                suffix,  # (b, dim0, *, dim)
            ],
            dim=2,
        )

        return prompts

    def forward(self, im_features, suffix=None): # suffix : X, length, dim

        bs = im_features.shape[0]
        if suffix is not None:
            class_num = suffix.shape[0]
        else:
            class_num = 1

        ctx = self.ctx  # (n_ctx, ctx_dim)
        bias = self.meta_net(im_features)  # (batch, ctx_dim)

        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)
        ctx_shifted = torch.repeat_interleave(ctx_shifted.unsqueeze(1), class_num, 1) #b, X, n_ctx, ctx_dim
        if suffix is not None:
            suffix = torch.repeat_interleave(suffix.unsqueeze(0), bs, 0) #b, X, length, dim


            prompts = self.construct_prompts(ctx_shifted, suffix)
        else:
            prompts = ctx_shifted
        return prompts # #b, (X+n_ctx), length, dim


class Debiased_Universal_Medical_Segmenation_Model(nn.Module):
    def __init__(self, out_channels, task_total_number=32, act='relu', word_prompt_path=None,
                 num_context=4, word_prompt_number=384,
                 trans_num_layers=4):
        super().__init__()

        final_num_features = 512
        self.task_total_number = task_total_number

        self.final_num_features = final_num_features
        self.down_tr64 = DownTransition(1, 0, act)
        self.down_tr128 = DownTransition(64, 1, act)
        self.down_tr256 = DownTransition(128, 2, act)
        self.down_tr512 = DownTransition(256, 3, act)

        self.text_embed_dim = 768

        text_tokenizer = AutoTokenizer.from_pretrained("/mnt/nas/yunboxiang.ybx/TMI_rebuttal/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                                                       local_files_only=True)
        self.text_encoder = AutoModel.from_pretrained("/mnt/nas/yunboxiang.ybx/TMI_rebuttal/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                                                      local_files_only=True)

        ############ load organ-level prompt (medical prior) ###################
        self.register_buffer('organ_embedding', torch.randn(32, word_prompt_number, self.text_embed_dim))
        if 'csv' in word_prompt_path.split('.')[-1]:
            text_orignal_prompt = pd.read_csv(word_prompt_path)
            explicit_prompt = []
            for i in range(len(text_orignal_prompt)):
                text_input = text_orignal_prompt.loc[i, 'prompt']
                encoded_input = text_tokenizer(text_input, padding='max_length', truncation=True,
                                            max_length=word_prompt_number, return_tensors='pt')['input_ids']
                text_embedding = self.text_encoder.embeddings(encoded_input) # [1, word_prompt_number, 768]
                explicit_prompt.append(text_embedding)
            explicit_prompt = torch.stack(explicit_prompt).squeeze(1)#[32, word_prompt_number, 768]
            self.organ_embedding.data = explicit_prompt.float()
            print('success load word embedding by csv')
        else:
            word_embedding = torch.load(word_prompt_path)
            self.organ_embedding.data = word_embedding.float()
            print('success load word embedding by offline method')



        self.promptlearner = PromptLearner(n_class=task_total_number, clip_model=self.down_tr64,
                                            ctx_dim=self.text_embed_dim, vis_dim=final_num_features, n_ctx=num_context)

        self.text_to_vision = nn.Linear(self.text_embed_dim, final_num_features)

        decoder_layer = TransformerDecoderLayer(d_model=final_num_features, nhead=8, normalize_before=True)
        decoder_norm = nn.LayerNorm(final_num_features)
        self.transformer_decoders = TransformerDecoder(decoder_layer=decoder_layer, num_layers=trans_num_layers, norm=decoder_norm)


        self.up_tr256 = UpTransition(512, 512, 2, act, 512)
        self.up_tr128 = UpTransition(256, 256, 1,act)
        self.up_tr64 = UpTransition(128, 128,0,act)

        self.out_tr = OutputTransition(64, out_channels, use_sigmoid=False)

    def convert_task_id(self, tasks_id, device):
        #e.g. 10_Decathlon/Task08_HepaticVessel
        tasks_id_ = tasks_id[0].split('/')[0].split('_')
        if 'Decathlon' == tasks_id_[1]:
            tasks_id = f"{tasks_id_[0]}_{tasks_id[0].split('/')[1][4:6]}"
        else:
            tasks_id = tasks_id_[0]
        return tasks_id

    def forward(self, x_in, tasks_id, bias=False):
        names_id = tasks_id
        tasks_id = self.convert_task_id(tasks_id, x_in.device)
        bs = x_in.shape[0]

        ############################## encoder #################################
        out64, skip_out64 = self.down_tr64(x_in)
        out128, skip_out128 = self.down_tr128(out64)
        out256, skip_out256 = self.down_tr256(out128)
        out512, skip_out512 = self.down_tr512(out256)

        vision_embedding = out512.mean((-1, -2, -3)).unsqueeze(1) # B, N, C

        word_embedding = torch.index_select(self.organ_embedding, 0,
                                            torch.tensor(np.array(TEMPLATE[tasks_id]) - 1, device=x_in.device)) # X, 384, 768(dim)

        if bias:
            ############### merge instance-level context and organ-level feature as prompt ############
            word_embedding_incontext = self.promptlearner(vision_embedding.squeeze(1), word_embedding)  #b, (X+n_ctx), length, dim
            word_embedding_incontext = torch.stack([self.text_encoder.encoder(word_embedding_incontext[:,w])['last_hidden_state'] for w in range(word_embedding_incontext.shape[1])], dim=1) #b, (X+n_ctx), length, dim
            word_embedding_incontext = F.relu(self.text_to_vision(word_embedding_incontext)).reshape(-1, self.final_num_features)#, 'b n l c -> b (n l) c')#.reshape(bs, -1, self.final_num_features)  # N, C
            word_embedding_incontext = torch.repeat_interleave(word_embedding_incontext.unsqueeze(0), bs, 0)
            oo_embedding, attn_weights = self.transformer_decoders(vision_embedding.transpose(1, 0), word_embedding_incontext.transpose(1, 0), pos=None) # B N C 
        else:
            ############## only organ-level feature as prompt ############################
            word_embedding = F.relu(self.text_to_vision(word_embedding)).reshape(-1, self.final_num_features)#, 'b n l c -> b (n l) c')#.reshape(bs, -1, self.final_num_features)  # N, C
            word_embedding = torch.repeat_interleave(word_embedding.unsqueeze(0), bs, 0)
            oo_embedding, attn_weights = self.transformer_decoders(vision_embedding.transpose(1, 0), word_embedding.transpose(1, 0), pos=None) # B N C
        
        ############################## fusion prompt and visual feautres #################################
        oo_embedding = repeat(oo_embedding.transpose(1, 0), 'b n c -> b (n r) c', r=out512.shape[-1]**3)
        oo_embedding = rearrange(oo_embedding, 'b (h w d) c -> b c h w d', h=out512.shape[-1],
                                 w=out512.shape[-1], d=out512.shape[-1])
        out512 = torch.cat([out512, oo_embedding], dim=1)
        ############################## decoder #################################
        out_up_256 = self.up_tr256(out512, skip_out256)
        out_up_128 = self.up_tr128(out_up_256, skip_out128)
        out_up_64 = self.up_tr64(out_up_128, skip_out64)
        out = self.out_tr(out_up_64)

        # loss_unused = 0.
        # loss_unused += 0. * sum(p.sum() for p in self.promptlearner.parameters())

        return out

