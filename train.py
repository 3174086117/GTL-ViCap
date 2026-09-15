from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import (SequentialSampler, RandomSampler)
import numpy as np
import random
import os
from collections import OrderedDict
from nlgeval.nlgeval import NLGEval
import time
import argparse
import matplotlib.pyplot as plt
from modules.tokenization import BertTokenizer
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling import CaptionGenerator
from modules.optimization import BertAdam
from modules.beam import Beam
from torch.utils.data import DataLoader
# from dataloaders.dataloader_msrvtt_feats import MSRVTT_Feats_DataLoader
from dataloaders.prior_dataloader_msrvtt_feats import MSRVTT_Feats_DataLoader
from feature_extractor.util import get_logger
from tqdm import tqdm
from dataloaders.prior_dataloader_msvd_feats import MSVD_Feats_DataLoader
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
import csv
from modules.until_module import LayerNorm


from Alignment.Thematic_retrieval import get_video_topic_topk



global logger

DEVICE = "cuda"

def get_args(description='CaptionGenerator'):
    parser = argparse.ArgumentParser(description=description)
    # parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    # parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--do_train", default=False, help="Whether to run training.")
    parser.add_argument("--do_eval", default=True, help="Whether to run eval on the dev set.")


    #MSRVTT
    # parser.add_argument('--data_path', type=str, default=r"C:\Users\admin\Desktop\YangYang\data\MSRVTT\MSRVTT_data.json",
    #                     help='caption and transcription file path')
    # parser.add_argument("--text_embedding_path", default=r"C:\Users\admin\Desktop\YangYang\UHCL-main\extracted_feats\msrvtt\merged_dpc1000_features.pt", help=" ")
    # # parser.add_argument('--features_path', type=str, default=r"C:\Users\admin\Desktop\YangYang\UHCL-main\extracted_feats\msrvtt\MSRVTT_CLIP4Clip_features_clip.pickle",
    # #                     help='feature path for CLIP features')
    # parser.add_argument('--features_path', type=str,
    #                     default=r"dataset/MSRVTT/MSRVTT_CLIP4Clip_features.pickle",
    #                     help='feature path for CLIP features')
    #MSVD
    parser.add_argument('--data_path', type=str, default=r"dataset/MSVD",
                        help='caption and transcription file path')
    parser.add_argument('--text_embedding_path', type=str, default=r"C:\Users\admin\Desktop\YangYang\V2T-CLIP4Caption-Reproduction-main\cluster_msvd\merged_dpc1000_features.pt",
                        help='')
    # parser.add_argument('--features_path', type=str, default=r"C:\Users\admin\Desktop\YangYang\UHCL-main\extracted_feats\msrvtt\MSRVTT_CLIP4Clip_features_clip.pickle",
    #                     help='feature path for CLIP features')
    parser.add_argument('--features_path', type=str,
                        default=r"dataset/MSVD/MSVD_CLIP4Clip_features.pickle",
                        help='feature path for CLIP features')


    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=256, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=512, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=48, help='')
    parser.add_argument('--max_frames', type=int, default=19, help='')
    parser.add_argument('--topk', type=int, default=1, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--min_time', type=float, default=5.0, help='Gather small clips')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--save_frequency", default=3, type=int, required=False,
                        help="save model frequency.")
    parser.add_argument("--output_dir", default="new_output/without_cluster_k5", type=str, required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--bert_model", default=r"D:\Desktop\YangYang\video-caption.pytorch-master\bert-base-uncased", type=str, required=False, help="Bert pre-trained model")
    parser.add_argument("--visual_model", default="visual-base", type=str, required=False, help="Visual module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    parser.add_argument("--init_model", default=r"C:\Users\admin\Desktop\YangYang\V2T-CLIP4Caption-Reproduction-main\msvd_output\topic_noun_verb\pytorch_model.bin.3", type=str, required=False, help="Initial model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--datatype", default="msvd", type=str, help="Point the dataset `msrvtt` to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether use MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=2, help="Layer NO. of visual.")
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=2, help="Layer NO. of decoder.")
    parser.add_argument('--d_model', type=int, default=512, help="dim of gcn model.")

    parser.add_argument('--patience', type=int, default=50, help="Number of epochs with no improvement after which training will be stopped.")
    parser.add_argument('--patience_metric', type=str, default="CIDEr", help="Metric which is used for early stopping.")
    parser.add_argument('--target_metric', type=str, default="CIDEr", help="Target metric which is used to select the best model.")

    args = parser.parse_args()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args

def set_seed_logger(args):
    global logger
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args.world_size = 1
    args.local_rank = 0

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    logger = get_logger(os.path.join(args.output_dir, "log.txt"))

    logger.info("Effective parameters:")
    for key in sorted(args.__dict__):
        logger.info("  <<< {}: {}".format(key, args.__dict__[key]))

    return args

def init_device(args, local_rank):
    global logger
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = 1
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))
    args.n_gpu = n_gpu

    if args.batch_size % args.n_gpu != 0 or args.batch_size_val % args.n_gpu != 0:
        raise ValueError("Invalid batch_size/batch_size_val and n_gpu parameter: {}%{} and {}%{}, should be == 0".format(
            args.batch_size, args.n_gpu, args.batch_size_val, args.n_gpu))

    return device, n_gpu

def init_model(args, device, n_gpu, local_rank):
    if args.init_model:
        model_state_dict = torch.load(args.init_model, map_location='cpu')
    else:
        model_state_dict = None

    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')

    model = CaptionGenerator.from_pretrained(args.bert_model, args.visual_model, args.decoder_model,
                                   cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
    # print(model)

    model.to(device)
    return model

def prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, local_rank, coef_lr=1.):
    if hasattr(model, 'module'):
        model = model.module

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    no_decay_param_tp = [(n, p) for n, p in param_optimizer if not any(nd in n for nd in no_decay)]
    decay_param_tp = [(n, p) for n, p in param_optimizer if any(nd in n for nd in no_decay)]

    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." in n]
    no_decay_nobert_param_tp = [(n, p) for n, p in no_decay_param_tp if "bert." not in n]

    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." in n]
    decay_nobert_param_tp = [(n, p) for n, p in decay_param_tp if "bert." not in n]

    optimizer_grouped_parameters = [
        {'params': [p for n, p in no_decay_bert_param_tp], 'weight_decay': 0.01, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in no_decay_nobert_param_tp], 'weight_decay': 0.01},
        {'params': [p for n, p in decay_bert_param_tp], 'weight_decay': 0.0, 'lr': args.lr * coef_lr},
        {'params': [p for n, p in decay_nobert_param_tp], 'weight_decay': 0.0}
    ]

    scheduler = None
    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.lr, warmup=args.warmup_proportion,
                         schedule='warmup_linear', t_total=num_train_optimization_steps, weight_decay=0.01,
                         max_grad_norm=1.0)

    return optimizer, scheduler, model

def dataloader_msrvtt_train(args, tokenizer):
    msrvtt_dataset = MSRVTT_Feats_DataLoader(
        json_path=args.data_path,
        features_path=args.features_path,
        text_embedding_path=args.text_embedding_path,
        max_words=args.max_words,
        topk=args.topk,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type="train",
    )

    train_sampler = RandomSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        sampler=train_sampler,
        drop_last=False,
    )

    return dataloader, len(msrvtt_dataset), train_sampler

def dataloader_msrvtt_val_test(args, tokenizer, split_type="test",):
    msrvtt_testset = MSRVTT_Feats_DataLoader(
        json_path=args.data_path,
        features_path=args.features_path,
        text_embedding_path=args.text_embedding_path,
        max_words=args.max_words,
        topk=args.topk,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
    )

    test_sampler = SequentialSampler(msrvtt_testset)
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)

def dataloader_msvd_train(args, tokenizer, split_type="train",):
    msvd = MSVD_Feats_DataLoader(
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
        feature_framerate=args.feature_framerate,
    )

    train_sampler = RandomSampler(msvd)
    dataloader_msvd = DataLoader(
        msvd,
        sampler=train_sampler,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=True,)
    return dataloader_msvd, len(msvd), train_sampler

def dataloader_msvd_val_test(args, tokenizer, split_type="val",):
    msvd = MSVD_Feats_DataLoader(
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        split_type=split_type,
    )

    sampler = SequentialSampler(msvd)
    dataloader_msvd = DataLoader(
        msvd,
        sampler=sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader_msvd, len(msvd)

def score(ref, hypo):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores
def diversity_loss(feat, temperature=0.5):
    """
    feat: (B, G, D) 已经 L2 归一化的特征
    返回惩罚项，使不同 group 的特征相似度尽可能小
    """
    B, G, D = feat.shape
    # 计算每个 batch 内不同 group 之间的余弦相似度矩阵 (B, G, G)
    sim = torch.matmul(feat, feat.transpose(-2, -1)) / temperature
    # 屏蔽对角线（自己和自己的相似度）
    mask = 1 - torch.eye(G, device=feat.device).unsqueeze(0)
    off_diag_sim = sim * mask
    # 惩罚相似度（希望趋近于0或负）
    loss_div = off_diag_sim.abs().mean()
    return loss_div
def convert_state_dict_type(state_dict, ttype=torch.FloatTensor):
    if isinstance(state_dict, dict):
        cpu_dict = OrderedDict()
        for k, v in state_dict.items():
            cpu_dict[k] = convert_state_dict_type(v)
        return cpu_dict
    elif isinstance(state_dict, list):
        return [convert_state_dict_type(v) for v in state_dict]
    elif torch.is_tensor(state_dict):
        return state_dict.type(ttype)
    else:
        return state_dict

def save_model(epoch, args, model, type_name=""):

    model_to_save = model.module if hasattr(model, 'module') else model
    # L1_output_model_file = os.path.join(
    #     args.output_dir, "L1_pytorch_model.bin.{}{}".format("" if type_name == "" else type_name + ".", epoch))
    output_model_file = os.path.join(
        args.output_dir, "pytorch_model.bin.{}{}".format("" if type_name=="" else type_name+".", epoch))
    # torch.save(model_to_save.state_dict(), L1_output_model_file)
    # logger.info("Model saved to %s", L1_output_model_file)
    torch.save(model_to_save.state_dict(), output_model_file)
    logger.info("Model saved to %s", output_model_file)
    # return L1_output_model_file,output_model_file
    return output_model_file


def load_model(epoch, args, n_gpu, device, model_file=None):
    if model_file is None or len(model_file) == 0:
        model_file = os.path.join(args.output_dir, "pytorch_model.bin.{}".format(epoch))
    if os.path.exists(model_file):
        model_state_dict = torch.load(model_file, map_location='cpu')
        logger.info("Model loaded from %s", model_file)
        cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed')
        model = CaptionGenerator.from_pretrained(args.bert_model, args.visual_model, args.decoder_model,
                                       cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
        model.to(device)
    else:
        model = None
    return model

def train_epoch(epoch, args, L1, model, train_dataloader, tokenizer, device, n_gpu, optimizer, scheduler,
                global_step, nlgEvalObj=None, local_rank=0):
    global logger
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0
    topic_feat = torch.load(
        args.text_embedding_path).to(device)

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        if args.datatype == "msrvtt":
            input_ids, input_mask, segment_ids, video, video_mask, noun_feats, verb_feats,\
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = batch
        if args.datatype == "msvd":
            input_ids, input_mask, segment_ids, video, video_mask, \
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = batch
        # input_ids, input_mask, segment_ids, video, video_mask, noun_feats, verb_feats,\
        # pairs_masked_text, pairs_token_labels, masked_video, video_labels_index,\
        # pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = batch
        video = video.squeeze(1)
        video_mask = video_mask.squeeze()
        topk_idx, topk_score, topk_feat = get_video_topic_topk(video, topic_feat, video_mask, topk=args.topk)
        topk_mask = torch.ones(topk_feat.shape[0],topk_feat.shape[1]).to(device)



        video = torch.cat([topk_feat, video],dim=1)
        video_mask = torch.cat([topk_mask, video_mask],dim=1)

        # video = torch.cat([video, topk_feat], dim=1)
        # video_mask = torch.cat([video_mask, topk_mask], dim=1)



        decoder_scores = model(video, video_mask,
                     input_caption_ids=pairs_input_caption_ids, decoder_mask=pairs_decoder_mask)

        pairs_output_caption_ids = pairs_output_caption_ids.view(-1, pairs_output_caption_ids.shape[-1])

        # loss = model.decoder_loss_fct(decoder_scores.view(-1, model.bert_config.vocab_size), pairs_output_caption_ids.view(-1))
        loss = model.decoder_loss_fct(
            decoder_scores.view(-1, model.bert_config.vocab_size),
            pairs_output_caption_ids.view(-1).long()  # 这里加 .long() 即可
        )
        # print(video_noun_feats.shape)
        # print(noun_feats.shape)
        # noun_loss = criterion(video_noun_feats, noun_feats)

        # div_noun_loss = diversity_loss(video_noun_feats)

        # verb_loss = criterion(video_verb_feats, verb_feats)
        # div_verb_loss = diversity_loss(video_verb_feats)
        # print(f"loss:{loss},noun_loss:{noun_loss},verb_loss:{verb_loss}")
        # loss = (loss+0.25*noun_loss+0.25*verb_loss)/3
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()
        total_loss += float(loss)

        if (step + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if scheduler is not None:
                scheduler.step()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % log_step == 0:
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), "-".join([str('%.6f'%itm) for itm in sorted(list(set(optimizer.get_lr())))]),
                            float(loss), (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))
                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step

def get_inst_idx_to_tensor_position_map(inst_idx_list):
    return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}

def collect_active_part(beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
    _, *d_hs = beamed_tensor.size()
    n_curr_active_inst = len(curr_active_inst_idx)
    new_shape = (n_curr_active_inst * n_bm, *d_hs)

    beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
    beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
    beamed_tensor = beamed_tensor.view(*new_shape)
    return beamed_tensor

def collate_active_info(input_tuples, inst_idx_to_position_map, active_inst_idx_list, n_bm, device):
    assert isinstance(input_tuples, tuple)
    visual_output_rpt, video_mask_rpt = input_tuples

    n_prev_active_inst = len(inst_idx_to_position_map)
    active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
    active_inst_idx = torch.LongTensor(active_inst_idx).to(device)

    active_visual_output_rpt = collect_active_part(visual_output_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_video_mask_rpt = collect_active_part(video_mask_rpt, active_inst_idx, n_prev_active_inst, n_bm)
    active_inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

    return (active_visual_output_rpt, active_video_mask_rpt), active_inst_idx_to_position_map

def beam_decode_step(decoder, inst_dec_beams, len_dec_seq,
                     inst_idx_to_position_map, n_bm, device, input_tuples, decoder_length=None):
    assert isinstance(input_tuples, tuple)

    def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
        dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
        dec_partial_seq = torch.stack(dec_partial_seq).to(device)
        dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
        return dec_partial_seq

    def predict_word(next_decoder_ids, n_active_inst, n_bm, device, input_tuples):
        visual_output_rpt, video_mask_rpt = input_tuples
        next_decoder_mask = torch.ones(next_decoder_ids.size(), dtype=torch.uint8).to(device)
        dec_output = decoder(visual_output_rpt, video_mask_rpt, next_decoder_ids, next_decoder_mask, shaped=True, get_logits=True)
        dec_output = dec_output[:, -1, :]
        word_prob = torch.nn.functional.log_softmax(dec_output, dim=1)
        word_prob = word_prob.view(n_active_inst, n_bm, -1)
        return word_prob

    def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map, decoder_length=None):
        active_inst_idx_list = []
        for inst_idx, inst_position in inst_idx_to_position_map.items():
            if decoder_length is None:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
            else:
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position], word_length=decoder_length[inst_idx])
            if not is_inst_complete:
                active_inst_idx_list += [inst_idx]
        return active_inst_idx_list

    n_active_inst = len(inst_idx_to_position_map)
    dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
    word_prob = predict_word(dec_seq, n_active_inst, n_bm, device, input_tuples)
    active_inst_idx_list = collect_active_inst_idx_list(inst_dec_beams, word_prob, inst_idx_to_position_map, decoder_length=decoder_length)
    return active_inst_idx_list

def collect_hypothesis_and_scores(inst_dec_beams, n_best):
    all_hyp, all_scores = [], []
    for inst_idx in range(len(inst_dec_beams)):
        scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
        all_scores += [scores[:n_best]]
        hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
        all_hyp += [hyps]
    return all_hyp, all_scores

def eval_epoch(args, L1, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=None, test_set=None):
    if hasattr(model, 'module'):
        model = model.module.to(device)

    all_result_lists = []
    all_caption_lists = []
    # L1.eval()
    model.eval()
    topic_feat = torch.load(
        args.text_embedding_path).to(device)

    for batch in tqdm(test_dataloader, desc='validation'):
        processed_batch = []
        for t in batch:
            if torch.is_tensor(t):
                if t.dtype == torch.float64:
                    t = t.float()  # 转换为 float32
                t = t.to(device=device, non_blocking=True)
            processed_batch.append(t)
        batch = processed_batch
        batch = tuple(t.to(device, non_blocking=True) for t in batch)
        if args.datatype == "msrvtt":
            input_ids, input_mask, segment_ids, video, video_mask, noun_feats, verb_feats,\
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = batch
        if args.datatype == "msvd":
            input_ids, input_mask, segment_ids, video, video_mask, \
            pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
            pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = batch

        video = video.squeeze()
        video_mask = video_mask.squeeze()
        topk_idx, topk_score, topk_feat = get_video_topic_topk(video, topic_feat, video_mask, topk=args.topk)
        topk_mask = torch.ones(video_mask.shape[0], topk_feat.shape[1]).to(device)

        video = torch.cat([topk_feat, video], dim=1)
        video_mask = torch.cat([topk_mask, video_mask], dim=1)

        # print(video.shape)
        with torch.no_grad():
            # video = L1(video.float())

            # video_noun_feats = model.noun_align_module(video.squeeze(), video_mask.squeeze())
            # video_verb_feats = model.verb_align_module(video.squeeze(), video_mask.squeeze())
            # # print(video_verb_feats.shape)
            # # print(video.squeeze().shape)
            # video = torch.cat([video_verb_feats.unsqueeze(1), video_noun_feats.unsqueeze(1), video.squeeze()], dim=1)
            #
            # prior_mask = torch.ones(video.shape[0], video_verb_feats.unsqueeze(1).shape[1]+video_noun_feats.unsqueeze(1).shape[1]).to(video_mask.device)
            #
            # video_mask = torch.cat([prior_mask, video_mask.squeeze()], dim=1)
            # print(video.shape)
            visual_output = model.get_visual_output(video, video_mask)

            n_bm = 5
            device = visual_output.device
            n_inst, len_v, v_h = visual_output.size()
            decoder = model.decoder_caption
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            visual_output_rpt = visual_output.repeat(1, n_bm, 1).view(n_inst * n_bm, len_v, v_h)
            video_mask_rpt = video_mask.repeat(1, n_bm).view(n_inst * n_bm, len_v)

            inst_dec_beams = [Beam(n_bm, device=device, tokenizer=tokenizer) for _ in range(n_inst)]
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = get_inst_idx_to_tensor_position_map(active_inst_idx_list)

            for len_dec_seq in range(1, args.max_words + 1):
                active_inst_idx_list = beam_decode_step(decoder, inst_dec_beams,
                                                        len_dec_seq, inst_idx_to_position_map, n_bm, device,
                                                        (visual_output_rpt, video_mask_rpt))
                if not active_inst_idx_list:
                    break
                (visual_output_rpt, video_mask_rpt), \
                inst_idx_to_position_map = collate_active_info((visual_output_rpt, video_mask_rpt),
                                                               inst_idx_to_position_map, active_inst_idx_list, n_bm, device)

            batch_hyp, batch_scores = collect_hypothesis_and_scores(inst_dec_beams, 1)
            result_list = [batch_hyp[i][0] for i in range(n_inst)]
            pairs_output_caption_ids = pairs_output_caption_ids.view(-1, pairs_output_caption_ids.shape[-1])
            caption_list = pairs_output_caption_ids.cpu().detach().numpy()

            for re_idx, re_list in enumerate(result_list):
                decode_text_list = tokenizer.convert_ids_to_tokens(re_list)
                if "[SEP]" in decode_text_list:
                    SEP_index = decode_text_list.index("[SEP]")
                    decode_text_list = decode_text_list[:SEP_index]
                if "[PAD]" in decode_text_list:
                    PAD_index = decode_text_list.index("[PAD]")
                    decode_text_list = decode_text_list[:PAD_index]
                decode_text = ' '.join(decode_text_list)
                decode_text = decode_text.replace(" ##", "").strip()
                all_result_lists.append(decode_text)

            for re_idx, re_list in enumerate(caption_list):
                decode_text_list = tokenizer.convert_ids_to_tokens(re_list)
                if "[SEP]" in decode_text_list:
                    SEP_index = decode_text_list.index("[SEP]")
                    decode_text_list = decode_text_list[:SEP_index]
                if "[PAD]" in decode_text_list:
                    PAD_index = decode_text_list.index("[PAD]")
                    decode_text_list = decode_text_list[:PAD_index]
                decode_text = ' '.join(decode_text_list)
                decode_text = decode_text.replace(" ##", "").strip()
                all_caption_lists.append(decode_text)

    if test_set is not None and hasattr(test_set, 'iter2video_pairs_dict'):
        hyp_path = os.path.join(args.output_dir, "hyp_complete_results.txt")
        with open(hyp_path, "w", encoding='utf-8') as writer:
            writer.write("{}\t{}\t{}\n".format("video_id", "start_time", "caption"))
            for idx, pre_txt in enumerate(all_result_lists):
                video_id, sub_id = test_set.iter2video_pairs_dict[idx]
                start_time = test_set.data_dict[video_id]['start'][sub_id]
                writer.write("{}\t{}\t{}\n".format(video_id, start_time, pre_txt))
        logger.info("File of complete results is saved in {}".format(hyp_path))

    hyp_path = os.path.join(args.output_dir, "hyp.txt")
    with open(hyp_path, "w", encoding='utf-8') as writer:
        for pre_txt in all_result_lists:
            writer.write(pre_txt+"\n")

    ref_path = os.path.join(args.output_dir, "ref.txt")
    with open(ref_path, "w", encoding='utf-8') as writer:
        for ground_txt in all_caption_lists:
            writer.write(ground_txt + "\n")

    if args.datatype == "msrvtt"or args.datatype == "msvd":
        all_caption_lists = []
        sentences_dict = test_dataloader.dataset.sentences_dict
        video_sentences_dict = test_dataloader.dataset.video_sentences_dict
        video_ids = set()
        for idx in range(len(sentences_dict)):
            video_id, _ = sentences_dict[idx]
            video_ids.add(video_id)
            sentences = video_sentences_dict[video_id]
            all_caption_lists.append(sentences)
        if args.datatype != "msvd":
            all_caption_lists = [list(itms) for itms in zip(*all_caption_lists)]
    else:
        all_caption_lists = [all_caption_lists]

    if args.datatype == "msvd":
        all_result_dict = {}
        all_caption_dict = {}
        for i in range(len(all_result_lists)):
            all_result_dict[i] = [all_result_lists[i]]
        for i in range(len(all_caption_lists)):
            all_caption_dict[i]=all_caption_lists[i]
        metrics_nlg = score(all_caption_dict,all_result_dict)
    else:
        metrics_nlg = nlgEvalObj.compute_metrics(ref_list=all_caption_lists, hyp_list=all_result_lists)

    logger.info(">>>  BLEU_1: {:.4f}, BLEU_2: {:.4f}, BLEU_3: {:.4f}, BLEU_4: {:.4f}".
                format(metrics_nlg["Bleu_1"], metrics_nlg["Bleu_2"], metrics_nlg["Bleu_3"], metrics_nlg["Bleu_4"]))
    logger.info(">>>  ROUGE_L: {:.4f}, CIDEr: {:.4f}, METEOR: {:.4f}".format(metrics_nlg["ROUGE_L"], metrics_nlg["CIDEr"],metrics_nlg["METEOR"]))
    return metrics_nlg



DATALOADER_DICT = {}
DATALOADER_DICT["msrvtt"] = {"train":dataloader_msrvtt_train, "val":dataloader_msrvtt_val_test, "test":dataloader_msrvtt_val_test}
DATALOADER_DICT["msvd"] = {"train":dataloader_msvd_train, "val":dataloader_msvd_val_test, "test":dataloader_msvd_val_test}

class Liner(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, in_freatures=512,out_freatures=1024):
        super(Liner, self).__init__()

        # self.word_embeddings = nn.Linear(config.vocab_size, config.hidden_size)
        self.L1 = nn.Linear(in_freatures, out_freatures)
        self.LayerNorm = LayerNorm(out_freatures, eps=1e-12)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file

        self.dropout = nn.Dropout(0.5)

    def forward(self, input_embeddings):
        return self.dropout(self.LayerNorm(self.L1(input_embeddings)))

def plot_and_save(data, save_path="result.png", title="Data Plot", xlabel="X", ylabel="Y"):
    """
    输入字典，绘制图像并保存
    :param data: 字典，如 {1:0.34, 2:0.58, 3:1.5}
    :param save_path: 保存的图片路径
    """
    # 取出 x 和 y
    x = list(data.keys())
    y = list(data.values())

    # 创建画布
    plt.figure(figsize=(8, 5))

    # 画图：折线 + 点
    plt.plot(x, y, marker='o', color='b', linestyle='-', linewidth=2, markersize=8)

    # 标题和标签
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # 网格
    plt.grid(True, alpha=0.3)

    # 保存图片（关键）
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # 关闭画布，不占用内存

def plot_training_metrics(results_dict, save_path="training_metrics.png"):
    """
    绘制训练过程中四个指标的变化曲线
    横坐标：训练轮次 Epoch
    纵坐标：指标分数
    所有指标画在同一张图，并自动保存高清图片
    """
    # 绘图设置（论文风格 + 高清）
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = 300

    # 专业配色 + 标记点
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    markers = ['o', 's', '^', 'D']
    line_styles = ['-', '-', '-', '-']

    # 获取 epoch 数量
    example_values = next(iter(results_dict.values()))
    epochs = np.arange(1, len(example_values) + 1)  # 1,2,3,4...

    # 创建画布
    plt.figure(figsize=(10, 6))

    # 绘制每个指标
    for i, (metric, scores) in enumerate(results_dict.items()):
        plt.plot(
            epochs, scores,
            label=metric,
            color=colors[i],
            marker=markers[i],
            linestyle=line_styles[i],
            linewidth=3,
            markersize=9
        )
        # 标注数值
        for e, s in zip(epochs, scores):
            plt.text(e, s + 0.015, f'{s:.2f}', ha='center', fontsize=10, fontweight='bold')

    # 坐标轴设置
    plt.xlabel("Training Epoch", fontsize=14, fontweight='bold')
    plt.ylabel("Metric Score", fontsize=14, fontweight='bold')
    plt.title("Training Metrics Over Epochs", fontsize=16, fontweight='bold')

    plt.xticks(epochs, fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()

    # 保存并显示
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    # plt.show()
    # print(f"✅ 图表已保存：{save_path}")




def main():
    global logger
    args = get_args()
    args = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    L1 = Liner(512,1024)
    # L1.to(device)
    model = init_model(args, device, n_gpu, args.local_rank)
    # print(model)
    nlgEvalObj = NLGEval(no_overlap=False, no_skipthoughts=True, no_glove=True, metrics_to_omit=["SPICE"])

    assert args.datatype in DATALOADER_DICT
    val_dataloader, val_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, "val")
    test_dataloader, test_length = DATALOADER_DICT[args.datatype]["test"](args, tokenizer, "test")

    logger.info("***** Running val *****")
    logger.info("  Num examples = %d", val_length)
    logger.info("  Batch size = %d", args.batch_size_val)
    logger.info("  Num steps = %d", len(val_dataloader))
    logger.info("***** Running test *****")
    logger.info("  Num examples = %d", test_length)
    logger.info("  Batch size = %d", args.batch_size_val)
    logger.info("  Num steps = %d", len(test_dataloader))

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        if args.init_model:
            coef_lr = 1.0
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_length)
        logger.info("  Batch size = %d", args.batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_score = {"CIDEr": 0.00001}
        best_output_model_file = {"CIDEr": None}

        results_dict = {"Bleu_4":[], "ROUGE_L":[], "CIDEr":[], "METEOR":[]}
        train_loss_dict = {}

        csv_path = os.path.join(args.output_dir, "epoch_metrics.csv")
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch", "Bleu_4", "ROUGE_L", "CIDEr", "METEOR", "Train_Loss"])

        assert args.target_metric in best_score.keys()
        assert args.patience_metric in best_score.keys()
        global_step = 0
        stop_signal = torch.zeros(2)

        for epoch in range(args.epochs):
            tr_loss, global_step = train_epoch(epoch, args, L1, model, train_dataloader, tokenizer, device, n_gpu, optimizer,
                                               scheduler, global_step, nlgEvalObj=nlgEvalObj, local_rank=args.local_rank)
            train_loss_dict[int(epoch)] = tr_loss
            logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
            # if epoch % args.save_frequency == 0:
            output_model_file = save_model(epoch, args, model, type_name="")


            if epoch >= 0:
                metric_scores = eval_epoch(args, L1, model, val_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)

                # 收集指标
                bleu4 = float(metric_scores["Bleu_4"])
                rouge = float(metric_scores["ROUGE_L"])
                cider = float(metric_scores["CIDEr"])
                meteor = float(metric_scores["METEOR"])

                # 存入字典
                results_dict["Bleu_4"].append(bleu4)
                results_dict["ROUGE_L"].append(rouge)
                results_dict["CIDEr"].append(cider)
                results_dict["METEOR"].append(meteor)

                # ===================== 保存本轮指标到 CSV =====================
                with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, bleu4, rouge, cider, meteor, tr_loss])
                plot_training_metrics(results_dict)
                plot_and_save(
                    data=train_loss_dict,
                    save_path=os.path.join(args.output_dir, "train_loss.png"),
                    title="train loss Chart",
                    xlabel="epoch",
                    ylabel="loss"
                )
                for met in best_score.keys():
                    if met == args.target_metric:
                        if metric_scores[met] <= 0.001:
                            logger.warning("One of the metrics is less than 0.001. The training will be stopped.")
                            stop_signal[1] = 1
                            break
                        if best_score[met] <= metric_scores[met]:
                            best_score[met] = metric_scores[met]
                            best_output_model_file[met] = output_model_file
                            if met==args.patience_metric:
                                stop_signal[0] = 0
                        else:
                            if met==args.patience_metric:
                                stop_signal[0] += 1
                        logger.info("The best model based on {} is: {}, the {} is: {:.4f}".format(met, best_output_model_file[met], met, best_score[met]))

                if stop_signal[0]>=args.patience:
                    logger.warning("Early stopping, no improvement after {} epochs".format(args.patience))
                    break
                if stop_signal[1] == 1:
                    break
            else:
                logger.warning("Skip the evaluation after {}-th epoch.".format(epoch+1))

        test_scores = {}
        for met in best_score.keys():
            if met == args.target_metric:
                model = None
                model = load_model(-1, args, n_gpu, device, model_file=best_output_model_file[met])
                metric_scores = eval_epoch(args, L1, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)
                test_scores[met] = metric_scores
        for met in test_scores.keys():
            logger.info("Test score based on the best {} model ({}) : {}".format(met, best_output_model_file[met], str(test_scores[met])))

    elif args.do_eval:
        eval_epoch(args, L1, model, test_dataloader, tokenizer, device, n_gpu, nlgEvalObj=nlgEvalObj)

if __name__ == "__main__":
    main()