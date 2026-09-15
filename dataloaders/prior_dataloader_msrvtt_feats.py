from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle5 as pickle
import pandas as pd
from collections import defaultdict
import json
import random
import torch


class MSRVTT_Feats_DataLoader(Dataset):
    """Implementation of the dataloader for MSRVTT. Mainly used in the model training and evaluation.
    Params:
        json_path: Path to the MSRVTT_data.json file.
        features_path: Path to the extracted feature file.
        tokenizer: Tokenizer used for tokenizing the caption.
        max_words: Max word length retained. Any more than the value will be truncated. Default: 30
        feature_framerate: sampling rate in second. Default: 1.0
        max_frames: Max frame sampled. Any more than the value will be ignored. Default: 100
        split_type: Either "train", "val", or "test". Default: ""
    """
    def __init__(
            self,
            json_path,
            features_path,
            text_embedding_path,
            tokenizer,
            max_words=30,
            topk=1,
            feature_framerate=1.0,
            max_frames=100,
            split_type="",
    ):

        self.data = json.load(open(json_path, 'r'))
        self.feature_dict = pickle.load(open(features_path, 'rb'))

        # self.noun_prior_feats = pickle.load(open("Ns_Detect/MSRVTT_video_nouns_aligned_features430.pkl","rb"))
        # self.verb_prior_feats = pickle.load(open("Vs_Detect/MSRVTT_video_verbs_aligned_features430.pkl", "rb"))
        # self.cluster_data = pickle.load(open(r"C:\Users\admin\Desktop\YangYang\V2T-NVvCap-main\Alignment_module\checkpoints\511331\data.pkl","rb"))

        self.noun_prior_feats = pickle.load(open(r"C:\Users\admin\Desktop\YangYang\V2T-CLIP4Caption-Reproduction-main\Ns_Detect\MSRVTT_video_nouns_aligned_features430.pkl","rb"))
        self.verb_prior_feats = pickle.load(open(r"C:\Users\admin\Desktop\YangYang\V2T-CLIP4Caption-Reproduction-main\Vs_Detect\MSRVTT_video_verbs_aligned_features430.pkl", "rb"))

        self.text_data = pickle.load(
            open(r"C:\Users\admin\Desktop\YangYang\V2T-NVvCap-main\data\text_feat_dict.pkl", 'rb'))

        # self.topic_feat = torch.load(
        #     text_embedding_path)

        # self.topic_feat = torch.load(text_embedding_path, map_location="cpu").float()

        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.tokenizer = tokenizer
        self.topk = topk

        self.feature_size = self.feature_dict[next(iter(self.feature_dict))].shape[-1]

        assert split_type in ["train", "val", "test"]
        # Train: video0 : video6512 (6513)
        # Val: video6513 : video7009 (497)
        # Test: video7010 : video9999 (2990)
        video_ids = [self.data['videos'][idx]['video_id'] for idx in range(len(self.data['videos']))]
        split_dict = {"train": video_ids[:6513], "val": video_ids[6513:6513 + 497], "test": video_ids[6513 + 497:]}
        choiced_video_ids = split_dict[split_type]

        self.sample_len = 0
        self.sentences_dict = {}
        self.video_sentences_dict = defaultdict(list)
        if split_type == "train":  # expand all sentence to train
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
                    self.video_sentences_dict[itm['video_id']].append(itm['caption'])
        elif split_type == "val" or split_type == "test":
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    self.video_sentences_dict[itm['video_id']].append(itm['caption'])
            for vid in choiced_video_ids:
                self.sentences_dict[len(self.sentences_dict)] = (vid, self.video_sentences_dict[vid][0])
        else:
            raise NotImplementedError

        self.sample_len = len(self.sentences_dict)



    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]
        pairs_text = np.zeros((k, self.max_words), dtype=np.long)

        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_output_caption_ids = np.zeros((k, self.max_words), dtype=np.long)
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.long)

        for i, video_id in enumerate(choice_video_ids):
            words = []
            words = ["[CLS]"] + words
            total_length_with_CLS = self.max_words - 1
            if len(words) > total_length_with_CLS:
                words = words[:total_length_with_CLS]
            words = words + ["[SEP]"]


            input_ids = self.tokenizer.convert_tokens_to_ids(words)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
            assert len(input_ids) == self.max_words

            pairs_text[i] = np.array(input_ids)

            # For generate captions
            if caption is not None:
                caption_words = self.tokenizer.tokenize(caption)
            else:
                caption_words = self._get_single_text(video_id)
            if len(caption_words) > total_length_with_CLS:
                caption_words = caption_words[:total_length_with_CLS]
            input_caption_words = ["[CLS]"] + caption_words
            output_caption_words = caption_words + ["[SEP]"]

            # For generate captions
            input_caption_ids = self.tokenizer.convert_tokens_to_ids(input_caption_words)
            output_caption_ids = self.tokenizer.convert_tokens_to_ids(output_caption_words)
            decoder_mask = [1] * len(input_caption_ids)
            while len(input_caption_ids) < self.max_words:
                input_caption_ids.append(0)
                output_caption_ids.append(0)
                decoder_mask.append(0)
            assert len(input_caption_ids) == self.max_words
            assert len(output_caption_ids) == self.max_words
            assert len(decoder_mask) == self.max_words

            pairs_input_caption_ids[i] = np.array(input_caption_ids)
            pairs_output_caption_ids[i] = np.array(output_caption_ids)
            pairs_decoder_mask[i] = np.array(decoder_mask)

        return pairs_text, np.array([]), np.array([]), np.array([]), np.array([]), \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, choice_video_ids

    def _get_single_text(self, video_id):
        rind = random.randint(0, len(self.sentences[video_id]) - 1)
        caption = self.sentences[video_id][rind]
        words = self.tokenizer.tokenize(caption)
        return words

    # def _get_video(self, choice_video_ids):
    #     prior_len = self.topk + 2  # topic topk + noun + verb
    #     total_len = self.max_frames + prior_len
    #
    #     video_mask = np.zeros((len(choice_video_ids), total_len), dtype=np.long)
    #     video = np.zeros((len(choice_video_ids), total_len, self.feature_size), dtype=np.float)
    #
    #     for i, video_id in enumerate(choice_video_ids):
    #         # 原始视频帧特征，不包含 noun/verb/topic
    #         video_slice = self.feature_dict[video_id]  # (T, 512)
    #
    #         if video_slice.shape[0] > self.max_frames:
    #             video_slice = video_slice[:self.max_frames]
    #
    #         frame_len = video_slice.shape[0]
    #
    #         frame_video = np.zeros((self.max_frames, self.feature_size), dtype=np.float)
    #         frame_mask = np.zeros((self.max_frames,), dtype=np.long)
    #
    #         frame_video[:frame_len] = video_slice
    #         frame_mask[:frame_len] = 1
    #
    #         # ========== topic 检索 ==========
    #         video_tensor = torch.from_numpy(frame_video).unsqueeze(0).float()  # (1, max_frames, 512)
    #         video_mask_tensor = torch.from_numpy(frame_mask).unsqueeze(0).float()  # (1, max_frames)
    #
    #         _, _, topk_topic_feat = get_video_topic_topk(
    #             video_tensor,
    #             self.topic_feat,
    #             video_mask_tensor,
    #             topk=self.topk,
    #         )
    #
    #         topk_topic_feat = topk_topic_feat.squeeze(0).cpu().numpy()  # (topk, 512)
    #
    #         # ========== noun / verb ==========
    #         video_noun_feat = self.noun_prior_feats[video_id]  # 通常 (1, 512)
    #         video_verb_feat = self.verb_prior_feats[video_id]  # 通常 (1, 512)
    #
    #         if video_noun_feat.ndim == 1:
    #             video_noun_feat = video_noun_feat.reshape(1, -1)
    #
    #         if video_verb_feat.ndim == 1:
    #             video_verb_feat = video_verb_feat.reshape(1, -1)
    #
    #         video_noun_feat = video_noun_feat[:1]
    #         video_verb_feat = video_verb_feat[:1]
    #
    #         # ========== 拼接：topic + noun + verb + frames ==========
    #         video_with_prior = np.concatenate(
    #             [
    #                 topk_topic_feat,
    #                 video_noun_feat,
    #                 video_verb_feat,
    #                 frame_video,
    #             ],
    #             axis=0,
    #         )
    #
    #         mask_with_prior = np.concatenate(
    #             [
    #                 np.ones((self.topk,), dtype=np.long),
    #                 np.ones((1,), dtype=np.long),
    #                 np.ones((1,), dtype=np.long),
    #                 frame_mask,
    #             ],
    #             axis=0,
    #         )
    #
    #         video[i, :video_with_prior.shape[0]] = video_with_prior
    #         video_mask[i, :mask_with_prior.shape[0]] = mask_with_prior
    #
    #     return video, video_mask, np.array([]), np.array([])


    def _get_video(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.long)

        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros((len(choice_video_ids), self.max_frames, self.feature_size), dtype=np.float)

        for i, video_id in enumerate(choice_video_ids):
            # video_slice = self.feature_dict[video_id]
            #
            video_noun_feat = self.noun_prior_feats[video_id]

            video_verb_feat = self.verb_prior_feats[video_id]
            # L = video_slice.shape[0]  # 视频长度 15~25
            #
            # # 1. 随机选一个插入位置（0~L 之间，任意位置）
            # insert_pos = random.randint(0, L)
            #
            # # 2. 把 noun + verb 拼在一起（顺序固定，也可以随机，我给你固定版）
            # insert_feats = np.concatenate([video_noun_feat, video_verb_feat], axis=0)  # (2,512)
            #
            # # 3. 在随机位置切开，插入，再拼接
            # video_slice = np.concatenate([
            #     video_slice[:insert_pos],  # 前半段
            #     insert_feats,  # 插入 noun + verb
            #     video_slice[insert_pos:]  # 后半段
            # ], axis=0)
            video_slice = self.feature_dict[video_id]


            # video_noun_feat = self.noun_p

            # print(video_noun_feat.shape)
            # print(video_verb_feat.shape)
            # video_slice = np.concatenate([video_verb_feat, video_slice], axis=0)
            video_slice = np.concatenate([video_noun_feat, video_verb_feat, video_slice], axis=0)
            # video_slice = np.concatenate([video_verb_feat, video_noun_feat, video_slice], axis=0)

            if self.max_frames < video_slice.shape[0]:
                video_slice = video_slice[:self.max_frames]

            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                print("video_id: {}".format(video_id))
            else:
                video[i][:slice_shape[0]] = video_slice
                video_mask[i, :slice_shape[0]] = 1

            # video = video.squeeze(1)

            # video_mask = video_mask.squeeze()
            # topk_idx, topk_score, topk_feat = get_video_topic_topk(torch.from_numpy(video), self.topic_feat, torch.from_numpy(video_mask), topk=self.topk)
            # topk_mask = torch.ones(topk_feat.shape[0], topk_feat.shape[1])
            #
            # video = torch.cat([topk_feat, torch.from_numpy(video)], dim=1)
            # video_mask = torch.cat([topk_mask, torch.from_numpy(video_mask)], dim=1)
            #
            # if self.max_frames < video.shape[1]:
            #     video = video[:,:self.max_frames,:]
            #     video_mask = video_mask[:, :self.max_frames,]
        return video, video_mask, np.array([]), np.array([])



    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]
        noun_feats = self.text_data[video_id]["text_noun_feats"][:5, :]
        verb_feats = self.text_data[video_id]["text_verb_feats"][:3, :]
        # prior_np = self.prior_data[video_id]  # 从pkl读取: (1, 512) —— numpy

        # 1. 给先验增加一个维度 → (1, 1, 512)
        # prior_feat = np.expand_dims(prior_np, axis=1)  # 只动这里！

        # 2. 先验掩码 → (1, 1)
        prior_mask = np.ones((1, 1), dtype=np.float32)

        pairs_text, pairs_mask, pairs_segment, \
        pairs_masked_text, pairs_token_labels, \
        pairs_input_caption_ids, pairs_decoder_mask, \
        pairs_output_caption_ids, choice_video_ids = self._get_text(video_id, caption)

        video, video_mask, masked_video, video_labels_index = self._get_video(choice_video_ids)


        pairs_mask, pairs_segment, pairs_masked_text, pairs_token_labels, masked_video, video_labels_index = np.array([]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([])

        # video = np.concatenate([video, prior_feat], axis=1)  # (1, 20, 512)
        # video_mask = np.concatenate([video_mask, prior_mask], axis=1)  # (1, 20)
        # concat_feat = np.concatenate([video, prior_feat], axis=1)  # (1, 20, 512)
        # concat_mask = np.concatenate([video_mask, prior_mask], axis=1)  # (1, 20)

        return pairs_text, pairs_mask, pairs_segment, video, video_mask,noun_feats,verb_feats,\
               pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
               pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids