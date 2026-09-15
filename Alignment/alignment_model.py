import pickle
import torch
from CLIP import clip
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import json
import numpy as np
import torch.nn.functional as F
# from .until import collate_fn
import argparse
import os
CLIP_MODEL_NAME = "ViT-B/32"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_args(description='CaptionGenerator'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--Choose_dataset", type=str, default="MSVD", help="Choose dataset in MSVD or MSRVTT")
    parser.add_argument("--align_type", type=str, default="nouns", help="Choose align types in nouns or verbs")
    args = parser.parse_args()
    return args
class VideoVerbAlignModel(nn.Module):
    """
    专为【视频特征 ↔ 动词特征对齐】设计的最优模型
    输入：video_embeds   [B, T, 512]  视频帧序列特征
    输出：video_feat     [B, 512]     对齐动词空间的视频特征
    """

    def __init__(self, embed_dim=512, num_layers=2, num_heads=4):
        super().__init__()

        # 1. 时序Transformer：建模视频动作、时序信息
        self.pos_emb = nn.Parameter(torch.randn(1, 100, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            dropout=0.1,
            activation="gelu"
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 2. 特征映射层：把视频特征映射到动词对齐空间（关键！）
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # 3. 归一化层
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, video_embeds, video_mask=None):
        B, T, D = video_embeds.shape

        # ---------------------------
        # 1. 加入位置编码
        # ---------------------------
        x = video_embeds + self.pos_emb[:, :T]

        # ---------------------------
        # 2. 视频时序建模（自动屏蔽padding帧）
        # ---------------------------
        if video_mask is not None:
            key_padding_mask = ~video_mask.squeeze(1).bool()
            x = self.temporal_encoder(x, src_key_padding_mask=key_padding_mask)
        else:
            x = self.temporal_encoder(x)

        # ---------------------------
        # 3. 均值池化（简单、稳定、有效）
        # ---------------------------
        if video_mask is not None:
            mask = video_mask.squeeze(1).unsqueeze(-1)
            sum_feat = (x * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-8)
            feat = sum_feat / count
        else:
            feat = x.mean(dim=1)

        # ---------------------------
        # 4. 映射到动词对齐空间（核心）
        # ---------------------------
        feat = self.ln(self.proj(feat))

        return feat



class MultiVerbClipLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temp = temperature

    def forward(self, video_feat, verb_feat):
        # 归一化（CLIP 必须）
        # verb_feat = verb_feat.mean(dim=1)
        video_feat = F.normalize(video_feat, dim=-1)  # [B, 512]
        verb_feat = F.normalize(verb_feat, dim=-1)   # [B, 512]

        video_feat = video_feat.squeeze()

        # print(video_feat.shape)
        # print(verb_feat.shape)

        # 相似度矩阵 [B, B]
        sim = video_feat @ verb_feat.T / self.temp

        # 标签：对角线是正样本
        labels = torch.arange(sim.size(0), device=sim.device)

        # 双向对比损失
        loss_v2t = F.cross_entropy(sim, labels)
        loss_t2v = F.cross_entropy(sim.T, labels)

        return (loss_v2t + loss_t2v) / 2

class msrvtt_dataset(Dataset):
    def __init__(self, features_path, json_path, align_type="nouns",max_frames=20,split="train"):
        self.data = pickle.load(open(features_path, "rb"))
        self.max_frames = max_frames
        self.captions = json.load(open(json_path, "r"))
        self.split = split
        self.align_type = align_type

        self.train_data = []
        self.val_data = []
        self.test_data = []
        for data in self.captions:
            vid = data["video_id"]
            vid_num = int(vid.replace("video", ""))
            if 0 <= vid_num <= 6512:
                self.train_data.append(data)
            elif 6513 <= vid_num <= 7009:
                self.val_data.append(data)
            elif 7010 <= vid_num <= 9999:
                self.test_data.append(data)


    def __len__(self):
        if self.split == "train":
            return len(self.train_data)
        elif self.split == "val":
            return len(self.val_data)
        else:
            return len(self.test_data)

    def __getitem__(self, item):
        if self.split == "train":
            caption = self.train_data[item]
        elif self.split == "val":
            caption = self.val_data[item]
        else:
            caption = self.test_data[item]
        verbs_list = caption[self.align_type]
        video_id = caption["video_id"]
        video_data = self.data[video_id]

        video = np.zeros((self.max_frames, 512), dtype=np.float32)
        video[:video_data.shape[0]] = video_data

        video_mask = np.zeros(self.max_frames, dtype=np.float32)
        video_mask[:video_data.shape[0]] = 1.0

        video = torch.from_numpy(video)
        video_mask = torch.from_numpy(video_mask).unsqueeze(0)

        return video_id, verbs_list, video, video_mask

class msvd_dataset(Dataset):
    def __init__(self, features_path, json_path, align_type="nouns",max_frames=20,split_type="train"):
        self.data = pickle.load(open(features_path, "rb"))
        self.max_frames = max_frames
        self.captions = json.load(open(json_path, "r"))
        self.align_type = align_type

        self.split_type = split_type
        with open(r"C:\Users\admin\Desktop\YangYang\V2T-CLIP4Caption-Reproduction-main\dataset\MSVD\train_list_mapping.txt","r")as f:
            self.train_list = f.readlines()
            self.train_list = [data[:-1] for data in self.train_list]
        with open(r"C:\Users\admin\Desktop\YangYang\V2T-CLIP4Caption-Reproduction-main\dataset\MSVD\val_list_mapping.txt","r")as f:
            self.val_list = f.readlines()
            self.val_list = [data[:-1] for data in self.val_list]
        with open(r"C:\Users\admin\Desktop\YangYang\V2T-CLIP4Caption-Reproduction-main\dataset\MSVD\test_list_mapping.txt","r")as f:
            self.test_list = f.readlines()
            self.test_list = [data[:-1] for data in self.test_list]
        self.train_data = []
        self.test_data = []
        self.val_data = []

        for caption in self.captions:
            vid = caption["video_id"]


            if vid in self.train_list:
                self.train_data.append(caption)
            elif vid in self.val_list:
                self.val_data.append(caption)
            else:
                self.test_data.append(caption)


    def __len__(self):
        if self.split_type == "train":
            return len(self.train_data)
        elif self.split_type == "val":
            return len(self.val_data)
        else:
            return len(self.test_data)

    def __getitem__(self, item):
        if self.split_type == "train":
            nouns_list = self.train_data[item][self.align_type]
            video_id = self.train_data[item]["video_id"]
        elif self.split_type == "val":
            nouns_list = self.val_data[item][self.align_type]
            video_id = self.val_data[item]["video_id"]
        else:
            nouns_list = self.test_data[item][self.align_type]
            video_id = self.test_data[item]["video_id"]
        # nouns_list = self.captions[item]["nouns"].split(" ")
        # video_id = self.captions[item]["video_id"]

        video_data = self.data[video_id]


        video = np.zeros((self.max_frames, 512), dtype=np.float32)
        video[:video_data.shape[0]] = video_data

        video_mask = np.zeros(self.max_frames, dtype=np.float32)
        video_mask[:video_data.shape[0]] = 1.0

        video = torch.from_numpy(video)
        video_mask = torch.from_numpy(video_mask).unsqueeze(0)

        return video_id, nouns_list, video, video_mask


if __name__ == "__main__":
    args = get_args()
    clip_model, _ = clip.load(CLIP_MODEL_NAME, DEVICE)
    clip_model = clip_model.float()
    for param in clip_model.parameters():
        param.requires_grad = False

    if args.Choose_dataset == "MSVD":
        json_path = r"C:\Users\admin\Desktop\YangYang\V2T-CLIP4Caption-Reproduction-main\N_Detect_msvd\new_V_N_MSVD.json"
        features_path = r"C:\Users\admin\Desktop\YangYang\V2T-CLIP4Caption-Reproduction-main\MSVD_CLIP4Clip_features.pickle"
        dataset = msvd_dataset(features_path, json_path, align_type=args.align_type, max_frames=20,split_type="train")

    if args.Choose_dataset == "MSRVTT":
        json_path = r"C:\Users\admin\Desktop\YangYang\UHCL-main\UHCL_data\dataset\MSRVTT\V_N_MSRVTT.json"
        features_path = r"C:\Users\admin\Desktop\YangYang\V2T-CLIP4Caption-Reproduction-main\MSRVTT_CLIP4Clip_features.pickle"

        dataset = msrvtt_dataset(features_path, json_path, align_type=args.align_type, max_frames=20, split_type="train")

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    if args.Choose_dataset == "MSVD":
        json_path = r"C:\Users\admin\Desktop\YangYang\V2T-CLIP4Caption-Reproduction-main\N_Detect_msvd\new_V_N_MSVD.json"
        features_path = r"C:\Users\admin\Desktop\YangYang\V2T-CLIP4Caption-Reproduction-main\MSVD_CLIP4Clip_features.pickle"
        eval_dataset = msvd_dataset(features_path, json_path, align_type=args.align_type, max_frames=20,split_type="val")

    if args.Choose_dataset == "MSRVTT":
        json_path = r"C:\Users\admin\Desktop\YangYang\UHCL-main\UHCL_data\dataset\MSRVTT\V_N_MSRVTT.json"
        features_path = r"C:\Users\admin\Desktop\YangYang\V2T-CLIP4Caption-Reproduction-main\MSRVTT_CLIP4Clip_features.pickle"

        eval_dataset = msrvtt_dataset(features_path, json_path, align_type=args.align_type, max_frames=20, split_type="val")




    eval_dataloader = DataLoader(eval_dataset, batch_size=32, shuffle=True)


    video_fusion = VideoVerbAlignModel(
        embed_dim=512,
        num_layers=1,  # 可改为 2/3
        num_heads=4
    ).to(DEVICE)

    # checkpoint = torch.load(r"C:\Users\admin\Desktop\YangYang\V2T-CLIP4Caption-Reproduction-main\Vs_Detect\verbs_video_mean_Liner429_6epoch.pth", map_location=DEVICE)
    # video_fusion.load_state_dict(checkpoint["video_fusion"])
    # print("✅ 成功加载预训练权重，继续训练...")

    optimizer = torch.optim.AdamW(
        video_fusion.parameters(),
        lr=5e-4,
        weight_decay=0.05
    )
    align_loss = MultiVerbClipLoss()

    # 训练
    print("开始训练：视频帧均值 + Transformer 版本\n")
    for epoch in range(10):
        print("start train>>>>>>>>>>>>>>>>>>>>>>>")
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/10")
        for step, batch in enumerate(pbar):
            vid, verbs_list, video, video_mask = batch

            video_fusion.train()

            video = video.to(DEVICE)
            video_mask = video_mask.to(DEVICE)



            video_feat = video_fusion(video, video_mask)

            verb_toks = [clip.tokenize(vs) for vs in verbs_list]
            verb_toks = torch.stack(verb_toks).to(DEVICE)

            B, K, L = verb_toks.shape
            verb_flat = verb_toks.reshape(B*K, L)
            verb_feat = clip_model.encode_text(verb_flat).reshape(B, K, -1).squeeze(1)

            # 损失


            loss = align_loss(video_feat, verb_feat)

            # 反向
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}", avg=f"{total_loss/(step+1):.3f}")
            if epoch % 2 == 0:
                torch.save({"video_fusion": video_fusion.state_dict()}, args.Choose_dataset+"_"+args.align_type+f"_align_{epoch}")

        print("start eval>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        total_loss = 0
        pbar = tqdm(eval_dataloader, desc=f"Epoch {epoch+1}/20")
        for step, batch in enumerate(pbar):
            vid, verbs_list, video, video_mask = batch
            video_fusion.eval()
            # print(verbs_list)
            # 设备
            video = video.to(DEVICE)
            video_mask = video_mask.to(DEVICE)


            with torch.no_grad():
                video_feat = video_fusion(video, video_mask)


            verb_toks = [clip.tokenize(vs) for vs in verbs_list]
            verb_toks = torch.stack(verb_toks).to(DEVICE)

            B, K, L = verb_toks.shape
            verb_flat = verb_toks.reshape(B*K, L)
            verb_feat = clip_model.encode_text(verb_flat).reshape(B, K, -1).squeeze(1)

            # 损失


            loss = align_loss(video_feat, verb_feat)


            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.3f}", avg=f"{total_loss/(step+1):.3f}")



    # 保存
    torch.save({"video_fusion": video_fusion.state_dict()}, args.Choose_dataset+"_"+args.align_type+f"_align_ending")
    # print("\n训练完成 → 已保存：verbs_video_mean_Liner1.pth")