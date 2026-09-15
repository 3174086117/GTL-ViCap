import torch
import pickle
from CLIP import clip
from torch import nn
import numpy as np

from tqdm import tqdm
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "ViT-B/32"

# ========================
# 模型必须和训练完全一致
# ========================
class VideoVerbAlignModel(nn.Module):
    def __init__(self, embed_dim=512, num_layers=2, num_heads=4):
        super().__init__()
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

        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, video_embeds, video_mask=None):
        B, T, D = video_embeds.shape
        x = video_embeds + self.pos_emb[:, :T]

        if video_mask is not None:
            key_padding_mask = ~video_mask.squeeze(1).bool()
            x = self.temporal_encoder(x, src_key_padding_mask=key_padding_mask)
        else:
            x = self.temporal_encoder(x)

        if video_mask is not None:
            mask = video_mask.squeeze(1).unsqueeze(-1)
            sum_feat = (x * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-8)
            feat = sum_feat / count
        else:
            feat = x.mean(dim=1)

        feat = self.ln(self.proj(feat))
        return feat


def extract_and_save_aligned_features(
    model_path="your_pre_train_model_weight_path",
    feature_path="your pre-train extracted dataset feats path",
    data_path="your dataset video id path",

    save_path="your save alignment feats path",
    max_frames=20
):
    # 1. 加载CLIP
    clip_model, _ = clip.load(CLIP_MODEL_NAME, device=DEVICE)
    clip_model = clip_model.float()
    clip_model.eval()

    # 2. 加载训练好的对齐模型
    video_fusion = VideoVerbAlignModel(embed_dim=512, num_layers=1, num_heads=4).to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE)
    video_fusion.load_state_dict(checkpoint["video_fusion"])
    video_fusion.eval()

    # 3. 加载视频原始特征 & 标注信息
    video_features = pickle.load(open(feature_path, "rb"))
    video_id_list = pd.read_csv(data_path)


    # 4. 开始批量提取
    result_dict = {}

    print("\n🚀 开始批量提取【视频动词对齐特征】...")
    for item in tqdm(video_id_list.video_id):
        # video_id = item["video_id"]
        video_id = item
        # 取出该视频的原始帧特征
        video_data = video_features[video_id]

        # 填充到固定帧数
        video = np.zeros((max_frames, 512), dtype=np.float32)
        video[:video_data.shape[0]] = video_data
        video_mask = np.zeros(max_frames, dtype=np.float32)
        video_mask[:video_data.shape[0]] = 1.0

        # 转tensor
        video = torch.from_numpy(video).unsqueeze(0).to(DEVICE).float()
        video_mask = torch.from_numpy(video_mask).unsqueeze(0).unsqueeze(0).to(DEVICE).float()

        # 推理：得到动词对齐后的视频特征
        with torch.no_grad():
            aligned_feat = video_fusion(video, video_mask)

        # 保存为numpy

        result_dict[video_id] = aligned_feat.cpu().numpy()

    # 保存字典
    with open(save_path, "wb") as f:
        pickle.dump(result_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n🎉 提取完成！已保存到：{save_path}")
    print(f"✅ 总视频数量：{len(result_dict)}")
    return result_dict

if __name__ == "__main__":
    extract_and_save_aligned_features()