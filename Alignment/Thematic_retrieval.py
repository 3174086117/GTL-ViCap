import torch
import torch.nn.functional as F

def get_video_topic_topk(
    video_feat: torch.Tensor,    # (B, 20, 512) 视频帧特征
    topic_feat: torch.Tensor,    # (1000, 512)  主题特征
    video_mask: torch.Tensor,   # (B, 20)      帧掩码
    topk: int = 5                # top-k 数量
):
    """
    返回：
        topk_indices:  (B, topk)  top-k 主题索引
        topk_scores:   (B, topk)  top-k 相似度分数
        topk_feat:     (B, topk, 512)  匹配到的 top-k 主题特征
    """

    B, T, C = video_feat.shape

    # ========== 1. 视频有效帧平均特征 ==========

    mask_expanded = video_mask.unsqueeze(-1).expand(B, T, C)
    video_feat_masked = video_feat * mask_expanded
    feat_sum = video_feat_masked.sum(dim=1)
    valid_frame_num = video_mask.sum(dim=1, keepdim=True).clamp(min=1e-8)
    video_global_feat = feat_sum / valid_frame_num

    # ========== 2. 余弦相似度 ==========
    video_norm = F.normalize(video_global_feat.float(), dim=-1)
    topic_norm = F.normalize(topic_feat.float(), dim=-1)
    similarity = torch.matmul(video_norm, topic_norm.transpose(0, 1))

    # ========== 3. Top-K 检索 ==========
    topk_scores, topk_indices = torch.topk(similarity, k=topk, dim=-1)
    # print(topk_scores)
    # ========== 4. 取出对应的 Top-K 主题特征 ==========
    # torch.gather 严格按索引取特征 → 输出形状 (B, topk, 512)
    topk_feat = torch.gather(
        topic_feat.unsqueeze(0).expand(B, topic_feat.shape[0], 512),  # (B, 1000, 512)
        dim=1,
        index=topk_indices.unsqueeze(-1).expand(B, topk, 512)  # (B, topk, 512)
    )

    return topk_indices, topk_scores, topk_feat


# 构造输入
if __name__ == "__main__":
    B = 8
    video_feat = torch.randn(B, 20, 512)
    topic_feat = torch.randn(1000, 512)
    video_mask = torch.randint(0, 2, (B, 20))

    # 推理
    topk_idx, topk_score, topk_feat = get_video_topic_topk(video_feat, topic_feat, video_mask, topk=5)

    # 输出形状
    print("topk 索引:", topk_idx.shape)  # torch.Size([8, 5])
    print("topk 分数:", topk_score.shape)  # torch.Size([8, 5])
    print("topk 主题特征:", topk_feat.shape)  # torch.Size([8, 5, 512]) ✅