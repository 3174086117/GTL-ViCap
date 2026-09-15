import json
import tqdm
import torch
from CLIP import clip
import argparse
# 设备配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_NAME = "ViT-B/32"

def CLIP_MSRVTT_text(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 加载 CLIP 模型
    clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
    clip_model.eval()  # 推理模式

    text_features = []

    # 批量编码文本
    with torch.no_grad():
        for d in tqdm.tqdm(data):
            # 分词 + 移到设备
            text_tokens = clip.tokenize([d]).to(DEVICE)  # 必须包成列表！
            # 提取特征
            feat = clip_model.encode_text(text_tokens)
            # 归一化（CLIP 官方标准做法）
            # feat = feat / feat.norm(dim=-1, keepdim=True)

            text_features.append(feat.cpu())  # 存到 CPU 避免爆显存

    # 把所有特征拼成一个张量
    text_features = torch.cat(text_features, dim=0)
    return text_features


def CLIP_MSVD_text(json_path,train_path):
    # 加载 caption 数据
    # with open("new_V_N_MSVD.json", "r", encoding="utf-8") as f:
    #     data = json.load(f)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(train_path, "r")as f:
        train_list = f.readlines()

    train_list = [vid[:-1] for vid in train_list]

    # 加载 CLIP 模型
    clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=DEVICE)
    clip_model.eval()  # 推理模式

    text_features = []

    # 批量编码文本
    with torch.no_grad():
        for d in tqdm.tqdm(data):

            vid = d["video_id"]
            if vid in train_list:

                caption = d["caption"]

                # 分词 + 移到设备
                text_tokens = clip.tokenize([caption]).to(DEVICE)  # 必须包成列表！
                # 提取特征
                feat = clip_model.encode_text(text_tokens)
                # print(feat.shape)
                # 归一化（CLIP 官方标准做法）
                # feat = feat / feat.norm(dim=-1, keepdim=True)

                text_features.append(feat.cpu())  # 存到 CPU 避免爆显存

    # 把所有特征拼成一个张量

    text_features = torch.cat(text_features, dim=0)
    return text_features

def index_points(points, idx):
    """Sample features following the index.
    Returns:
        new_points:, indexed points data, [B, S, C]

    Args:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def merge_tokens(token_dict, idx_cluster, dist_to_center, cluster_num):
    """
    无学习参数：距离聚类中心越近权重越大，越远权重越小
    权重公式: weight = 1 / (dist + 1e-6)
    """
    x = token_dict['x']
    idx_token = token_dict['idx_token']
    agg_weight = token_dict['agg_weight']

    B, N, C = x.shape

    # 距离反比权重：近大远小
    token_weight = 1.0 / (dist_to_center.unsqueeze(-1) + 1e-6)

    idx_batch = torch.arange(B, device=x.device)[:, None]
    idx = idx_cluster + idx_batch * cluster_num

    # 按簇累加权重
    all_weight = token_weight.new_zeros(B * cluster_num, 1)
    all_weight.index_add_(dim=0, index=idx.reshape(B * N),
                          source=token_weight.reshape(B * N, 1))
    all_weight = all_weight + 1e-6
    norm_weight = token_weight / all_weight[idx]

    # 加权聚合特征
    x_merged = x.new_zeros(B * cluster_num, C)
    source = x * norm_weight
    x_merged.index_add_(dim=0, index=idx.reshape(B * N),
                        source=source.reshape(B * N, C).type(x.dtype))
    x_merged = x_merged.reshape(B, cluster_num, C)

    idx_token_new = index_points(idx_cluster[..., None], idx_token).squeeze(-1)
    weight_t = index_points(norm_weight, idx_token)
    agg_weight_new = agg_weight * weight_t
    agg_weight_new = agg_weight_new / (agg_weight_new.max(dim=1, keepdim=True)[0] + 1e-6)

    out_dict = {}
    out_dict['x'] = x_merged
    out_dict['token_num'] = cluster_num
    out_dict['idx_token'] = idx_token_new
    out_dict['agg_weight'] = agg_weight_new
    out_dict['mask'] = None
    return out_dict


def aggregate_and_save(feats, target_num, save_path="merged_features.pt", block_size=2000):
    print("开始聚合，原始形状:", feats.shape)
    device = feats.device
    x = feats.unsqueeze(0)  # [1, N, 512]
    B, N, C = x.shape

    # 分配结果
    idx_cluster = torch.zeros(N, dtype=torch.long, device=device)
    dist_to_center = torch.zeros(N, dtype=torch.float32, device=device)

    # 随机选聚类中心（不爆显存）
    rand_idx = torch.randperm(N)[:target_num].to(device)
    centers = x[:, rand_idx]

    # 分块算距离
    for i in range(0, N, block_size):
        end = min(i + block_size, N)
        block = x[:, i:end]
        dist = torch.cdist(block, centers).squeeze(0)
        min_dist, min_idx = dist.min(dim=-1)
        idx_cluster[i:end] = min_idx
        dist_to_center[i:end] = min_dist

    token_dict = {
        "x": x,
        "idx_token": torch.arange(N, device=device).unsqueeze(0),
        "agg_weight": torch.ones(1, N, 1, device=device),
        "mask": None
    }

    # 融合
    out_dict = merge_tokens(token_dict, idx_cluster.unsqueeze(0), dist_to_center.unsqueeze(0), target_num)
    merged = out_dict["x"].squeeze(0)

    # ===================== 保存到文件 =====================
    torch.save(merged.cpu(), save_path)
    print(f"✅ 聚合完成！形状: {merged.shape}")
    print(f"✅ 已保存到: {save_path}")

    return merged

def get_args(description='CaptionGenerator'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--Choice_dataset", type=str, default="MSVD", help="Choice dataset in MSVD or MSRVTT")
    parser.add_argument('--MSRVTT_json_path', type=str,
                        default=r"D:\Desktop\YangYang\clip4caption-main\dataset\MSRVTT\captions.json",
                        help='MSRVTT caption file path')
    parser.add_argument('--MSVD_json_path', type=str,
                        default=r"D:\Desktop\YangYang\clip4caption-main\dataset\MSVD\MSVD\new_V_N_MSVD.json",
                        help='MSVD caption file path')
    parser.add_argument('--MSVD_train_path', type=str,
                        default=r"D:\Desktop\YangYang\clip4caption-main\dataset\MSVD\train_list_mapping.txt",
                        help='MSVD train mapping file path')
    parser.add_argument('--save_text_embedding',default=False,help="whether save text embeddings")
    parser.add_argument('--target_topic_num',default=1000,help="number of the cluster topic")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.Choice_dataset == "MSVD":
        text_features = CLIP_MSVD_text(args.MSVD_json_path, args.MSVD_train_path)
    if args.Choice_dataset == "MSRVTT":
        text_features = CLIP_MSRVTT_text(args.MSRVTT_json_path)

    if args.save_text_embedding == True:
        torch.save(text_features, "text_embeddings.pt")

    aggregate_and_save(
        text_features,
        target_num=args.target_topic_num,
        save_path=args.Choice_dataset+"topic_embedding.pt",  # 保存路径
        block_size=2000
    )