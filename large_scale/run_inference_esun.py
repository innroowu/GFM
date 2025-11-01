"""
E.SUN 資料集推論腳本
支援 few-shot abnormal detection
"""
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score, precision_recall_curve
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import random
import pickle
from tqdm import tqdm
import sys
import os
import pandas as pd
from utils import *
import dgl

# 加入模組路徑
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


parser = argparse.ArgumentParser(description='E.SUN 資料集推論')
parser.add_argument('--seed', type=int, default=0, help='隨機種子')
parser.add_argument('--beta', type=float, default=6.0, help='正常/異常分數權重')
parser.add_argument('--subgraph_size', type=int, default=10, help='子圖大小')
parser.add_argument('--embedding_dim', type=int, default=300, help='嵌入維度')
parser.add_argument('--model_path', type=str, default='pretrain/model_residual2.pth', 
                    help='預訓練模型路徑')
parser.add_argument('--data_prefix', type=str, default='esun', help='資料檔案前綴')
parser.add_argument('--num_few_shot', type=int, default=100, 
                    help='Few-shot 樣本數 (從異常帳戶中選擇)')
args = parser.parse_args()

# 設定隨機種子
dgl.random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("=" * 80)
print(f"E.SUN 資料集推論 - Few-shot Abnormal Detection")
print("=" * 80)
print(f"\n參數設定:")
print(f"  隨機種子: {args.seed}")
print(f"  Beta 權重: {args.beta}")
print(f"  子圖大小: {args.subgraph_size}")
print(f"  Few-shot 數量: {args.num_few_shot}")
print(f"  模型路徑: {args.model_path}")

# 載入資料
print(f"\n{'=' * 80}")
print("1. 載入資料")
print(f"{'=' * 80}")

feature_file = f'./dataset/{args.data_prefix}_feature_{args.subgraph_size}.npy'
label_file = f'./dataset/{args.data_prefix}_label_{args.subgraph_size}.npy'
mask_file = f'./dataset/{args.data_prefix}_mask_{args.subgraph_size}.pkl'

print(f"特徵檔案: {feature_file}")
print(f"標籤檔案: {label_file}")
print(f"Mask 檔案: {mask_file}")

sample_feature = np.load(feature_file)
sample_labels = np.load(label_file)

with open(mask_file, 'rb') as f:
    masks = pickle.load(f)
    train_mask = masks['train_mask']
    test_mask = masks['test_mask']

nodes_num = sample_labels.shape[0]
feature_dim = sample_feature.shape[2]

print(f"\n資料統計:")
print(f"  總節點數: {nodes_num:,}")
print(f"  特徵維度: {feature_dim}")
print(f"  子圖大小: {args.subgraph_size}")
print(f"  已知警示帳戶數: {np.sum(sample_labels == 1):,}")
print(f"  未標記帳戶數: {np.sum(sample_labels == -1):,}")
print(f"  可用於 few-shot 的警示帳戶: {np.sum(train_mask):,}")
print(f"  需要預測的帳戶數: {np.sum(test_mask):,}")

# Few-shot 設定
print(f"\n{'=' * 80}")
print("2. Few-shot 設定")
print(f"{'=' * 80}")

# 從訓練集(警示帳戶)中隨機選擇 few-shot 樣本
abnormal_indices = np.where(train_mask)[0]
abnormal_indices_list = abnormal_indices.tolist()
random.shuffle(abnormal_indices_list)

if len(abnormal_indices) < args.num_few_shot:
    print(f"警告: 可用警示帳戶數 ({len(abnormal_indices)}) 少於設定的 few-shot 數量 ({args.num_few_shot})")
    few_shot_indices = abnormal_indices
else:
    few_shot_indices = np.array(abnormal_indices_list[:args.num_few_shot])

print(f"Few-shot 樣本索引 (前10個): {few_shot_indices[:10]}")
print(f"實際使用的 few-shot 數量: {len(few_shot_indices)}")

# 測試集: 所有未標記的帳戶
eval_mask = test_mask.copy()
eval_indices = np.where(eval_mask)[0]

print(f"需要預測的帳戶數: {len(eval_indices):,}")
print(f"注意: 由於這些是未標記帳戶,我們無法計算真實的評估指標")

# 轉換為 tensor
sample_feature = torch.FloatTensor(sample_feature)

# 載入模型
print(f"\n{'=' * 80}")
print("3. 載入預訓練模型")
print(f"{'=' * 80}")

try:
    model = torch.load(args.model_path)
    model.eval()
    print(f"模型載入成功: {args.model_path}")
except Exception as e:
    print(f"錯誤: 無法載入模型 - {e}")
    print(f"請確認模型檔案存在且路徑正確")
    sys.exit(1)

# 建立鄰接矩陣 (單位矩陣,因為子圖已經包含了鄰居資訊)
adj = torch.eye(args.subgraph_size, args.subgraph_size)
adj_norm = torch.FloatTensor(normalize_adj(adj).todense())

# 推論
print(f"\n{'=' * 80}")
print("4. 模型推論")
print(f"{'=' * 80}")

score_abnormal_all = []
score_normal_all = []

# 初始化 prompt
normal_prompt = torch.randn(args.embedding_dim)
abnormal_prompt = torch.randn(args.embedding_dim)

print("開始推論...")
with torch.no_grad():
    for i in tqdm(range(nodes_num), desc="推論進度"):
        feat = sample_feature[i, :, :]
        
        # 模型前向傳播
        logits_test, logits_test_residual, emb_test, emb_residual_test, \
        normal_prompt_test, abnormal_prompt_test = model(
            feat.unsqueeze(0), 
            adj_norm.unsqueeze(0), 
            adj, 
            normal_prompt, 
            abnormal_prompt
        )
        
        # 提取中心節點的 residual embedding (最後一個節點)
        residual_test = torch.squeeze(emb_residual_test[:, -1, :])
        
        # 計算與 prompt 的相似度
        score_abnormal = F.cosine_similarity(
            residual_test.unsqueeze(0), 
            abnormal_prompt_test.unsqueeze(0)
        )
        score_normal = F.cosine_similarity(
            residual_test.unsqueeze(0), 
            normal_prompt_test.unsqueeze(0)
        )
        
        # 組合分數
        score = torch.exp(score_abnormal) + args.beta * torch.exp(-score_normal)
        
        score_abnormal_all.append(score.detach().numpy())
        score_normal_all.append(score_normal.detach().numpy())

# 轉換為 numpy array
ano_score = np.squeeze(np.array(score_abnormal_all))

# 評估
print(f"\n{'=' * 80}")
print("5. 預測結果分析")
print(f"{'=' * 80}")

# 測試集的標籤和分數
test_labels = sample_labels[eval_indices]
test_scores = ano_score[eval_indices]

# 檢查是否有真實標籤可用於評估
has_ground_truth = np.any(test_labels != -1)

if has_ground_truth:
    # 如果有部分真實標籤(例如從 acct_test.csv 提供),可以計算指標
    valid_indices = test_labels != -1
    valid_labels = test_labels[valid_indices]
    valid_scores = test_scores[valid_indices]
    
    print(f"\n有真實標籤的測試樣本數: {np.sum(valid_indices):,}")
    
    auc = roc_auc_score(valid_labels, valid_scores)
    ap = average_precision_score(valid_labels, valid_scores, average='macro', pos_label=1)
    
    # 計算最佳閾值下的 F1, Precision, Recall
    precisions, recalls, thresholds = precision_recall_curve(valid_labels, valid_scores)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_f1 = f1_scores[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]
    
    print(f"\n評估指標 (有真實標籤的樣本):")
    print(f"  AUC-ROC: {auc:.4f}")
    print(f"  AP (Average Precision): {ap:.4f}")
    print(f"  最佳 F1 分數: {best_f1:.4f} (閾值={best_threshold:.4f})")
    print(f"  最佳 Precision: {best_precision:.4f}")
    print(f"  最佳 Recall: {best_recall:.4f}")
    
    print(f"\n異常分數分佈 (有真實標籤的樣本):")
    if np.sum(valid_labels == 0) > 0:
        print(f"  正常帳戶平均分數: {np.mean(valid_scores[valid_labels == 0]):.4f} ± {np.std(valid_scores[valid_labels == 0]):.4f}")
    if np.sum(valid_labels == 1) > 0:
        print(f"  異常帳戶平均分數: {np.mean(valid_scores[valid_labels == 1]):.4f} ± {np.std(valid_scores[valid_labels == 1]):.4f}")
else:
    print(f"\n注意: 測試集為未標記帳戶,無真實標籤可用於評估")
    print(f"將輸出預測結果供人工檢視")
    auc = None
    ap = None
    best_f1 = None
    best_precision = None
    best_recall = None
    best_threshold = None

# 分析所有預測結果的分佈
print(f"\n所有預測帳戶的異常分數統計:")
print(f"  平均分數: {np.mean(test_scores):.4f}")
print(f"  中位數: {np.median(test_scores):.4f}")
print(f"  標準差: {np.std(test_scores):.4f}")
print(f"  最小值: {np.min(test_scores):.4f}")
print(f"  最大值: {np.max(test_scores):.4f}")

# Top-K 高風險帳戶
print(f"\nTop-K 高風險帳戶:")
for k in [10, 50, 100, 200, 500]:
    if k <= len(test_scores):
        top_k_indices = np.argsort(test_scores)[-k:]
        top_k_scores = test_scores[top_k_indices]
        top_k_actual_indices = eval_indices[top_k_indices]
        print(f"  Top-{k:4d}: 平均分數 = {np.mean(top_k_scores):.4f}, "
              f"最低分數 = {np.min(top_k_scores):.4f}")
        
        # 如果有真實標籤,顯示精確度
        if has_ground_truth and np.any(test_labels[top_k_indices] != -1):
            valid_in_topk = test_labels[top_k_indices] != -1
            if np.sum(valid_in_topk) > 0:
                topk_precision = np.sum(test_labels[top_k_indices][valid_in_topk] == 1) / np.sum(valid_in_topk)
                print(f"             Precision (有標籤部分) = {topk_precision:.4f}")

# 儲存結果
print(f"\n{'=' * 80}")
print("6. 儲存結果")
print(f"{'=' * 80}")

# 儲存完整結果
results = {
    'scores': ano_score,
    'labels': sample_labels,
    'eval_indices': eval_indices,
    'few_shot_indices': few_shot_indices,
    'metrics': {
        'auc': auc,
        'ap': ap,
        'best_f1': best_f1,
        'best_precision': best_precision,
        'best_recall': best_recall,
        'best_threshold': best_threshold
    } if has_ground_truth else None,
    'args': vars(args)
}

result_file = f'{args.data_prefix}_results_fewshot{args.num_few_shot}.pkl'
with open(result_file, 'wb') as f:
    pickle.dump(results, f)
print(f"完整結果已儲存至: {result_file}")


# 載入帳戶映射
mapping_file = f'./dataset/{args.data_prefix}_graph_mapping.pkl'
if os.path.exists(mapping_file):
    with open(mapping_file, 'rb') as f:
        mapping = pickle.load(f)
        idx_to_acct = mapping['idx_to_acct']
    
    # 創建預測結果 DataFrame
    prediction_df = pd.DataFrame({
        'acct': [idx_to_acct[idx] for idx in eval_indices],
        'anomaly_score': test_scores,
        'rank': np.argsort(np.argsort(test_scores)[::-1]) + 1  # 排名 (1=最高風險)
    })
    
    # 按異常分數降序排列
    prediction_df = prediction_df.sort_values('anomaly_score', ascending=False)
    prediction_df = prediction_df.reset_index(drop=True)
    
    # 添加風險等級
    prediction_df['risk_level'] = pd.cut(
        prediction_df['anomaly_score'],
        bins=[-np.inf, 
              prediction_df['anomaly_score'].quantile(0.5),
              prediction_df['anomaly_score'].quantile(0.8),
              prediction_df['anomaly_score'].quantile(0.95),
              np.inf],
        labels=['Low', 'Medium', 'High', 'Critical']
    )
    
    # 儲存預測結果
    csv_file = f'{args.data_prefix}_predictions_fewshot{args.num_few_shot}.csv'
    prediction_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"預測結果已儲存至: {csv_file}")
    
    # 顯示前20個高風險帳戶
    print(f"\nTop 20 高風險帳戶預覽:")
    print(prediction_df.head(20).to_string(index=False))
    
    # 風險等級統計
    print(f"\n風險等級分佈:")
    risk_counts = prediction_df['risk_level'].value_counts().sort_index()
    for level in ['Critical', 'High', 'Medium', 'Low']:
        if level in risk_counts.index:
            count = risk_counts[level]
            pct = count / len(prediction_df) * 100
            print(f"  {level:8s}: {count:6,d} ({pct:5.2f}%)")
else:
    print(f"警告: 找不到帳戶映射檔案 {mapping_file},無法輸出帳戶ID")
    
    # 儲存索引版本
    prediction_df = pd.DataFrame({
        'node_index': eval_indices,
        'anomaly_score': test_scores,
        'rank': np.argsort(np.argsort(test_scores)[::-1]) + 1
    })
    prediction_df = prediction_df.sort_values('anomaly_score', ascending=False)
    
    csv_file = f'{args.data_prefix}_predictions_fewshot{args.num_few_shot}.csv'
    prediction_df.to_csv(csv_file, index=False)
    print(f"預測結果(僅索引)已儲存至: {csv_file}")

print(f"\n{'=' * 80}")
print("推論完成!")
print(f"{'=' * 80}")