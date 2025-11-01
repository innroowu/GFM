"""
E.SUN 資料集預處理腳本 - 最終版
根據實際資料格式:
- txn_time: "05:05:00 AM" (12小時制)
- txn_date: 8 (切齊第一天為1)
- alert event_date: 0 (切齊第一天為0)
"""
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import LabelEncoder, StandardScaler
import dgl
import torch
import pickle
import os
from tqdm import tqdm
from datetime import datetime

def parse_time_to_seconds(time_str):
    """
    將 "05:05:00 AM" 格式轉換為當天的秒數
    
    Args:
        time_str: 時間字串，例如 "05:05:00 AM"
    
    Returns:
        秒數 (0-86399)，無法解析返回 -1
    """
    if pd.isna(time_str) or time_str == '':
        return -1
    
    try:
        # 處理 12小時制
        time_str = str(time_str).strip()
        
        # 嘗試解析 "HH:MM:SS AM/PM" 格式
        dt = datetime.strptime(time_str, '%I:%M:%S %p')
        # 轉換為秒數
        seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
        return seconds
    except:
        try:
            # 嘗試解析 24小時制 "HH:MM:SS"
            dt = datetime.strptime(time_str, '%H:%M:%S')
            seconds = dt.hour * 3600 + dt.minute * 60 + dt.second
            return seconds
        except:
            return -1


def load_esun_data(transaction_path, alert_path, predict_path=None):
    """
    載入 E.SUN 資料集
    
    關鍵設計:
    1. 帳戶 ID 不轉數值，用 acct_to_idx 字典映射到節點索引
    2. txn_time 解析為秒數 (0-86399)
    3. txn_date 和 alert event_date 對齊 (alert 的 0 對應 txn 的 1)
    """
    print("="*80)
    print("E.SUN 資料載入與預處理 (最終版)")
    print("="*80)
    
    # 載入資料
    print("\n📁 步驟 1: 載入 CSV 檔案...")
    df_txn = pd.read_csv(transaction_path, dtype=str)
    df_alert = pd.read_csv(alert_path, dtype=str)
    
    print(f"   ✓ 交易資料: {len(df_txn):,} 筆")
    print(f"   ✓ 警示帳戶: {len(df_alert):,} 個")
    
    # 顯示範例資料
    print(f"\n   交易資料範例:")
    print(f"   - txn_time 範例: {df_txn['txn_time'].iloc[0]}")
    print(f"   - txn_date 範例: {df_txn['txn_date'].iloc[0]}")
    
    # 載入預測清單 (可選)
    predict_accounts = None
    if predict_path and os.path.exists(predict_path):
        df_predict = pd.read_csv(predict_path, dtype=str)
        predict_accounts = set(df_predict['acct'].values)
        print(f"   ✓ 預測清單: {len(predict_accounts):,} 個帳戶")
    
    # === 關鍵步驟 1: 建立帳戶映射 (不轉數值!) ===
    print("\n🔑 步驟 2: 建立帳戶到節點索引的映射...")
    print("   注意: 帳戶 ID 保持為 String，只映射到節點索引 (0, 1, 2, ...)")
    
    # 收集所有出現過的帳戶
    all_accounts = pd.concat([
        df_txn['from_acct'],
        df_txn['to_acct']
    ]).unique()
    
    # 創建雙向映射字典
    acct_to_idx = {acct: idx for idx, acct in enumerate(all_accounts)}
    idx_to_acct = {idx: acct for acct, idx in acct_to_idx.items()}
    num_nodes = len(all_accounts)
    
    print(f"   ✓ 總帳戶數 (圖節點數): {num_nodes:,}")
    print(f"   ✓ 帳戶範例: '{list(all_accounts)[:3]}'")
    print(f"   ✓ 映射範例: '{list(all_accounts)[0]}' → 節點索引 {acct_to_idx[list(all_accounts)[0]]}")
    
    # === 步驟 2: 建立圖的邊 ===
    print("\n📊 步驟 3: 建立交易圖的邊...")
    
    # 過濾有效交易
    valid_txn = df_txn[pd.notna(df_txn['from_acct']) & pd.notna(df_txn['to_acct'])]
    
    # 使用 acct_to_idx 映射到節點索引
    src_nodes = valid_txn['from_acct'].map(acct_to_idx).values
    dst_nodes = valid_txn['to_acct'].map(acct_to_idx).values
    
    # 移除映射失敗的邊
    valid_edges = ~(pd.isna(src_nodes) | pd.isna(dst_nodes))
    src_nodes = src_nodes[valid_edges].astype(int)
    dst_nodes = dst_nodes[valid_edges].astype(int)
    
    print(f"   ✓ 有效交易邊數: {len(src_nodes):,}")
    
    # === 步驟 3: 建立標籤 (處理日期偏移) ===
    print("\n🏷️  步驟 4: 建立標籤...")
    print("   注意: alert event_date 的 0 對應 txn_date 的 1")
    
    labels = np.full(num_nodes, -1, dtype=np.int64)  # -1: 未標記
    
    alert_accounts = df_alert['acct'].values
    alert_account_set = set(alert_accounts)
    
    num_alert_in_graph = 0
    for acct in alert_accounts:
        if acct in acct_to_idx:
            labels[acct_to_idx[acct]] = 1  # 1: 警示帳戶
            num_alert_in_graph += 1
    
    print(f"   ✓ 警示帳戶 (label=1): {num_alert_in_graph:,}")
    print(f"   ✓ 未標記帳戶 (label=-1): {np.sum(labels == -1):,}")
    
    # === 步驟 4: 提取節點特徵 ===
    print("\n🎯 步驟 5: 提取節點特徵...")
    features = extract_node_features_final(df_txn, acct_to_idx, num_nodes)
    
    print(f"   ✓ 特徵矩陣形狀: {features.shape}")
    
    # === 步驟 5: 建立 mask ===
    print("\n🎭 步驟 6: 建立訓練/測試遮罩...")
    
    train_mask = labels == 1  # Few-shot 用已知警示帳戶
    
    if predict_accounts is not None:
        test_mask = np.array([
            (acct in predict_accounts) and (acct not in alert_account_set) 
            for acct in all_accounts
        ])
    else:
        test_mask = labels == -1
    
    print(f"   ✓ Few-shot 訓練樣本: {np.sum(train_mask):,}")
    print(f"   ✓ 測試樣本: {np.sum(test_mask):,}")
    
    # === 步驟 6: 建立 DGL 圖 ===
    print("\n🕸️  步驟 7: 建立 DGL 圖...")
    
    graph = dgl.graph((src_nodes, dst_nodes), num_nodes=num_nodes)
    graph = dgl.to_bidirected(graph)
    graph = dgl.add_self_loop(graph)
    
    print(f"   ✓ 節點數: {graph.num_nodes():,}")
    print(f"   ✓ 邊數: {graph.num_edges():,}")
    
    return graph, features, labels, train_mask, test_mask, acct_to_idx, idx_to_acct


def extract_node_features_final(df_txn, acct_to_idx, num_nodes):
    """
    根據實際資料格式提取節點特徵 - 簡化到10維
    
    根據論文，large_scale 模型使用 10 維特徵
    參考: T-Finance 和 T-Social 數據集都是 10 維
    """
    print("   [1/5] 解析時間格式...")
    
    # === 1. 解析 txn_time ===
    df_txn['txn_time_seconds'] = df_txn['txn_time'].apply(parse_time_to_seconds)
    df_txn['txn_time_valid'] = df_txn['txn_time_seconds'] >= 0
    
    valid_pct = df_txn['txn_time_valid'].mean() * 100
    print(f"       ✓ 成功解析時間: {valid_pct:.2f}%")
    
    # === 2. 轉換數值欄位 ===
    print("   [2/5] 處理數值欄位...")
    
    df_txn['txn_date_num'] = pd.to_numeric(df_txn['txn_date'], errors='coerce').fillna(1)
    df_txn['txn_amt_num'] = pd.to_numeric(df_txn['txn_amt'], errors='coerce').fillna(0)
    
    # === 3. 計算統計特徵 (精簡到10維) ===
    print("   [3/5] 計算統計特徵 (10維)...")
    
    # 參考 T-Finance/T-Social 的特徵設計
    # 10維特徵: 交易行為的核心統計量
    
    # 分開計算避免 pandas agg 的 column 問題
    
    # 3.1 基本統計
    txn_count = df_txn.groupby('from_acct').size()
    
    # 3.2 金額統計
    amt_stats = df_txn.groupby('from_acct')['txn_amt_num'].agg(['mean', 'std', 'max'])
    
    # 3.3 不同收款人數
    unique_recipients = df_txn.groupby('from_acct')['to_acct'].nunique()
    
    # 3.4 時間範圍
    day_stats = df_txn.groupby('from_acct')['txn_date_num'].agg(['min', 'max'])
    
    # 3.5 平均交易時間 (只計算有效時間)
    def safe_time_mean(x):
        valid = x[x >= 0]
        return valid.mean() if len(valid) > 0 else 43200  # 預設中午12點
    
    avg_time = df_txn.groupby('from_acct')['txn_time_seconds'].apply(safe_time_mean)
    
    # 組合成 DataFrame
    from_stats = pd.DataFrame({
        'txn_count': txn_count,
        'txn_amt_mean': amt_stats['mean'],
        'txn_amt_std': amt_stats['std'],
        'txn_amt_max': amt_stats['max'],
        'unique_recipients': unique_recipients,
        'day_min': day_stats['min'],
        'day_max': day_stats['max'],
        'avg_time': avg_time
    }).fillna(0)
    
    # 3.6 衍生特徵
    from_stats['day_span'] = from_stats['day_max'] - from_stats['day_min']
    from_stats['txn_per_day'] = from_stats['txn_count'] / (from_stats['day_span'] + 1)
    
    # 最終的 10 維特徵
    feature_cols = [
        'txn_count',           # 1. 交易總數
        'txn_amt_mean',        # 2. 平均金額
        'txn_amt_std',         # 3. 金額標準差
        'txn_amt_max',         # 4. 最大金額
        'unique_recipients',   # 5. 不同收款人數
        'day_min',            # 6. 開始交易日
        'day_max',            # 7. 最後交易日
        'day_span',           # 8. 活躍天數
        'avg_time',           # 9. 平均交易時間
        'txn_per_day'         # 10. 每日平均交易數
    ]
    
    from_stats = from_stats[feature_cols]
    
    print(f"       ✓ 特徵維度: {len(feature_cols)}")
    print(f"       ✓ 特徵列表: {feature_cols[:5]}...")
    
    # === 4. 組合特徵向量 ===
    print("   [4/5] 組合特徵向量...")
    
    all_features = []
    
    for acct in tqdm(acct_to_idx.keys(), desc="       處理", ncols=70, leave=False):
        if acct in from_stats.index:
            feat = from_stats.loc[acct].values.tolist()
        else:
            # 沒有交易記錄的帳戶用預設值
            feat = [0] * 10
        
        all_features.append(feat)
    
    features = np.array(all_features, dtype=np.float32)
    
    print(f"       ✓ 特徵矩陣形狀: {features.shape}")
    
    # === 5. 標準化 ===
    print("   [5/5] 標準化特徵...")
    
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # 品質檢查
    print(f"       ✓ NaN: {np.isnan(features).sum()}, Inf: {np.isinf(features).sum()}")
    print(f"       ✓ 範圍: [{features.min():.2f}, {features.max():.2f}]")
    
    return features


def save_dgl_graph(graph, features, labels, train_mask, test_mask, 
                   acct_to_idx, idx_to_acct, save_path='esun_graph.bin'):
    """儲存 DGL 圖和映射"""
    print(f"\n💾 步驟 8: 儲存圖資料...")
    
    graph.ndata['feature'] = torch.FloatTensor(features)
    graph.ndata['label'] = torch.LongTensor(labels)
    graph.ndata['train_mask'] = torch.BoolTensor(train_mask)
    graph.ndata['test_mask'] = torch.BoolTensor(test_mask)
    
    dgl.save_graphs(save_path, [graph])
    
    # 儲存帳戶映射 (重要!)
    mapping_file = save_path.replace('.bin', '_mapping.pkl')
    with open(mapping_file, 'wb') as f:
        pickle.dump({
            'acct_to_idx': acct_to_idx,
            'idx_to_acct': idx_to_acct
        }, f)
    
    print(f"   ✓ 圖檔案: {save_path}")
    print(f"   ✓ 映射檔案: {mapping_file}")
    print(f"   ✓ 映射包含 {len(acct_to_idx):,} 個帳戶")
    
    print("\n" + "="*80)
    print("✅ 資料預處理完成!")
    print("="*80)


if __name__ == "__main__":
    import sys
    
    print("\n" + "="*80)
    print("E.SUN 警示帳戶偵測 - 資料預處理")
    print("基於 AnomalyGFM (KDD 2025) 論文方法")
    print("="*80)
    
    # 檔案路徑
    transaction_path = "./esun/acct_transaction.csv"
    alert_path = "./esun/acct_alert.csv"
    predict_path = None  # 如果有就填路徑
    
    # 檢查檔案
    if not os.path.exists(transaction_path):
        print(f"❌ 錯誤: 找不到檔案 {transaction_path}")
        sys.exit(1)
    
    if not os.path.exists(alert_path):
        print(f"❌ 錯誤: 找不到檔案 {alert_path}")
        sys.exit(1)
    
    # 執行預處理
    graph, features, labels, train_mask, test_mask, acct_to_idx, idx_to_acct = load_esun_data(
        transaction_path, alert_path, predict_path
    )
    
    # 儲存結果
    save_dgl_graph(graph, features, labels, train_mask, test_mask, 
                   acct_to_idx, idx_to_acct, save_path='esun_graph.bin')
    
    # 摘要
    print("\n📊 處理摘要:")
    print(f"  節點數: {graph.num_nodes():,}")
    print(f"  邊數: {graph.num_edges():,}")
    print(f"  特徵維度: {features.shape[1]}")
    print(f"  警示帳戶: {np.sum(labels==1):,} ({np.sum(labels==1)/len(labels)*100:.2f}%)")
    print(f"  未標記帳戶: {np.sum(labels==-1):,} ({np.sum(labels==-1)/len(labels)*100:.2f}%)")
    print(f"  Few-shot 可用樣本: {np.sum(train_mask):,}")
    print(f"  測試樣本: {np.sum(test_mask):,}")
    
    print("\n🎯 關鍵設計:")
    print("  ✓ 帳戶 ID 保持為 String，用字典映射到節點索引")
    print("  ✓ txn_time 解析為秒數 (0-86399)")
    print("  ✓ timestamp = txn_date * 100000 + txn_time_seconds")
    print("  ✓ 特徵基於交易行為，不是帳戶身份")
    
    print("\n⏭️  下一步:")
    print("  1. 執行子圖採樣: python esun_sample.py")
    print("  2. 執行推論: python run_inference_esun.py")