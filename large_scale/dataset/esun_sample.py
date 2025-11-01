# -*- coding: utf-8 -*-
"""
E.SUN 資料集子圖採樣
使用 Random Walk with Restart (RWR) 為每個節點生成固定大小的子圖
"""
import dgl
import numpy as np
import torch
import pickle
from tqdm import tqdm

def generate_rwr_subgraph(dgl_graph, subgraph_size, restart_prob=0.9):
    """
    使用 Random Walk with Restart 為每個節點生成子圖
    
    Args:
        dgl_graph: DGL 圖
        subgraph_size: 子圖大小 (包含中心節點)
        restart_prob: 重啟概率
    
    Returns:
        subgraphs: list of lists, 每個節點的鄰居列表
    """
    num_nodes = dgl_graph.number_of_nodes()
    all_idx = torch.arange(num_nodes)
    reduced_size = subgraph_size - 1  # 扣除中心節點
    
    print(f"開始為 {num_nodes} 個節點生成子圖...")
    print(f"子圖大小: {subgraph_size}, 重啟概率: {restart_prob}")
    
    subv = []
    
    # 批次處理以提高效率
    batch_size = 1000
    num_batches = (num_nodes + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="生成子圖"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, num_nodes)
        batch_nodes = all_idx[start_idx:end_idx]
        
        # Random walk
        traces, _ = dgl.sampling.random_walk(
            dgl_graph, 
            batch_nodes, 
            restart_prob=restart_prob, 
            length=subgraph_size * 3
        )
        
        for i, trace in enumerate(traces):
            node_idx = start_idx + i
            
            # 獲取唯一節點
            unique_nodes = torch.unique(trace[trace >= 0], sorted=False).tolist()
            
            # 如果採樣的節點不夠，重試
            retry_time = 0
            while len(unique_nodes) < reduced_size and retry_time < 10:
                cur_trace, _ = dgl.sampling.random_walk(
                    dgl_graph, 
                    torch.tensor([node_idx]), 
                    restart_prob=0.5, 
                    length=subgraph_size * 5
                )
                unique_nodes = torch.unique(cur_trace[0][cur_trace[0] >= 0], sorted=False).tolist()
                retry_time += 1
            
            # 如果還是不夠，用重複填充
            if len(unique_nodes) < reduced_size:
                unique_nodes = (unique_nodes * ((reduced_size // len(unique_nodes)) + 1))[:reduced_size]
            
            # 截取到所需大小並加入中心節點
            unique_nodes = unique_nodes[:reduced_size]
            unique_nodes.append(node_idx)
            
            subv.append(unique_nodes)
    
    print(f"子圖生成完成! 總共 {len(subv)} 個子圖")
    return subv


def extract_subgraph_features(dgl_graph, subgraphs, subgraph_size):
    """
    提取每個子圖的特徵矩陣
    
    Args:
        dgl_graph: DGL 圖
        subgraphs: 子圖節點列表
        subgraph_size: 子圖大小
    
    Returns:
        sample_features: [num_nodes, subgraph_size, feature_dim]
    """
    features = dgl_graph.ndata['feature']
    num_nodes = len(subgraphs)
    feature_dim = features.shape[1]
    
    sample_features = np.zeros((num_nodes, subgraph_size, feature_dim), dtype=np.float32)
    
    print("正在提取子圖特徵...")
    for i in tqdm(range(num_nodes), desc="提取特徵"):
        subgraph_nodes = subgraphs[i]
        
        # 提取子圖節點的特徵
        # 注意: 中心節點在最後一個位置
        for j, node_idx in enumerate(subgraph_nodes):
            sample_features[i, j, :] = features[node_idx].numpy()
    
    return sample_features


def process_esun_large_scale(graph_path='esun_graph.bin', 
                              subgraph_size=10,
                              save_prefix='esun'):
    """
    處理大規模 E.SUN 資料集
    
    Args:
        graph_path: DGL 圖檔案路徑
        subgraph_size: 子圖大小
        save_prefix: 儲存檔案前綴
    """
    print("=" * 50)
    print("E.SUN 大規模資料集處理")
    print("=" * 50)
    
    # 載入圖
    print(f"\n1. 載入圖資料: {graph_path}")
    graphs, _ = dgl.load_graphs(graph_path)
    graph = graphs[0]
    
    print(f"   節點數: {graph.num_nodes():,}")
    print(f"   邊數: {graph.num_edges():,}")
    print(f"   特徵維度: {graph.ndata['feature'].shape[1]}")
    
    # 生成子圖
    print(f"\n2. 生成子圖 (大小={subgraph_size})")
    subgraphs = generate_rwr_subgraph(graph, subgraph_size=subgraph_size)
    
    # 提取特徵
    print(f"\n3. 提取子圖特徵")
    sample_features = extract_subgraph_features(graph, subgraphs, subgraph_size)
    
    # 提取標籤
    labels = graph.ndata['label'].numpy()
    train_mask = graph.ndata['train_mask'].numpy()
    test_mask = graph.ndata['test_mask'].numpy()
    
    print(f"\n4. 儲存處理後的資料")
    
    # 儲存特徵
    feature_file = f'{save_prefix}_feature_{subgraph_size}.npy'
    np.save(feature_file, sample_features)
    print(f"   特徵檔案: {feature_file}")
    print(f"   特徵形狀: {sample_features.shape}")
    
    # 儲存標籤
    label_file = f'{save_prefix}_label_{subgraph_size}.npy'
    np.save(label_file, labels)
    print(f"   標籤檔案: {label_file}")
    
    # 儲存 mask
    mask_file = f'{save_prefix}_mask_{subgraph_size}.pkl'
    with open(mask_file, 'wb') as f:
        pickle.dump({
            'train_mask': train_mask,
            'test_mask': test_mask
        }, f)
    print(f"   Mask 檔案: {mask_file}")
    
    # 儲存子圖結構
    subgraph_file = f'{save_prefix}_subgraphs_{subgraph_size}.pkl'
    with open(subgraph_file, 'wb') as f:
        pickle.dump(subgraphs, f)
    print(f"   子圖檔案: {subgraph_file}")
    
    print("\n" + "=" * 50)
    print("處理完成!")
    print("=" * 50)
    print(f"\n資料集統計:")
    print(f"  總節點數: {len(labels):,}")
    print(f"  警示帳戶數 (label=1): {np.sum(labels == 1):,} ({np.sum(labels == 1)/len(labels)*100:.2f}%)")
    print(f"  未標記帳戶數 (label=-1): {np.sum(labels == -1):,} ({np.sum(labels == -1)/len(labels)*100:.2f}%)")
    print(f"  訓練樣本數 (few-shot可用): {np.sum(train_mask):,}")
    print(f"  測試樣本數 (需要預測): {np.sum(test_mask):,}")
    print(f"  子圖大小: {subgraph_size}")
    print(f"  特徵維度: {sample_features.shape[2]}")
    
    return sample_features, labels, train_mask, test_mask


if __name__ == "__main__":
    # 處理 E.SUN 資料集
    # 您可以調整子圖大小,建議範圍: 7-15
    process_esun_large_scale(
        graph_path='esun_graph.bin',
        subgraph_size=10,  # 可調整
        save_prefix='esun'
    )