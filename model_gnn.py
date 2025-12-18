"""
GNN (图神经网络) 模型实现
专门用于THRB分子活性预测
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from rdkit import Chem
from rdkit.Chem import AllChem
import warnings
warnings.filterwarnings('ignore')


class MolecularGNN(nn.Module):
    """分子图神经网络模型"""
    
    def __init__(self, num_node_features=9, hidden_dim=128, num_classes=2, dropout=0.3):
        """
        初始化GNN模型
        
        参数:
            num_node_features: 节点特征维度
            hidden_dim: 隐藏层维度
            num_classes: 分类类别数
            dropout: Dropout比例
        """
        super(MolecularGNN, self).__init__()
        
        # 图卷积层
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # 批归一化
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2因为使用mean+max pooling
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        """前向传播"""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 图卷积层
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # 全局池化（同时使用mean和max）
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


def smiles_to_graph(smiles):
    """
    将SMILES字符串转换为图数据
    
    参数:
        smiles: SMILES字符串
    
    返回:
        PyTorch Geometric Data对象
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # 节点特征（原子特征）
    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),  # 原子序数
            atom.GetTotalDegree(),  # 度
            atom.GetFormalCharge(),  # 形式电荷
            atom.GetTotalNumHs(),  # 氢原子数
            atom.GetNumRadicalElectrons(),  # 自由基电子数
            int(atom.GetIsAromatic()),  # 是否芳香
            int(atom.IsInRing()),  # 是否在环中
            atom.GetExplicitValence(),  # 显式化合价
            atom.GetImplicitValence(),  # 隐式化合价
        ]
        atom_features.append(features)
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # 边索引（化学键）
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])  # 无向图，添加反向边
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # 创建Data对象
    data = Data(x=x, edge_index=edge_index)
    
    return data


class GNNClassifier:
    """
    GNN分类器包装类
    提供与sklearn兼容的接口
    """
    
    def __init__(self, hidden_dim=128, num_epochs=100, batch_size=32, 
                 learning_rate=0.001, device=None, random_state=42):
        """
        初始化GNN分类器
        
        参数:
            hidden_dim: 隐藏层维度
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            device: 计算设备
            random_state: 随机种子
        """
        self.hidden_dim = hidden_dim
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 设置随机种子
        torch.manual_seed(random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(random_state)
        
        self.model = None
        self.smiles_list = None
        
    def _prepare_graph_data(self, X, y=None):
        """将特征矩阵和SMILES转换为图数据"""
        graph_list = []
        
        for idx, smiles in enumerate(self.smiles_list):
            graph = smiles_to_graph(smiles)
            if graph is not None:
                if y is not None:
                    graph.y = torch.tensor([y[idx]], dtype=torch.long)
                graph_list.append(graph)
        
        return graph_list
    
    def fit(self, X, y, smiles_list):
        """
        训练模型
        
        参数:
            X: 特征矩阵（这里主要用于兼容，实际使用smiles_list）
            y: 标签
            smiles_list: SMILES字符串列表
        """
        self.smiles_list = smiles_list
        
        # 准备图数据
        print("  正在将分子转换为图数据...")
        graph_list = self._prepare_graph_data(X, y)
        
        # 创建数据加载器
        train_loader = DataLoader(graph_list, batch_size=self.batch_size, shuffle=True)
        
        # 初始化模型
        num_node_features = graph_list[0].num_node_features
        self.model = MolecularGNN(
            num_node_features=num_node_features,
            hidden_dim=self.hidden_dim,
            num_classes=2
        ).to(self.device)
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # 训练
        print(f"  开始训练GNN模型（{self.num_epochs}轮）...")
        self.model.train()
        
        for epoch in range(self.num_epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                
                optimizer.zero_grad()
                out = self.model(batch)
                loss = criterion(out, batch.y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = out.argmax(dim=1)
                correct += (pred == batch.y).sum().item()
                total += batch.y.size(0)
            
            if (epoch + 1) % 20 == 0:
                acc = correct / total
                print(f"  Epoch {epoch+1}/{self.num_epochs}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.4f}")
        
        return self
    
    def predict(self, X, smiles_list):
        """
        预测
        
        参数:
            X: 特征矩阵
            smiles_list: SMILES字符串列表
        
        返回:
            预测标签
        """
        self.smiles_list = smiles_list
        graph_list = self._prepare_graph_data(X)
        
        test_loader = DataLoader(graph_list, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                pred = out.argmax(dim=1)
                predictions.extend(pred.cpu().numpy())
        
        return np.array(predictions)
    
    def predict_proba(self, X, smiles_list):
        """
        预测概率
        
        参数:
            X: 特征矩阵
            smiles_list: SMILES字符串列表
        
        返回:
            预测概率（2列：非活性概率，活性概率）
        """
        self.smiles_list = smiles_list
        graph_list = self._prepare_graph_data(X)
        
        test_loader = DataLoader(graph_list, batch_size=self.batch_size, shuffle=False)
        
        self.model.eval()
        probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                out = self.model(batch)
                proba = F.softmax(out, dim=1)
                probabilities.extend(proba.cpu().numpy())
        
        return np.array(probabilities)


def test_gnn():
    """测试GNN模型"""
    print("测试GNN模型...")
    
    # 测试SMILES转图
    test_smiles = [
        'CCO',  # 乙醇
        'CC(=O)O',  # 乙酸
        'c1ccccc1'  # 苯
    ]
    
    for smiles in test_smiles:
        graph = smiles_to_graph(smiles)
        if graph is not None:
            print(f"SMILES: {smiles}")
            print(f"  节点数: {graph.num_nodes}")
            print(f"  边数: {graph.num_edges}")
            print(f"  节点特征维度: {graph.num_node_features}")
        else:
            print(f"SMILES: {smiles} - 转换失败")
    
    print("\nGNN模型测试完成！")


if __name__ == '__main__':
    test_gnn()

