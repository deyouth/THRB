"""
GNN模型快速测试脚本
验证GNN依赖是否正确安装
"""

import sys

def test_imports():
    """测试依赖导入"""
    print("="*60)
    print("测试GNN依赖...")
    print("="*60)
    
    # 测试PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} 安装成功")
        print(f"   CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
        if torch.cuda.is_available():
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   GPU设备: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        print(f"❌ PyTorch未安装: {e}")
        return False
    
    # 测试PyTorch Geometric
    try:
        import torch_geometric
        print(f"✅ PyTorch Geometric {torch_geometric.__version__} 安装成功")
    except ImportError as e:
        print(f"❌ PyTorch Geometric未安装: {e}")
        print("\n安装方法:")
        print("pip install torch-geometric")
        return False
    
    # 测试RDKit
    try:
        from rdkit import Chem
        print(f"✅ RDKit 安装成功")
    except ImportError as e:
        print(f"❌ RDKit未安装: {e}")
        return False
    
    print("\n所有依赖安装正确！")
    return True


def test_gnn_model():
    """测试GNN模型"""
    print("\n" + "="*60)
    print("测试GNN模型...")
    print("="*60)
    
    try:
        from model_gnn import MolecularGNN, smiles_to_graph, GNNClassifier
        print("✅ GNN模型导入成功")
    except ImportError as e:
        print(f"❌ GNN模型导入失败: {e}")
        return False
    
    # 测试SMILES转图
    print("\n测试SMILES到图的转换...")
    test_smiles = [
        ('CCO', '乙醇'),
        ('CC(=O)O', '乙酸'),
        ('c1ccccc1', '苯'),
        ('CC(C)Cc1ccc(cc1)C(C)C(=O)O', '布洛芬')
    ]
    
    for smiles, name in test_smiles:
        graph = smiles_to_graph(smiles)
        if graph is not None:
            print(f"✅ {name} ({smiles})")
            print(f"   节点数: {graph.num_nodes}, 边数: {graph.num_edges}, 特征维度: {graph.num_node_features}")
        else:
            print(f"❌ {name} ({smiles}) - 转换失败")
    
    # 测试GNN分类器
    print("\n测试GNN分类器...")
    import numpy as np
    
    try:
        # 创建一个小型测试数据集
        train_smiles = ['CCO', 'CCCO', 'CCCCO', 'c1ccccc1', 'c1ccc(O)cc1']
        train_labels = np.array([0, 0, 0, 1, 1])
        X_train = np.zeros((len(train_smiles), 10))  # Dummy特征
        
        # 初始化并训练
        clf = GNNClassifier(
            hidden_dim=64,
            num_epochs=5,
            batch_size=2,
            learning_rate=0.01
        )
        
        print("正在训练小型测试模型（5个epoch）...")
        clf.fit(X_train, train_labels, train_smiles)
        
        # 测试预测
        test_smiles = ['CC(C)O', 'c1ccc(C)cc1']
        X_test = np.zeros((len(test_smiles), 10))
        
        predictions = clf.predict(X_test, test_smiles)
        probabilities = clf.predict_proba(X_test, test_smiles)
        
        print("\n✅ GNN分类器测试成功！")
        print("\n预测示例:")
        for i, smiles in enumerate(test_smiles):
            print(f"  SMILES: {smiles}")
            print(f"    预测: {'Active' if predictions[i] == 1 else 'Inactive'}")
            print(f"    概率: Inactive={probabilities[i][0]:.3f}, Active={probabilities[i][1]:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ GNN分类器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("\n" + "="*60)
    print("THRB GNN 模型环境测试")
    print("="*60 + "\n")
    
    # 测试依赖
    if not test_imports():
        print("\n❌ 依赖测试失败，请先安装所需依赖")
        print("\n安装命令:")
        print("pip install torch torch-geometric")
        sys.exit(1)
    
    # 测试GNN模型
    if not test_gnn_model():
        print("\n❌ GNN模型测试失败")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("✅ 所有测试通过！GNN模型可以正常使用")
    print("="*60)
    print("\n下一步:")
    print("1. 运行完整训练: python model_training.py")
    print("2. 查看评估结果: python model_evaluation.py")
    print("3. 阅读详细指南: GNN_GUIDE.md")


if __name__ == '__main__':
    main()

