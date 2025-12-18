"""
预测脚本
功能：使用训练好的模型对新化合物进行THRB活性预测
支持6种模型：Random Forest, XGBoost, Gradient Boosting, SVM, Logistic Regression, THRB GNN
"""

import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import os
import warnings
warnings.filterwarnings('ignore')

from feature_extraction import MolecularFeatureExtractor

# 尝试导入GNN模型
try:
    from model_gnn import GNNClassifier
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    print("提示: GNN模型依赖未安装，GNN预测功能不可用")


class THRBPredictor:
    """THRB活性预测器（支持6种模型）"""
    
    def __init__(self, model_name='best', model_path=None,
                 scaler_path='models/scaler.pkl',
                 fingerprint_type='combined'):
        """
        初始化预测器
        
        参数:
            model_name: 模型名称，可选值:
                - 'best': 最佳模型（默认）
                - 'random_forest': 随机森林
                - 'xgboost': XGBoost
                - 'gradient_boosting': 梯度提升
                - 'gradient_boosting': 支持向量机
                - 'logistic_regression': 逻辑回归
                - 'gnn' 或 'thrb_gnn': 图神经网络
            model_path: 自定义模型路径（可选，会覆盖model_name）
            scaler_path: 标准化器路径
            fingerprint_type: 特征提取类型
        """
        self.model_name = model_name
        self.fingerprint_type = fingerprint_type
        self.scaler_path = scaler_path
        
        # 模型映射
        self.model_files = {
            'best': 'models/best_model.pkl',
            'random_forest': 'models/model_random_forest.pkl',
            'xgboost': 'models/model_xgboost.pkl',
            'gradient_boosting': 'models/model_gradient_boosting.pkl',
            'svm': 'models/model_svm.pkl',
            'logistic_regression': 'models/model_logistic_regression.pkl',
            'gnn': 'models/model_thrb_gnn.pkl',
            'thrb_gnn': 'models/model_thrb_gnn.pkl'
        }
        
        # 确定模型路径
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = self.model_files.get(model_name.lower(), 'models/best_model.pkl')
        
        # 检查是否是GNN模型
        self.is_gnn = 'gnn' in model_name.lower() or 'gnn' in self.model_path.lower()
        
        # 加载模型和标准化器
        self.model = None
        self.scaler = None
        self.feature_extractor = None
        
        self._load_model()
        
    def _load_model(self):
        """加载模型和相关组件"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        # GNN模型检查
        if self.is_gnn:
            if not GNN_AVAILABLE:
                raise ImportError("GNN模型需要PyTorch和PyTorch Geometric，请先安装：\n"
                                "pip install torch torchvision torchaudio\n"
                                "pip install torch-geometric")
            print("正在加载GNN模型...")
        else:
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"标准化器文件不存在: {self.scaler_path}")
            print(f"正在加载模型: {self.model_name}...")
        
        self.model = joblib.load(self.model_path)
        
        # 传统ML模型需要scaler和特征提取器
        if not self.is_gnn:
            self.scaler = joblib.load(self.scaler_path)
            
            # 初始化特征提取器
            self.feature_extractor = MolecularFeatureExtractor(
                fingerprint_type=self.fingerprint_type,
                radius=2,
                n_bits=2048
            )
        
        print("模型加载完成！")
        
        # 显示模型信息
        if self.is_gnn:
            print("  模型类型: 图神经网络 (GNN)")
            print("  特点: 直接从分子图结构学习")
        else:
            print(f"  模型类型: {self.model_name}")
            print(f"  特征类型: {self.fingerprint_type}")
    
    def validate_smiles(self, smiles):
        """验证SMILES有效性"""
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    
    def predict_single(self, smiles, return_proba=True):
        """
        预测单个化合物的活性
        
        参数:
            smiles: SMILES字符串
            return_proba: 是否返回预测概率
            
        返回:
            预测结果字典
        """
        # 验证SMILES
        if not self.validate_smiles(smiles):
            return {
                'smiles': smiles,
                'valid': False,
                'error': 'Invalid SMILES',
                'model': self.model_name
            }
        
        # GNN模型使用不同的预测方式
        if self.is_gnn:
            # GNN需要SMILES，不需要特征提取
            X_dummy = np.zeros((1, 2058))  # Dummy特征矩阵
            smiles_list = [smiles]
            
            # 预测
            prediction = self.model.predict(X_dummy, smiles_list)[0]
            
            result = {
                'smiles': smiles,
                'valid': True,
                'prediction': int(prediction),
                'model': self.model_name
            }
            
            if return_proba:
                proba = self.model.predict_proba(X_dummy, smiles_list)[0]
                result['probability_active'] = float(proba[1])
                result['confidence'] = float(max(proba))
        else:
            # 传统ML模型：提取特征
            features = self.feature_extractor.extract_features_from_smiles(smiles)
            features = features.reshape(1, -1)
            
            # 标准化
            features = self.scaler.transform(features)
            
            # 预测
            prediction = self.model.predict(features)[0]
            
            result = {
                'smiles': smiles,
                'valid': True,
                'prediction': int(prediction),
                'model': self.model_name
            }
            
            if return_proba:
                proba = self.model.predict_proba(features)[0]
                result['probability_active'] = float(proba[1])
                result['confidence'] = float(max(proba))
        
        # 计算分子性质
        mol = Chem.MolFromSmiles(smiles)
        result['molecular_weight'] = Descriptors.MolWt(mol)
        result['logp'] = Descriptors.MolLogP(mol)
        result['h_bond_donors'] = Descriptors.NumHDonors(mol)
        result['h_bond_acceptors'] = Descriptors.NumHAcceptors(mol)
        
        return result
    
    def predict_batch(self, smiles_list, return_proba=True):
        """
        批量预测化合物活性
        
        参数:
            smiles_list: SMILES列表
            return_proba: 是否返回预测概率
            
        返回:
            预测结果列表
        """
        print(f"正在使用 {self.model_name} 模型预测 {len(smiles_list)} 个化合物...")
        
        # GNN模型可以批量预测（更高效）
        if self.is_gnn:
            results = []
            valid_smiles = []
            valid_indices = []
            
            # 验证SMILES
            for idx, smiles in enumerate(smiles_list):
                if self.validate_smiles(smiles):
                    valid_smiles.append(smiles)
                    valid_indices.append(idx)
                else:
                    results.append({
                        'smiles': smiles,
                        'valid': False,
                        'error': 'Invalid SMILES',
                        'model': self.model_name
                    })
            
            # 批量预测有效的SMILES
            if valid_smiles:
                print(f"  有效SMILES: {len(valid_smiles)}/{len(smiles_list)}")
                X_dummy = np.zeros((len(valid_smiles), 2058))
                predictions = self.model.predict(X_dummy, valid_smiles)
                
                if return_proba:
                    probabilities = self.model.predict_proba(X_dummy, valid_smiles)
                
                # 整理结果
                valid_results = []
                for i, smiles in enumerate(valid_smiles):
                    mol = Chem.MolFromSmiles(smiles)
                    result = {
                        'smiles': smiles,
                        'valid': True,
                        'prediction': int(predictions[i]),
                        'model': self.model_name,
                        'molecular_weight': Descriptors.MolWt(mol),
                        'logp': Descriptors.MolLogP(mol),
                        'h_bond_donors': Descriptors.NumHDonors(mol),
                        'h_bond_acceptors': Descriptors.NumHAcceptors(mol)
                    }
                    
                    if return_proba:
                        result['probability_active'] = float(probabilities[i][1])
                        result['confidence'] = float(max(probabilities[i]))
                    
                    valid_results.append(result)
                
                # 合并结果（保持原始顺序）
                final_results = []
                valid_idx = 0
                invalid_idx = 0
                for idx in range(len(smiles_list)):
                    if idx in valid_indices:
                        final_results.append(valid_results[valid_idx])
                        valid_idx += 1
                    else:
                        final_results.append(results[invalid_idx])
                        invalid_idx += 1
                
                results = final_results
        else:
            # 传统ML模型：逐个预测
            results = []
            for idx, smiles in enumerate(smiles_list):
                if (idx + 1) % 100 == 0:
                    print(f"  进度: {idx + 1}/{len(smiles_list)}")
                
                result = self.predict_single(smiles, return_proba=return_proba)
                results.append(result)
        
        print("预测完成！")
        return results
    
    def predict_from_file(self, input_file, smiles_column='smiles', 
                         output_file='predictions.csv'):
        """
        从文件读取SMILES并预测
        
        参数:
            input_file: 输入文件路径（CSV格式）
            smiles_column: SMILES列名
            output_file: 输出文件路径
        """
        # 读取输入文件
        print(f"正在读取文件: {input_file}")
        df = pd.read_csv(input_file)
        
        if smiles_column not in df.columns:
            raise ValueError(f"列 '{smiles_column}' 不存在于输入文件中")
        
        print(f"文件包含 {len(df)} 个化合物")
        
        # 批量预测
        smiles_list = df[smiles_column].tolist()
        results = self.predict_batch(smiles_list)
        
        # 转换为DataFrame
        results_df = pd.DataFrame(results)
        
        # 合并原始数据
        output_df = pd.concat([df, results_df.drop('smiles', axis=1)], axis=1)
        
        # 按活性概率排序（降序）
        if 'probability_active' in output_df.columns:
            output_df = output_df.sort_values('probability_active', ascending=False)
            print("结果已按活性概率从高到低排序")
        
        # 对数值列保留两位小数
        numeric_columns = ['probability_active', 'confidence', 
                          'molecular_weight', 'logp', 'h_bond_donors', 'h_bond_acceptors']
        for col in numeric_columns:
            if col in output_df.columns:
                output_df[col] = output_df[col].round(2)
        
        # 保存结果
        output_df.to_csv(output_file, index=False)
        print(f"预测结果已保存到: {output_file}")
        
        return output_df
    
    def visualize_molecule(self, smiles, output_path=None):
        """
        可视化分子结构
        
        参数:
            smiles: SMILES字符串
            output_path: 输出图片路径（可选）
        """
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            print(f"无效的SMILES: {smiles}")
            return None
        
        img = Draw.MolToImage(mol, size=(400, 400))
        
        if output_path:
            img.save(output_path)
            print(f"分子结构图已保存到: {output_path}")
        
        return img


def compare_models(smiles):
    """比较所有可用模型的预测结果"""
    print("\n" + "="*80)
    print("多模型预测对比")
    print("="*80)
    print(f"SMILES: {smiles}\n")
    
    # 所有可用的模型
    models_to_test = [
        ('best', '最佳模型'),
        ('random_forest', 'Random Forest'),
        ('xgboost', 'XGBoost'),
        ('gradient_boosting', 'Gradient Boosting'),
        ('svm', 'SVM'),
        ('logistic_regression', 'Logistic Regression'),
        ('gnn', 'THRB GNN')
    ]
    
    results = []
    
    for model_name, display_name in models_to_test:
        try:
            predictor = THRBPredictor(model_name=model_name, fingerprint_type='combined')
            result = predictor.predict_single(smiles)
            results.append({
                '模型': display_name,
                '活性概率': f"{result['probability_active']:.2f}",
                '置信度': f"{result['confidence']:.2f}"
            })
            print(f"✅ {display_name}: 活性概率={result['probability_active']:.2f}")
        except FileNotFoundError:
            print(f"⚠️  {display_name}: 模型文件不存在")
        except Exception as e:
            print(f"❌ {display_name}: {str(e)}")
    
    if results:
        print("\n模型预测汇总:")
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        return df
    return None


def main():
    """主函数：演示预测功能"""
    
    print("="*80)
    print("THRB 活性预测系统 - 支持6种模型")
    print("="*80)
    
    # 示例1：使用最佳模型预测
    print("\n示例 1: 使用最佳模型预测单个化合物")
    print("-" * 80)
    
    # 创建预测器（使用最佳模型）
    predictor = THRBPredictor(model_name='best', fingerprint_type='combined')
    
    # T3（三碘甲状腺原氨酸）- 已知THRB激动剂
    smiles_t3 = "N[C@@H](Cc1cc(I)c(Oc2cc(I)c(O)c(I)c2)c(I)c1)C(=O)O"
    
    result = predictor.predict_single(smiles_t3)
    
    print(f"\nSMILES: {result['smiles'][:50]}...")
    print(f"使用模型: {result['model']}")
    print(f"活性概率: {result['probability_active']:.2f}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"分子量: {result['molecular_weight']:.2f}")
    print(f"LogP: {result['logp']:.2f}")
    
    # 示例2：使用GNN模型
    if GNN_AVAILABLE and os.path.exists('models/model_thrb_gnn.pkl'):
        print("\n示例 2: 使用GNN模型预测")
        print("-" * 80)
        
        gnn_predictor = THRBPredictor(model_name='gnn')
        gnn_result = gnn_predictor.predict_single(smiles_t3)
        
        print(f"\n使用模型: GNN (图神经网络)")
        print(f"活性概率: {gnn_result['probability_active']:.2f}")
        print(f"置信度: {gnn_result['confidence']:.2f}")
    
    # 示例3：批量预测
    print("\n示例 3: 批量预测多个化合物")
    print("-" * 80)
    
    # 多个测试化合物
    test_compounds = [
        "CCOc1ccc(C2NC(=O)NC2=O)cc1",  # 测试化合物1
        "Cc1ccc(O)cc1",  # 对甲酚
        "c1ccc(cc1)c2ccccc2",  # 联苯
        "CC(=O)Oc1ccccc1C(=O)O",  # 阿司匹林
    ]
    
    results = predictor.predict_batch(test_compounds)
    
    # 显示结果
    results_df = pd.DataFrame(results)
    print("\n预测结果:")
    display_cols = ['smiles', 'model', 'probability_active', 'confidence']
    print(results_df[display_cols].to_string(index=False))
    
    # 保存结果
    results_df.to_csv('example_predictions.csv', index=False)
    print("\n✅ 结果已保存到: example_predictions.csv")
    
    # 示例4：多模型对比
    print("\n示例 4: 多模型预测对比")
    print("-" * 80)
    test_smiles = "Cc1ccc(O)cc1"
    compare_models(test_smiles)
    
    # 使用说明
    print("\n" + "="*80)
    print("💡 使用说明")
    print("="*80)
    print("\n1. 选择特定模型预测:")
    print("   predictor = THRBPredictor(model_name='xgboost')  # 或 'gnn', 'random_forest' 等")
    print("   result = predictor.predict_single('YOUR_SMILES')")
    
    print("\n2. 批量预测:")
    print("   results = predictor.predict_batch(['SMILES1', 'SMILES2', ...])")
    
    print("\n3. 从文件预测:")
    print("   predictor.predict_from_file('input.csv', output_file='output.csv')")
    
    print("\n4. 可用的模型:")
    print("   - 'best': 最佳模型（自动选择）")
    print("   - 'random_forest': 随机森林")
    print("   - 'xgboost': XGBoost")
    print("   - 'gradient_boosting': 梯度提升")
    print("   - 'svm': 支持向量机")
    print("   - 'logistic_regression': 逻辑回归")
    print("   - 'gnn': 图神经网络 ⭐")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='THRB活性预测工具（支持6种模型）')
    parser.add_argument('--model', type=str, default='best',
                       choices=['best', 'random_forest', 'xgboost', 'gradient_boosting', 
                               'svm', 'logistic_regression', 'gnn', 'thrb_gnn'],
                       help='选择预测模型（默认: best）')
    parser.add_argument('--smiles', type=str, help='单个SMILES字符串')
    parser.add_argument('--input', type=str, help='输入CSV文件路径')
    parser.add_argument('--output', type=str, default='predictions.csv', 
                       help='输出CSV文件路径（默认: predictions.csv）')
    parser.add_argument('--smiles-column', type=str, default='smiles',
                       help='CSV文件中的SMILES列名（默认: smiles）')
    parser.add_argument('--compare', action='store_true',
                       help='比较所有模型的预测结果')
    
    args = parser.parse_args()
    
    # 检查模型目录是否存在
    if not os.path.exists('models'):
        print("❌ 错误：models目录不存在！")
        print("\n请先运行以下命令进行模型训练：")
        print("  python data_preprocessing.py")
        print("  python feature_extraction.py")
        print("  python model_training.py")
        exit(1)
    
    # 命令行模式
    if args.smiles or args.input or args.compare:
        try:
            predictor = THRBPredictor(model_name=args.model, fingerprint_type='combined')
            
            if args.compare and args.smiles:
                # 比较模式
                compare_models(args.smiles)
            elif args.smiles:
                # 单个SMILES预测
                result = predictor.predict_single(args.smiles)
                print(f"\n预测结果:")
                print(f"  SMILES: {result['smiles']}")
                print(f"  模型: {result['model']}")
                print(f"  活性概率: {result['probability_active']:.2f}")
                print(f"  置信度: {result['confidence']:.2f}")
            elif args.input:
                # 文件批量预测
                predictor.predict_from_file(
                    input_file=args.input,
                    smiles_column=args.smiles_column,
                    output_file=args.output
                )
            
        except FileNotFoundError as e:
            print(f"❌ 错误: {e}")
            print("\n提示: 确保已运行完整的模型训练流程")
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    else:
        # 演示模式
        if not os.path.exists('models/best_model.pkl'):
            print("❌ 错误：未找到训练好的模型！")
            print("\n请先运行以下命令进行模型训练：")
            print("  python data_preprocessing.py")
            print("  python feature_extraction.py")
            print("  python model_training.py")
        else:
            main()

