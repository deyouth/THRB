"""
模型训练脚本
功能：训练多种机器学习模型并进行比较（包括GNN）
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# 导入GNN模型
try:
    from model_gnn import GNNClassifier
    GNN_AVAILABLE = True
except ImportError:
    print("警告: GNN模型依赖未安装，将跳过GNN训练")
    print("请安装: pip install torch torch-geometric")
    GNN_AVAILABLE = False


class THRBModelTrainer:
    """THRB模型训练器"""
    
    def __init__(self, features_path='data/features_combined.npz', 
                 data_csv_path='data/thrb_processed.csv', 
                 test_size=0.2, random_state=42, use_gnn=True):
        """
        初始化训练器
        
        参数:
            features_path: 特征文件路径
            data_csv_path: 原始数据CSV路径（包含SMILES）
            test_size: 测试集比例
            random_state: 随机种子
            use_gnn: 是否使用GNN模型
        """
        self.features_path = features_path
        self.data_csv_path = data_csv_path
        self.test_size = test_size
        self.random_state = random_state
        self.use_gnn = use_gnn and GNN_AVAILABLE
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.smiles_train = None
        self.smiles_test = None
        self.scaler = None
        
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """加载特征数据和SMILES"""
        print("正在加载特征数据...")
        data = np.load(self.features_path)
        features = data['features']
        labels = data['labels']
        
        print(f"特征矩阵形状: {features.shape}")
        print(f"标签数量: {len(labels)}")
        print(f"活性样本: {np.sum(labels == 1)} ({np.sum(labels == 1)/len(labels)*100:.2f}%)")
        print(f"非活性样本: {np.sum(labels == 0)} ({np.sum(labels == 0)/len(labels)*100:.2f}%)")
        
        # 加载SMILES（用于GNN）
        smiles_list = None
        if self.use_gnn:
            print("正在加载SMILES数据（用于GNN）...")
            df = pd.read_csv(self.data_csv_path)
            smiles_list = df['canonical_smiles'].values
            print(f"SMILES数量: {len(smiles_list)}")
            
            if len(smiles_list) != len(labels):
                print(f"警告: SMILES数量({len(smiles_list)})与标签数量({len(labels)})不匹配！")
                smiles_list = None
                self.use_gnn = False
        
        return features, labels, smiles_list
    
    def split_data(self, features, labels, smiles_list=None, use_smote=True):
        """
        划分训练集和测试集，并可选择使用SMOTE进行过采样
        
        参数:
            features: 特征矩阵
            labels: 标签
            smiles_list: SMILES列表（用于GNN）
            use_smote: 是否使用SMOTE处理类别不平衡
        """
        print(f"\n正在划分数据集（测试集比例: {self.test_size}）...")
        
        # 划分数据集
        if smiles_list is not None:
            self.X_train, self.X_test, self.y_train, self.y_test, self.smiles_train, self.smiles_test = train_test_split(
                features, labels, smiles_list,
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=labels
            )
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                features, labels, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=labels
            )
        
        print(f"训练集大小: {len(self.X_train)}")
        print(f"测试集大小: {len(self.X_test)}")
        
        # 标准化特征
        print("正在标准化特征...")
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        # 使用SMOTE处理类别不平衡（注意：SMOTE后不改变smiles对应关系）
        if use_smote:
            print("\n使用SMOTE处理类别不平衡...")
            print(f"SMOTE前 - 训练集活性样本: {np.sum(self.y_train == 1)}, "
                  f"非活性样本: {np.sum(self.y_train == 0)}")
            
            smote = SMOTE(random_state=self.random_state)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            
            print(f"SMOTE后 - 训练集活性样本: {np.sum(self.y_train == 1)}, "
                  f"非活性样本: {np.sum(self.y_train == 0)}")
            print("注意: GNN模型将使用SMOTE前的原始数据训练")
        
        return self
    
    def initialize_models(self):
        """初始化多个分类模型"""
        print("\n正在初始化模型...")
        
        self.models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            
            'XGBoost': XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.random_state,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.random_state
            ),
            
            'SVM': SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=self.random_state
            ),
            
            'Logistic Regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        
        # 添加GNN模型
        if self.use_gnn and self.smiles_train is not None:
            print("正在初始化GNN模型...")
            self.models['THRB GNN'] = GNNClassifier(
                hidden_dim=128,
                num_epochs=100,
                batch_size=32,
                learning_rate=0.001,
                random_state=self.random_state
            )
        
        print(f"已初始化 {len(self.models)} 个模型")
        return self
    
    def train_and_evaluate(self):
        """训练并评估所有模型"""
        print("\n" + "="*80)
        print("开始训练模型...")
        print("="*80)
        
        # 保存原始训练数据（GNN需要）
        X_train_original = self.scaler.inverse_transform(self.X_train)[:len(self.smiles_train)] if self.smiles_train is not None else None
        y_train_original = self.y_train[:len(self.smiles_train)] if self.smiles_train is not None else None
        
        for name, model in self.models.items():
            print(f"\n{'='*80}")
            print(f"训练模型: {name}")
            print(f"{'='*80}")
            
            # 特殊处理GNN模型
            if name == 'THRB GNN' and self.smiles_train is not None:
                # GNN使用原始数据（未经SMOTE）和SMILES
                print(f"  GNN使用原始训练数据（未经SMOTE）")
                model.fit(X_train_original, y_train_original, self.smiles_train)
                
                # 预测
                y_train_pred = model.predict(X_train_original, self.smiles_train)
                y_test_pred = model.predict(self.X_test, self.smiles_test)
                
                # 预测概率
                y_train_proba = model.predict_proba(X_train_original, self.smiles_train)[:, 1]
                y_test_proba = model.predict_proba(self.X_test, self.smiles_test)[:, 1]
                
                # 计算评估指标
                train_metrics = self._calculate_metrics(y_train_original, y_train_pred, y_train_proba)
                test_metrics = self._calculate_metrics(self.y_test, y_test_pred, y_test_proba)
                
                # GNN跳过交叉验证（太慢）
                cv_scores = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
                print("  注意: GNN模型跳过交叉验证以节省时间")
                
            else:
                # 传统机器学习模型
                model.fit(self.X_train, self.y_train)
                
                # 在训练集和测试集上进行预测
                y_train_pred = model.predict(self.X_train)
                y_test_pred = model.predict(self.X_test)
                
                # 预测概率
                y_train_proba = model.predict_proba(self.X_train)[:, 1]
                y_test_proba = model.predict_proba(self.X_test)[:, 1]
                
                # 计算评估指标
                train_metrics = self._calculate_metrics(self.y_train, y_train_pred, y_train_proba)
                test_metrics = self._calculate_metrics(self.y_test, y_test_pred, y_test_proba)
                
                # 交叉验证
                cv_scores = self._cross_validation(model, name)
            
            # 保存结果
            self.results[name] = {
                'model': model,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_scores': cv_scores,
                'y_test_pred': y_test_pred,
                'y_test_proba': y_test_proba
            }
            
            # 打印结果
            self._print_results(name, train_metrics, test_metrics, cv_scores)
        
        return self
    
    def _calculate_metrics(self, y_true, y_pred, y_proba):
        """计算评估指标"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        return metrics
    
    def _cross_validation(self, model, name):
        """执行交叉验证"""
        print(f"  执行5折交叉验证...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        # 对于某些模型，交叉验证可能很慢，因此限制数据量
        if name in ['SVM'] and len(self.X_train) > 5000:
            print(f"  注意: {name}模型交叉验证使用前5000个样本以加快速度")
            X_cv = self.X_train[:5000]
            y_cv = self.y_train[:5000]
        else:
            X_cv = self.X_train
            y_cv = self.y_train
        
        cv_scores = cross_val_score(model, X_cv, y_cv, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        return cv_scores
    
    def _print_results(self, name, train_metrics, test_metrics, cv_scores):
        """打印评估结果"""
        print(f"\n{name} - 评估结果:")
        print("-" * 60)
        print(f"训练集:")
        print(f"  准确率: {train_metrics['accuracy']:.4f}")
        print(f"  精确率: {train_metrics['precision']:.4f}")
        print(f"  召回率: {train_metrics['recall']:.4f}")
        print(f"  F1分数: {train_metrics['f1']:.4f}")
        print(f"  ROC AUC: {train_metrics['roc_auc']:.4f}")
        
        print(f"\n测试集:")
        print(f"  准确率: {test_metrics['accuracy']:.4f}")
        print(f"  精确率: {test_metrics['precision']:.4f}")
        print(f"  召回率: {test_metrics['recall']:.4f}")
        print(f"  F1分数: {test_metrics['f1']:.4f}")
        print(f"  ROC AUC: {test_metrics['roc_auc']:.4f}")
        
        print(f"\n交叉验证 ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        print(f"\n混淆矩阵:")
        print(test_metrics['confusion_matrix'])
    
    def get_best_model(self):
        """根据测试集ROC AUC选择最佳模型"""
        best_name = None
        best_auc = 0
        
        for name, result in self.results.items():
            test_auc = result['test_metrics']['roc_auc']
            if test_auc > best_auc:
                best_auc = test_auc
                best_name = name
        
        return best_name, self.results[best_name]['model']
    
    def save_results(self, output_dir='models'):
        """保存所有模型和结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存最佳模型
        best_name, best_model = self.get_best_model()
        joblib.dump(best_model, f'{output_dir}/best_model.pkl')
        joblib.dump(self.scaler, f'{output_dir}/scaler.pkl')
        
        print(f"\n最佳模型 ({best_name}) 已保存到: {output_dir}/best_model.pkl")
        
        # 保存所有模型
        for name, result in self.results.items():
            safe_name = name.replace(' ', '_').lower()
            joblib.dump(result['model'], f'{output_dir}/model_{safe_name}.pkl')
        
        # 保存SMILES测试集（用于GNN评估）
        if self.smiles_test is not None:
            np.save(f'{output_dir}/smiles_test.npy', self.smiles_test)
            print(f"测试集SMILES已保存")
        
        # 保存评估结果
        results_df = self._create_results_dataframe()
        results_df.to_csv(f'{output_dir}/model_comparison.csv', index=False)
        
        # 保存详细报告
        self._save_detailed_report(output_dir)
        
        print(f"所有结果已保存到: {output_dir}")
        
        return best_name, best_model
    
    def _create_results_dataframe(self):
        """创建结果对比表"""
        data = []
        for name, result in self.results.items():
            test_m = result['test_metrics']
            cv_scores = result['cv_scores']
            
            data.append({
                'Model': name,
                'Test_Accuracy': test_m['accuracy'],
                'Test_Precision': test_m['precision'],
                'Test_Recall': test_m['recall'],
                'Test_F1': test_m['f1'],
                'Test_ROC_AUC': test_m['roc_auc'],
                'CV_ROC_AUC_Mean': cv_scores.mean(),
                'CV_ROC_AUC_Std': cv_scores.std()
            })
        
        return pd.DataFrame(data).sort_values('Test_ROC_AUC', ascending=False)
    
    def _save_detailed_report(self, output_dir):
        """保存详细评估报告"""
        report_path = f'{output_dir}/evaluation_report.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("THRB 二分类模型评估报告\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"数据集信息:\n")
            f.write(f"  训练集大小: {len(self.X_train)}\n")
            f.write(f"  测试集大小: {len(self.X_test)}\n")
            f.write(f"  特征维度: {self.X_train.shape[1]}\n\n")
            
            # 最佳模型
            best_name, _ = self.get_best_model()
            f.write(f"最佳模型: {best_name}\n")
            f.write(f"测试集 ROC AUC: {self.results[best_name]['test_metrics']['roc_auc']:.4f}\n\n")
            
            # 各模型详细结果
            for name, result in self.results.items():
                f.write("="*80 + "\n")
                f.write(f"{name}\n")
                f.write("="*80 + "\n\n")
                
                f.write("测试集性能:\n")
                test_m = result['test_metrics']
                f.write(f"  准确率: {test_m['accuracy']:.4f}\n")
                f.write(f"  精确率: {test_m['precision']:.4f}\n")
                f.write(f"  召回率: {test_m['recall']:.4f}\n")
                f.write(f"  F1分数: {test_m['f1']:.4f}\n")
                f.write(f"  ROC AUC: {test_m['roc_auc']:.4f}\n\n")
                
                f.write("混淆矩阵:\n")
                f.write(str(test_m['confusion_matrix']) + "\n\n")
                
                f.write("交叉验证结果:\n")
                cv_scores = result['cv_scores']
                f.write(f"  ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n")
                f.write(f"  各折分数: {[f'{s:.4f}' for s in cv_scores]}\n\n")
        
        print(f"详细报告已保存到: {report_path}")


def main():
    """主函数"""
    print("="*80)
    print("THRB 二分类模型训练（包括GNN）")
    print("="*80)
    
    # 创建训练器
    trainer = THRBModelTrainer(
        features_path='data/features_combined.npz',
        data_csv_path='data/thrb_processed.csv',
        test_size=0.2,
        random_state=42,
        use_gnn=True  # 启用GNN
    )
    
    # 加载数据
    features, labels, smiles_list = trainer.load_data()
    
    # 划分数据集（使用SMOTE处理类别不平衡）
    trainer.split_data(features, labels, smiles_list, use_smote=True)
    
    # 初始化模型
    trainer.initialize_models()
    
    # 训练和评估
    trainer.train_and_evaluate()
    
    # 保存结果
    best_name, best_model = trainer.save_results()
    
    print("\n" + "="*80)
    print(f"训练完成！最佳模型: {best_name}")
    print("="*80)


if __name__ == '__main__':
    main()

