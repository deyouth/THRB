"""
模型评估和可视化脚本
功能：生成详细的评估图表和可视化（包括GNN）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, precision_recall_curve, 
                             confusion_matrix, classification_report)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# 导入GNN模型
try:
    from model_gnn import GNNClassifier
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False

# 设置字体为Arial
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 20  # 增大默认字号


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, models_dir='models', output_dir='results'):
        """
        初始化评估器
        
        参数:
            models_dir: 模型文件目录
            output_dir: 输出目录
        """
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.smiles_test = None
        os.makedirs(output_dir, exist_ok=True)
        
    def load_test_data(self, features_path='data/features_combined.npz', test_size=0.2):
        """加载测试数据"""
        from sklearn.model_selection import train_test_split
        
        data = np.load(features_path)
        features = data['features']
        labels = data['labels']
        
        # 使用相同的随机种子划分数据
        _, X_test, _, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # 加载scaler并标准化
        scaler = joblib.load(f'{self.models_dir}/scaler.pkl')
        X_test = scaler.transform(X_test)
        
        return X_test, y_test
    
    def load_all_models(self):
        """加载所有训练好的模型"""
        models = {}
        
        model_files = [
            ('Random Forest', 'model_random_forest.pkl'),
            ('XGBoost', 'model_xgboost.pkl'),
            ('Gradient Boosting', 'model_gradient_boosting.pkl'),
            ('SVM', 'model_svm.pkl'),
            ('Logistic Regression', 'model_logistic_regression.pkl'),
            ('THRB GNN', 'model_thrb_gnn.pkl')
        ]
        
        for name, filename in model_files:
            filepath = f'{self.models_dir}/{filename}'
            if os.path.exists(filepath):
                models[name] = joblib.load(filepath)
                print(f"已加载: {name}")
        
        # 加载SMILES测试集（用于GNN）
        smiles_path = f'{self.models_dir}/smiles_test.npy'
        if os.path.exists(smiles_path):
            self.smiles_test = np.load(smiles_path, allow_pickle=True)
            print(f"已加载测试集SMILES: {len(self.smiles_test)} 个")
        else:
            self.smiles_test = None
        
        return models
    
    def plot_roc_curves(self, X_test, y_test, models):
        """绘制ROC曲线"""
        plt.figure(figsize=(12, 8))
        
        for name, model in models.items():
            # GNN模型需要SMILES
            if name == 'THRB GNN' and self.smiles_test is not None:
                y_proba = model.predict_proba(X_test, self.smiles_test)[:, 1]
            else:
                y_proba = model.predict_proba(X_test)[:, 1]
                
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random (AUC = 0.500)')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=24, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=24, fontweight='bold')
        plt.title('ROC Curves', fontsize=28, fontweight='bold')
        plt.legend(loc='lower right', fontsize=18)
        plt.grid(alpha=0.3, linewidth=1.2)
        plt.tick_params(axis='both', labelsize=24, width=1.2, length=6)
        
        output_path = f'{self.output_dir}/roc_curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC曲线已保存到: {output_path}")
        plt.close()
    
    def plot_precision_recall_curves(self, X_test, y_test, models):
        """绘制Precision-Recall曲线"""
        plt.figure(figsize=(12, 8))
        
        for name, model in models.items():
            # GNN模型需要SMILES
            if name == 'THRB GNN' and self.smiles_test is not None:
                y_proba = model.predict_proba(X_test, self.smiles_test)[:, 1]
            else:
                y_proba = model.predict_proba(X_test)[:, 1]
                
            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            
            plt.plot(recall, precision, lw=2, label=f'{name}')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=24, fontweight='bold')
        plt.ylabel('Precision', fontsize=24, fontweight='bold')
        plt.title('Precision-Recall Curves', fontsize=28, fontweight='bold')
        plt.legend(loc='best', fontsize=18)
        plt.grid(alpha=0.3)
        plt.tick_params(axis='both', labelsize=24)
        
        output_path = f'{self.output_dir}/precision_recall_curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall曲线已保存到: {output_path}")
        plt.close()
    
    def plot_confusion_matrices(self, X_test, y_test, models):
        """绘制混淆矩阵"""
        n_models = len(models)
        fig, axes = plt.subplots(2, 3, figsize=(20, 13))
        axes = axes.flatten()
        
        for idx, (name, model) in enumerate(models.items()):
            # GNN模型需要SMILES
            if name == 'THRB GNN' and self.smiles_test is not None:
                y_pred = model.predict(X_test, self.smiles_test)
            else:
                y_pred = model.predict(X_test)
                
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 20, 'fontweight': 'bold'})
            axes[idx].set_title(f'{name}', fontsize=22, fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontsize=20, fontweight='bold')
            axes[idx].set_ylabel('True', fontsize=20, fontweight='bold')
            axes[idx].set_xticklabels(['Inactive (0)', 'Active (1)'], fontsize=20)
            axes[idx].set_yticklabels(['Inactive (0)', 'Active (1)'], fontsize=20)
        
        # 隐藏多余的子图
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Confusion Matrices - THRB Binary Classification', 
                    fontsize=28, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        output_path = f'{self.output_dir}/confusion_matrices.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存到: {output_path}")
        plt.close()
    
    def plot_model_comparison(self):
        """绘制模型性能对比图"""
        # 读取模型比较结果
        comparison_df = pd.read_csv(f'{self.models_dir}/model_comparison.csv')
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = [
            ('Test_Accuracy', 'Accuracy'),
            ('Test_Precision', 'Precision'),
            ('Test_F1', 'F1 Score'),
            ('Test_ROC_AUC', 'ROC AUC')
        ]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            
            # 按性能排序
            df_sorted = comparison_df.sort_values(metric, ascending=True)
            
            # 绘制水平条形图
            bars = ax.barh(df_sorted['Model'], df_sorted[metric])
            
            # 根据性能着色（学术论文常用蓝色色调）
            colors = plt.cm.Blues(df_sorted[metric] * 0.7 + 0.3)
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # 添加数值标签
            for i, v in enumerate(df_sorted[metric]):
                    ax.text(v - 0.02, i, f'{v:.3f}', va='center', ha='right', 
                           fontsize=16, color='white', fontweight='bold')
            ax.text(v - 0.02, i, f'{v:.3f}', va='center', ha='right', fontsize=16, color='white', fontweight='bold')
            ax.set_xlabel('Score', fontsize=20, fontweight='bold')
            ax.set_title(title, fontsize=22, fontweight='bold')
            ax.set_xlim([0, 1.05])
            ax.grid(axis='x', alpha=0.3)
            ax.tick_params(axis='both', labelsize=16)
        
        plt.suptitle('Model Performance Comparison - THRB Binary Classification', 
                    fontsize=28, fontweight='bold')
        plt.tight_layout()
        
        output_path = f'{self.output_dir}/model_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"模型对比图已保存到: {output_path}")
        plt.close()
    
    def plot_feature_importance(self):
        """绘制特征重要性（针对Random Forest和XGBoost）"""
        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        
        models_to_plot = [
            ('Random Forest', 'model_random_forest.pkl'),
            ('XGBoost', 'model_xgboost.pkl')
        ]
        
        for idx, (name, filename) in enumerate(models_to_plot):
            filepath = f'{self.models_dir}/{filename}'
            if os.path.exists(filepath):
                model = joblib.load(filepath)
                
                # 获取特征重要性
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    # 获取前20个最重要的特征
                    indices = np.argsort(importances)[-20:]
                    
                    axes[idx].barh(range(len(indices)), importances[indices])
                    axes[idx].set_yticks(range(len(indices)))
                    axes[idx].set_yticklabels([f'Feature {i}' for i in indices], fontsize=16)
                    axes[idx].set_xlabel('Feature Importance', fontsize=20, fontweight='bold')
                    axes[idx].set_title(f'{name} - Top 20 Features', fontsize=22, fontweight='bold')
                    axes[idx].grid(axis='x', alpha=0.3)
                    axes[idx].tick_params(axis='x', labelsize=16)
        
        plt.tight_layout()
        output_path = f'{self.output_dir}/feature_importance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"特征重要性图已保存到: {output_path}")
        plt.close()
    
    def plot_prediction_distribution(self, X_test, y_test, models):
        """绘制预测概率分布"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (name, model) in enumerate(models.items()):
            # GNN模型需要SMILES
            if name == 'THRB GNN' and self.smiles_test is not None:
                y_proba = model.predict_proba(X_test, self.smiles_test)[:, 1]
            else:
                y_proba = model.predict_proba(X_test)[:, 1]
            
            # 分别绘制活性和非活性样本的预测概率分布（使用鲜艳的红蓝色调）
            axes[idx].hist(y_proba[y_test == 0], bins=50, alpha=0.8, 
                          label='Inactive (True)', color="#0679DF", density=True)
            axes[idx].hist(y_proba[y_test == 1], bins=50, alpha=0.8, 
                          label='Active (True)', color="#E60F0C", density=True)
            
            axes[idx].axvline(x=0.5, color='black', linestyle='--', linewidth=2, 
                            label='Threshold (0.5)')
            axes[idx].set_xlabel('Predicted Probability', fontsize=20, fontweight='bold')
            axes[idx].set_ylabel('Density', fontsize=20, fontweight='bold')
            axes[idx].set_title(f'{name}', fontsize=22, fontweight='bold')
            axes[idx].legend(fontsize=14)
            axes[idx].grid(alpha=0.3)
            axes[idx].tick_params(axis='both', labelsize=16)
        
        # 隐藏多余的子图
        for idx in range(len(models), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('Prediction Probability Distributions - THRB Binary Classification', 
                    fontsize=28, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        output_path = f'{self.output_dir}/prediction_distributions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"预测概率分布图已保存到: {output_path}")
        plt.close()
    
    def generate_classification_reports(self, X_test, y_test, models):
        """生成分类报告"""
        report_path = f'{self.output_dir}/classification_reports.txt'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("THRB 二分类模型 - 详细分类报告\n")
            f.write("="*80 + "\n\n")
            
            for name, model in models.items():
                # GNN模型需要SMILES
                if name == 'THRB GNN' and self.smiles_test is not None:
                    y_pred = model.predict(X_test, self.smiles_test)
                else:
                    y_pred = model.predict(X_test)
                
                f.write("="*80 + "\n")
                f.write(f"{name}\n")
                f.write("="*80 + "\n\n")
                
                report = classification_report(y_test, y_pred, 
                                               target_names=['Inactive', 'Active'])
                f.write(report)
                f.write("\n\n")
        
        print(f"分类报告已保存到: {report_path}")
    
    def run_full_evaluation(self):
        """执行完整的评估流程"""
        print("="*80)
        print("开始模型评估和可视化")
        print("="*80)
        
        # 加载测试数据
        print("\n正在加载测试数据...")
        X_test, y_test = self.load_test_data()
        print(f"测试集大小: {len(X_test)}")
        print(f"活性样本: {np.sum(y_test == 1)}, 非活性样本: {np.sum(y_test == 0)}")
        
        # 加载模型
        print("\n正在加载模型...")
        models = self.load_all_models()
        print(f"已加载 {len(models)} 个模型")
        
        # 生成各种可视化
        print("\n正在生成可视化图表...")
        print("-" * 60)
        
        self.plot_roc_curves(X_test, y_test, models)
        self.plot_precision_recall_curves(X_test, y_test, models)
        self.plot_confusion_matrices(X_test, y_test, models)
        self.plot_model_comparison()
        # self.plot_feature_importance()  # 已禁用
        self.plot_prediction_distribution(X_test, y_test, models)
        
        # 生成分类报告
        print("\n正在生成分类报告...")
        self.generate_classification_reports(X_test, y_test, models)
        
        print("\n" + "="*80)
        print("评估完成！所有结果已保存到 results 目录")
        print("="*80)


def main():
    """主函数"""
    evaluator = ModelEvaluator(
        models_dir='models',
        output_dir='results'
    )
    
    evaluator.run_full_evaluation()


if __name__ == '__main__':
    main()

