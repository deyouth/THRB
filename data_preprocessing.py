"""
THRB 数据预处理脚本
功能：从原始数据集中筛选THRB相关数据，并进行清洗和标注
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings
warnings.filterwarnings('ignore')


class THRBDataPreprocessor:
    """THRB数据预处理器"""
    
    def __init__(self, data_path, activity_threshold=6.0):
        """
        初始化预处理器
        
        参数:
            data_path: 原始数据文件路径
            activity_threshold: 活性阈值，pchembl_value >= threshold 为活性化合物
        """
        self.data_path = data_path
        self.activity_threshold = activity_threshold
        self.df = None
        self.df_thrb = None
        
    def load_data(self):
        """加载原始数据"""
        print("正在加载数据...")
        self.df = pd.read_csv(self.data_path)
        print(f"原始数据集大小: {len(self.df)} 条记录")
        print(f"数据集包含的核受体类型: {self.df['gene_symbol'].nunique()} 种")
        return self
    
    def filter_thrb_data(self):
        """筛选THRB相关数据"""
        print("\n正在筛选THRB相关数据...")
        # 筛选THRB数据（NR1A2对应THRB）
        self.df_thrb = self.df[self.df['gene_symbol'].str.contains('THRB|NR1A2', case=False, na=False)].copy()
        print(f"THRB相关数据: {len(self.df_thrb)} 条记录")
        return self
    
    def clean_data(self):
        """数据清洗"""
        print("\n正在清洗数据...")
        initial_count = len(self.df_thrb)
        
        # 删除缺失SMILES的数据
        self.df_thrb = self.df_thrb.dropna(subset=['canonical_smiles', 'pchembl_value'])
        print(f"删除缺失值后: {len(self.df_thrb)} 条记录")
        
        # 验证SMILES有效性
        valid_smiles = []
        for idx, row in self.df_thrb.iterrows():
            mol = Chem.MolFromSmiles(row['canonical_smiles'])
            if mol is not None:
                valid_smiles.append(idx)
        
        self.df_thrb = self.df_thrb.loc[valid_smiles]
        print(f"验证SMILES有效性后: {len(self.df_thrb)} 条记录")
        
        # 去重（基于SMILES）
        self.df_thrb = self.df_thrb.drop_duplicates(subset=['canonical_smiles'])
        print(f"去重后: {len(self.df_thrb)} 条记录")
        
        return self
    
    def create_binary_labels(self):
        """
        创建二分类标签
        活性化合物（Active）: pchembl_value >= threshold
        非活性化合物（Inactive）: pchembl_value < threshold
        """
        print(f"\n正在创建二分类标签（阈值 = {self.activity_threshold}）...")
        self.df_thrb['activity'] = (self.df_thrb['pchembl_value'] >= self.activity_threshold).astype(int)
        
        active_count = (self.df_thrb['activity'] == 1).sum()
        inactive_count = (self.df_thrb['activity'] == 0).sum()
        
        print(f"活性化合物数量: {active_count} ({active_count/len(self.df_thrb)*100:.2f}%)")
        print(f"非活性化合物数量: {inactive_count} ({inactive_count/len(self.df_thrb)*100:.2f}%)")
        
        return self
    
    def add_molecular_properties(self):
        """添加基本分子性质"""
        print("\n正在计算分子性质...")
        
        mol_weights = []
        logps = []
        hbd = []  # 氢键供体
        hba = []  # 氢键受体
        
        for smiles in self.df_thrb['canonical_smiles']:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol_weights.append(Descriptors.MolWt(mol))
                logps.append(Descriptors.MolLogP(mol))
                hbd.append(Descriptors.NumHDonors(mol))
                hba.append(Descriptors.NumHAcceptors(mol))
            else:
                mol_weights.append(np.nan)
                logps.append(np.nan)
                hbd.append(np.nan)
                hba.append(np.nan)
        
        self.df_thrb['mol_weight'] = mol_weights
        self.df_thrb['logp'] = logps
        self.df_thrb['h_bond_donors'] = hbd
        self.df_thrb['h_bond_acceptors'] = hba
        
        print("分子性质计算完成！")
        return self
    
    def save_processed_data(self, output_path='data/thrb_processed.csv'):
        """保存处理后的数据"""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存完整数据
        self.df_thrb.to_csv(output_path, index=False)
        print(f"\n处理后的数据已保存到: {output_path}")
        
        # 保存统计信息
        stats_path = 'data/data_statistics.txt'
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("THRB 数据集统计信息\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"总样本数: {len(self.df_thrb)}\n")
            f.write(f"活性化合物: {(self.df_thrb['activity'] == 1).sum()}\n")
            f.write(f"非活性化合物: {(self.df_thrb['activity'] == 0).sum()}\n")
            f.write(f"活性阈值: pchembl_value >= {self.activity_threshold}\n\n")
            
            f.write("分子性质统计:\n")
            f.write("-" * 40 + "\n")
            f.write(self.df_thrb[['mol_weight', 'logp', 'h_bond_donors', 'h_bond_acceptors', 'pchembl_value']].describe().to_string())
            f.write("\n\n")
            
            f.write("数据类型分布:\n")
            f.write("-" * 40 + "\n")
            f.write(self.df_thrb['standard_type'].value_counts().to_string())
        
        print(f"统计信息已保存到: {stats_path}")
        return self
    
    def run(self, output_path='data/thrb_processed.csv'):
        """执行完整的数据预处理流程"""
        self.load_data()
        self.filter_thrb_data()
        self.clean_data()
        self.create_binary_labels()
        self.add_molecular_properties()
        self.save_processed_data(output_path)
        
        print("\n" + "=" * 60)
        print("数据预处理完成！")
        print("=" * 60)
        
        return self.df_thrb


def main():
    """主函数"""
    # 创建预处理器实例
    preprocessor = THRBDataPreprocessor(
        data_path='nr_activities.csv',
        activity_threshold=6.0  # pchembl_value >= 6.0 为活性化合物
    )
    
    # 执行预处理
    df_processed = preprocessor.run()
    
    print("\n预处理后的数据前5行:")
    print(df_processed[['canonical_smiles', 'pchembl_value', 'activity', 'mol_weight', 'logp']].head())


if __name__ == '__main__':
    main()

