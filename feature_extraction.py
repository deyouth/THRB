"""
特征提取模块
功能：从SMILES字符串生成分子指纹特征
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem import MACCSkeys
import warnings
warnings.filterwarnings('ignore')


class MolecularFeatureExtractor:
    """分子特征提取器"""
    
    def __init__(self, fingerprint_type='morgan', radius=2, n_bits=2048):
        """
        初始化特征提取器
        
        参数:
            fingerprint_type: 指纹类型，可选 'morgan', 'rdkit', 'maccs', 'combined'
            radius: Morgan指纹的半径（仅对morgan有效）
            n_bits: 指纹位数
        """
        self.fingerprint_type = fingerprint_type
        self.radius = radius
        self.n_bits = n_bits
        
    def smiles_to_mol(self, smiles):
        """将SMILES转换为分子对象"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol
        except:
            return None
    
    def get_morgan_fingerprint(self, mol):
        """
        获取Morgan指纹（ECFP）
        这是最常用的分子指纹之一，适合活性预测
        """
        if mol is None:
            return np.zeros(self.n_bits)
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(
                mol, self.radius, nBits=self.n_bits
            )
            return np.array(fp)
        except:
            return np.zeros(self.n_bits)
    
    def get_rdkit_fingerprint(self, mol):
        """获取RDKit拓扑指纹"""
        if mol is None:
            return np.zeros(self.n_bits)
        try:
            fp = Chem.RDKFingerprint(mol, fpSize=self.n_bits)
            return np.array(fp)
        except:
            return np.zeros(self.n_bits)
    
    def get_maccs_fingerprint(self, mol):
        """
        获取MACCS指纹
        固定166位的结构键指纹
        """
        if mol is None:
            return np.zeros(167)  # MACCS keys are 167 bits
        try:
            fp = MACCSkeys.GenMACCSKeys(mol)
            return np.array(fp)
        except:
            return np.zeros(167)
    
    def get_molecular_descriptors(self, mol):
        """
        计算常用的分子描述符
        """
        if mol is None:
            return np.zeros(10)
        
        try:
            descriptors = [
                Descriptors.MolWt(mol),                    # 分子量
                Descriptors.MolLogP(mol),                  # LogP
                Descriptors.NumHDonors(mol),               # 氢键供体数
                Descriptors.NumHAcceptors(mol),            # 氢键受体数
                Descriptors.TPSA(mol),                     # 拓扑极性表面积
                Descriptors.NumRotatableBonds(mol),        # 可旋转键数
                rdMolDescriptors.CalcNumAromaticRings(mol),# 芳香环数
                rdMolDescriptors.CalcNumRings(mol),        # 总环数
                Descriptors.NumHeteroatoms(mol),           # 杂原子数
                Descriptors.FractionCSP3(mol)              # sp3碳分数
            ]
            return np.array(descriptors)
        except:
            return np.zeros(10)
    
    def extract_features_from_smiles(self, smiles):
        """从单个SMILES提取特征"""
        mol = self.smiles_to_mol(smiles)
        
        if self.fingerprint_type == 'morgan':
            features = self.get_morgan_fingerprint(mol)
        elif self.fingerprint_type == 'rdkit':
            features = self.get_rdkit_fingerprint(mol)
        elif self.fingerprint_type == 'maccs':
            features = self.get_maccs_fingerprint(mol)
        elif self.fingerprint_type == 'combined':
            # 组合多种特征
            morgan_fp = self.get_morgan_fingerprint(mol)
            descriptors = self.get_molecular_descriptors(mol)
            features = np.concatenate([morgan_fp, descriptors])
        else:
            raise ValueError(f"未知的指纹类型: {self.fingerprint_type}")
        
        return features
    
    def extract_features_from_dataframe(self, df, smiles_column='canonical_smiles'):
        """
        从DataFrame批量提取特征
        
        参数:
            df: 包含SMILES的DataFrame
            smiles_column: SMILES列名
            
        返回:
            特征矩阵（numpy array）
        """
        print(f"正在提取特征（指纹类型: {self.fingerprint_type}）...")
        
        features_list = []
        valid_indices = []
        
        for idx, smiles in enumerate(df[smiles_column]):
            if idx % 500 == 0:
                print(f"进度: {idx}/{len(df)}")
            
            features = self.extract_features_from_smiles(smiles)
            
            # 检查特征是否有效
            if not np.all(features == 0):
                features_list.append(features)
                valid_indices.append(idx)
        
        features_array = np.array(features_list)
        print(f"特征提取完成！特征矩阵形状: {features_array.shape}")
        
        return features_array, valid_indices
    
    def save_features(self, features, labels, output_path='data/features.npz'):
        """保存特征和标签"""
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        np.savez(output_path, features=features, labels=labels)
        print(f"特征已保存到: {output_path}")
    
    def load_features(self, input_path='data/features.npz'):
        """加载特征和标签"""
        data = np.load(input_path)
        return data['features'], data['labels']


def main():
    """主函数：从处理后的数据提取特征"""
    
    # 加载处理后的数据
    print("正在加载处理后的数据...")
    df = pd.read_csv('data/thrb_processed.csv')
    print(f"数据集大小: {len(df)}")
    
    # 创建特征提取器（使用组合特征以获得最佳性能）
    extractor = MolecularFeatureExtractor(
        fingerprint_type='combined',  # 组合Morgan指纹和分子描述符
        radius=2,
        n_bits=2048
    )
    
    # 提取特征
    features, valid_indices = extractor.extract_features_from_dataframe(df)
    
    # 获取对应的标签
    labels = df.iloc[valid_indices]['activity'].values
    
    print(f"\n特征提取结果:")
    print(f"  特征矩阵形状: {features.shape}")
    print(f"  标签数量: {len(labels)}")
    print(f"  活性化合物: {np.sum(labels == 1)}")
    print(f"  非活性化合物: {np.sum(labels == 0)}")
    
    # 保存特征
    extractor.save_features(features, labels, 'data/features_combined.npz')
    
    # 同时保存Morgan指纹（作为备选）
    print("\n" + "="*60)
    print("额外生成Morgan指纹特征...")
    extractor_morgan = MolecularFeatureExtractor(
        fingerprint_type='morgan',
        radius=2,
        n_bits=2048
    )
    features_morgan, _ = extractor_morgan.extract_features_from_dataframe(df)
    extractor_morgan.save_features(features_morgan, labels, 'data/features_morgan.npz')
    
    print("\n特征提取全部完成！")


if __name__ == '__main__':
    main()

