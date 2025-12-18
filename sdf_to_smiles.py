"""
SDF文件转换为SMILES格式脚本
功能：从SDF文件提取SMILES和化合物名称
"""

import pandas as pd
from rdkit import Chem
import sys


def sdf_to_smiles_csv(sdf_file, output_csv='compounds_smiles.csv'):
    """
    将SDF文件转换为SMILES CSV文件
    
    参数:
        sdf_file: SDF文件路径
        output_csv: 输出CSV文件路径
    """
    print(f"正在读取SDF文件: {sdf_file}")
    
    # 读取SDF文件
    suppl = Chem.SDMolSupplier(sdf_file)
    
    smiles_list = []
    name_list = []
    
    total = 0
    success = 0
    
    for mol in suppl:
        total += 1
        
        if mol is None:
            print(f"警告: 第 {total} 个分子无法解析，跳过")
            continue
        
        # 生成SMILES
        smiles = Chem.MolToSmiles(mol)
        
        # 获取化合物名称（尝试多个可能的属性名）
        name = None
        possible_name_fields = ['_Name', 'Name', 'ID', 'COMPOUND_NAME', 'Title', 'Catalog Number']
        
        for field in possible_name_fields:
            if mol.HasProp(field):
                name = mol.GetProp(field)
                break
        
        # 如果没有名称，使用索引
        if name is None or name.strip() == '':
            name = f"Compound_{total}"
        
        smiles_list.append(smiles)
        name_list.append(name)
        success += 1
        
        if total % 500 == 0:
            print(f"已处理: {total} 个分子，成功: {success} 个")
    
    print(f"\n转换完成！")
    print(f"总分子数: {total}")
    print(f"成功转换: {success}")
    print(f"失败: {total - success}")
    
    # 创建DataFrame
    df = pd.DataFrame({
        'smiles': smiles_list,
        'compound_name': name_list
    })
    
    # 保存为CSV
    df.to_csv(output_csv, index=False)
    print(f"\n结果已保存到: {output_csv}")
    print(f"前5行预览:")
    print(df.head())
    
    return df


def main():
    """主函数"""
    # 默认输入输出文件
    sdf_file = 'Bioactive_Compound_Library-6200.sdf'
    output_csv = 'bioactive_compounds_smiles.csv'
    
    # 如果命令行提供了参数，使用命令行参数
    if len(sys.argv) > 1:
        sdf_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_csv = sys.argv[2]
    
    # 执行转换
    sdf_to_smiles_csv(sdf_file, output_csv)


if __name__ == '__main__':
    main()
