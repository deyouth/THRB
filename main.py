"""
THRB 二分类模型主程序
提供命令行界面来运行完整的建模流程
"""

import argparse
import os
import sys


def run_preprocessing():
    """运行数据预处理"""
    print("\n" + "="*80)
    print("步骤 1: 数据预处理")
    print("="*80)
    
    from data_preprocessing import main as preprocess_main
    preprocess_main()


def run_feature_extraction():
    """运行特征提取"""
    print("\n" + "="*80)
    print("步骤 2: 特征提取")
    print("="*80)
    
    from feature_extraction import main as feature_main
    feature_main()


def run_model_training():
    """运行模型训练"""
    print("\n" + "="*80)
    print("步骤 3: 模型训练")
    print("="*80)
    
    from model_training import main as train_main
    train_main()


def run_model_evaluation():
    """运行模型评估"""
    print("\n" + "="*80)
    print("步骤 4: 模型评估和可视化")
    print("="*80)
    
    from model_evaluation import main as eval_main
    eval_main()


def run_prediction(input_file=None, smiles=None):
    """运行预测"""
    print("\n" + "="*80)
    print("化合物活性预测")
    print("="*80)
    
    from predict import THRBPredictor
    
    predictor = THRBPredictor(
        model_path='models/best_model.pkl',
        scaler_path='models/scaler.pkl',
        fingerprint_type='combined'
    )
    
    if smiles:
        # 预测单个化合物
        result = predictor.predict_single(smiles)
        
        print(f"\n预测结果:")
        print(f"  SMILES: {result['smiles']}")
        print(f"  有效性: {'有效' if result['valid'] else '无效'}")
        
        if result['valid']:
            print(f"  预测类别: {result['activity']}")
            print(f"  活性概率: {result['probability_active']:.4f}")
            print(f"  置信度: {result['confidence']:.4f}")
            print(f"  分子量: {result['molecular_weight']:.2f}")
            print(f"  LogP: {result['logp']:.2f}")
        else:
            print(f"  错误: {result.get('error', 'Unknown error')}")
    
    elif input_file:
        # 从文件批量预测
        if not os.path.exists(input_file):
            print(f"错误：文件不存在 - {input_file}")
            return
        
        output_file = input_file.replace('.csv', '_predictions.csv')
        predictor.predict_from_file(input_file, output_file=output_file)
    
    else:
        print("错误：请提供 --smiles 或 --input_file 参数")


def run_full_pipeline():
    """运行完整的建模流程"""
    print("="*80)
    print("THRB 二分类模型 - 完整建模流程")
    print("="*80)
    
    try:
        # 1. 数据预处理
        run_preprocessing()
        
        # 2. 特征提取
        run_feature_extraction()
        
        # 3. 模型训练
        run_model_training()
        
        # 4. 模型评估
        run_model_evaluation()
        
        print("\n" + "="*80)
        print("完整流程执行成功！")
        print("="*80)
        print("\n生成的文件：")
        print("  - data/thrb_processed.csv: 处理后的数据")
        print("  - data/features_combined.npz: 提取的特征")
        print("  - models/best_model.pkl: 最佳模型")
        print("  - results/: 评估结果和可视化图表")
        print("\n使用以下命令进行预测：")
        print("  python main.py predict --smiles 'YOUR_SMILES'")
        print("  python main.py predict --input_file your_compounds.csv")
        
    except Exception as e:
        print(f"\n错误：{str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='THRB 二分类模型 - 化合物活性预测',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 运行完整流程（数据预处理、特征提取、模型训练、评估）
  python main.py full
  
  # 运行单个步骤
  python main.py preprocess    # 数据预处理
  python main.py extract       # 特征提取
  python main.py train         # 模型训练
  python main.py evaluate      # 模型评估
  
  # 预测新化合物
  python main.py predict --smiles "CCOc1ccc(C2NC(=O)NC2=O)cc1"
  python main.py predict --input_file compounds.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 完整流程
    subparsers.add_parser('full', help='运行完整的建模流程')
    
    # 单个步骤
    subparsers.add_parser('preprocess', help='运行数据预处理')
    subparsers.add_parser('extract', help='运行特征提取')
    subparsers.add_parser('train', help='运行模型训练')
    subparsers.add_parser('evaluate', help='运行模型评估')
    
    # 预测
    predict_parser = subparsers.add_parser('predict', help='预测化合物活性')
    predict_parser.add_argument('--smiles', type=str, help='单个化合物的SMILES字符串')
    predict_parser.add_argument('--input_file', type=str, help='包含SMILES的CSV文件')
    
    args = parser.parse_args()
    
    # 检查数据文件
    if args.command != 'predict' and not os.path.exists('nr_activities.csv'):
        print("错误：未找到数据文件 'nr_activities.csv'")
        print("请确保 nr_activities.csv 文件在当前目录下")
        sys.exit(1)
    
    # 执行命令
    if args.command == 'full':
        run_full_pipeline()
    elif args.command == 'preprocess':
        run_preprocessing()
    elif args.command == 'extract':
        run_feature_extraction()
    elif args.command == 'train':
        run_model_training()
    elif args.command == 'evaluate':
        run_model_evaluation()
    elif args.command == 'predict':
        run_prediction(input_file=args.input_file, smiles=args.smiles)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

