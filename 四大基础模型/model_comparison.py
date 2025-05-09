# model_comparison.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
import warnings
import pickle
import os
import tensorflow as tf
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import StratifiedKFold
from improved_network_detection import create_enhanced_features, preprocess_data
from gbdt_model import find_optimal_threshold

warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


def load_models():
    """加载所有训练好的模型"""
    models = {}
    
    # 加载随机森林模型
    try:
        with open('unsw_nb15_model.pkl', 'rb') as f:
            rf_data = pickle.load(f)
            models['RandomForest'] = {
                'model': rf_data['model'],
                'threshold': rf_data['threshold'],
                'accuracy': rf_data['accuracy'],
                'auc': rf_data['auc'],
                'selected_features': rf_data['selected_features']
            }
    except Exception as e:
        print(f"加载RandomForest模型时出错: {e}")
    
    # 加载GBDT模型
    try:
        with open('gbdt/unsw_nb15_gbdt_model.pkl', 'rb') as f:
            gbdt_data = pickle.load(f)
            models['GBDT'] = {
                'model': gbdt_data['model'],
                'threshold': gbdt_data['threshold'],
                'accuracy': gbdt_data['accuracy'],
                'auc': gbdt_data['auc'],
                'selected_features': gbdt_data['selected_features']
            }
    except Exception as e:
        print(f"加载GBDT模型时出错: {e}")
    
    # 加载XGBoost模型
    try:
        with open('XGBOOST/unsw_nb15_xgboost_model.pkl', 'rb') as f:
            xgb_data = pickle.load(f)
            models['XGBoost'] = {
                'model': xgb_data['model'],
                'threshold': xgb_data['threshold'],
                'accuracy': xgb_data['accuracy'],
                'auc': xgb_data['auc'],
                'selected_features': xgb_data['selected_features']
            }
    except Exception as e:
        print(f"加载XGBoost模型时出错: {e}")
    
    # 加载LightGBM模型
    try:
        with open('LightGBM/unsw_nb15_lightgbm_model.pkl', 'rb') as f:
            lgb_data = pickle.load(f)
            models['LightGBM'] = {
                'model': lgb_data['model'],
                'threshold': lgb_data['threshold'],
                'accuracy': lgb_data['accuracy'],
                'auc': lgb_data['auc'],
                'selected_features': lgb_data['selected_features']
            }
    except Exception as e:
        print(f"加载LightGBM模型时出错: {e}")
    
    # 加载LSTM模型
    try:
        # 加载Keras模型
        lstm_model = tf.keras.models.load_model('lstm_model.h5')
        
        # 加载模型信息
        with open('LSTM/lstm_model_info.pkl', 'rb') as f:
            lstm_info = pickle.load(f)
        
        # 加载scaler
        with open('LSTM/lstm_scaler.pkl', 'rb') as f:
            lstm_scaler = pickle.load(f)
        
        models['LSTM'] = {
            'model': lstm_model,
            'threshold': lstm_info['threshold'],
            'accuracy': lstm_info['accuracy'],
            'auc': lstm_info['auc'],
            'feature_names': lstm_info['feature_names'],
            'scaler': lstm_scaler,
            'is_lstm': True
        }
    except Exception as e:
        print(f"加载LSTM模型时出错: {e}")
    
    return models


def compare_performance(models):
    """比较不同模型的性能"""
    # 提取模型性能指标
    results = {}
    for name, model_data in models.items():
        results[name] = {
            'Accuracy': model_data['accuracy'],
            'AUC': model_data['auc']
        }
    
    # 创建性能比较DataFrame
    performance_df = pd.DataFrame(results).T
    print("模型性能比较:")
    print(performance_df)
    
    # 可视化性能比较
    plt.figure(figsize=(12, 6))
    
    # 准确率对比
    plt.subplot(1, 2, 1)
    bars = plt.bar(performance_df.index, performance_df['Accuracy'], color='skyblue')
    plt.title('各模型准确率比较')
    plt.ylabel('准确率')
    plt.ylim(0.8, 1.0)  # 调整y轴范围更好地显示差异
    plt.grid(axis='y', alpha=0.3)
    
    # 在条形上添加准确率值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    # AUC对比
    plt.subplot(1, 2, 2)
    bars = plt.bar(performance_df.index, performance_df['AUC'], color='lightgreen')
    plt.title('各模型AUC比较')
    plt.ylabel('AUC')
    plt.ylim(0.8, 1.0)  # 调整y轴范围更好地显示差异
    plt.grid(axis='y', alpha=0.3)
    
    # 在条形上添加AUC值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png')
    
    return performance_df


def test_ensemble_voting(X_test, y_test, models):
    """测试集成投票分类器的性能"""
    print("\n测试集成投票分类器...")
    
    # 筛选非LSTM模型进行集成
    clf_models = {}
    for name, model_data in models.items():
        if name != 'LSTM' and 'model' in model_data:  # 确保模型已成功加载
            clf_models[name] = model_data['model']
    
    if len(clf_models) < 2:
        print("没有足够的模型进行集成")
        return None
    
    # 创建投票分类器
    voting_clf = VotingClassifier(
        estimators=[(name, model) for name, model in clf_models.items()],
        voting='soft'  # 使用概率
    )
    
    # 训练投票分类器（实际上不需要训练，但需要拟合）
    # 为了简化，我们直接用模型在测试集上的预测
    predictions = {}
    for name, model in clf_models.items():
        # 获取概率预测
        try:
            if hasattr(model, 'predict_proba'):
                y_probs = model.predict_proba(X_test)[:, 1]
            else:
                y_probs = model.decision_function(X_test)
            predictions[name] = y_probs
        except Exception as e:
            print(f"预测时出错 ({name}): {e}")
    
    # 如果没有成功的预测，则返回None
    if not predictions:
        print("所有模型预测失败，无法进行集成")
        return None
    
    # 计算加权平均概率（简单集成）
    ensemble_probs = np.zeros(len(y_test))
    total_weight = 0
    
    for name, probs in predictions.items():
        # 可以根据各模型的AUC加权
        if name in models and 'auc' in models[name]:
            weight = models[name]['auc']
            ensemble_probs += weight * probs
            total_weight += weight
    
    if total_weight == 0:
        print("无法计算加权平均，模型AUC值缺失")
        return None
        
    ensemble_probs /= total_weight
    
    # 寻找最佳阈值
    try:
        best_threshold = find_optimal_threshold(y_test, ensemble_probs)
        print(f"集成模型最佳阈值: {best_threshold:.4f}")
    except Exception as e:
        print(f"寻找最佳阈值时出错: {e}")
        best_threshold = 0.5
    
    # 使用最佳阈值进行预测
    ensemble_preds = (ensemble_probs >= best_threshold).astype(int)
    
    # 计算评估指标
    accuracy = (ensemble_preds == y_test).mean()
    auc = roc_auc_score(y_test, ensemble_probs)
    
    print(f"集成模型准确率: {accuracy:.4f}")
    print(f"集成模型AUC: {auc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, ensemble_preds))
    
    # 打印混淆矩阵
    cm = confusion_matrix(y_test, ensemble_preds)
    print("\n混淆矩阵:")
    print(cm)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['正常', '攻击'],
                yticklabels=['正常', '攻击'])
    plt.xlabel('预测')
    plt.ylabel('实际')
    plt.title('集成模型混淆矩阵')
    plt.savefig('ensemble_confusion_matrix.png')
    
    # 可视化ROC曲线
    plt.figure(figsize=(10, 8))
    
    # 绘制所有模型的ROC曲线
    for name, model_data in models.items():
        if name in predictions:
            fpr, tpr, _ = roc_curve(y_test, predictions[name])
            plt.plot(fpr, tpr, label=f'{name} (AUC = {model_data["auc"]:.4f})')
    
    # 绘制集成模型的ROC曲线
    fpr, tpr, _ = roc_curve(y_test, ensemble_probs)
    plt.plot(fpr, tpr, 'r-', linewidth=2, label=f'集成模型 (AUC = {auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线比较')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('ensemble_roc_curves.png')
    
    # 创建威胁热图 - 可视化不同模型对相同样本的预测差异
    # 限制样本数以提高绘图效率
    n_samples = min(100, len(y_test))
    sample_indices = np.random.choice(len(y_test), n_samples, replace=False)
    
    # 提取各模型对样本的预测概率
    pred_matrix = np.zeros((n_samples, len(predictions) + 1))
    model_names = list(predictions.keys()) + ['集成模型']
    
    for i, name in enumerate(predictions.keys()):
        pred_matrix[:, i] = predictions[name][sample_indices]
    
    # 添加集成模型的预测
    pred_matrix[:, -1] = ensemble_probs[sample_indices]
    
    # 获取这些样本的真实标签
    true_labels = y_test[sample_indices]
    
    # 创建热图
    plt.figure(figsize=(12, 10))
    
    # 左侧创建真实标签热图
    plt.subplot(1, 10, 1)
    plt.imshow(true_labels.reshape(-1, 1), cmap='RdYlGn_r', aspect='auto')
    plt.title('真实标签')
    plt.xticks([])
    plt.yticks([])
    
    # 右侧展示预测概率热图
    plt.subplot(1, 10, (2, 10))
    sns.heatmap(pred_matrix, cmap='YlOrRd',
               xticklabels=model_names,
               yticklabels=False)
    plt.title('各模型预测概率比较')
    plt.tight_layout()
    plt.savefig('model_prediction_comparison.png')
    
    return {'accuracy': accuracy, 'auc': auc, 'threshold': best_threshold}


def main():
    try:
        # 1. 加载数据
        print("加载UNSW-NB15数据集...")
        try:
            train_data = pd.read_csv("./data/UNSW_NB15_training-set.csv")
            test_data = pd.read_csv("./data/UNSW_NB15_testing-set.csv")
        except FileNotFoundError:
            # 尝试备用路径
            print("默认路径未找到数据集，尝试备用路径...")
            train_data = pd.read_csv("UNSW_NB15_training-set.csv")
            test_data = pd.read_csv("UNSW_NB15_testing-set.csv")

        print(f"训练集形状: {train_data.shape}")
        print(f"测试集形状: {test_data.shape}")

        # 2. 特征工程
        print("\n创建增强特征...")
        train_enhanced = create_enhanced_features(train_data)
        test_enhanced = create_enhanced_features(test_data)

        # 3. 数据预处理和特征选择
        print("\n预处理数据和选择特征...")
        X_train, y_train, X_test, y_test, _ = preprocess_data(
            train_enhanced, test_enhanced
        )

        # 4. 加载所有模型
        print("\n加载训练好的模型...")
        models = load_models()
        
        if not models:
            print("没有成功加载任何模型，程序终止")
            return
        
        # 5. 比较模型性能
        print("\n比较模型性能...")
        performance_df = compare_performance(models)
        
        # 6. 测试集成模型
        ensemble_results = test_ensemble_voting(X_test, y_test, models)
        
        # 7. 将集成模型结果添加到性能比较
        if ensemble_results:
            performance_df.loc['Ensemble'] = {
                'Accuracy': ensemble_results['accuracy'],
                'AUC': ensemble_results['auc']
            }
            
            # 更新性能比较图表
            plt.figure(figsize=(12, 6))
            
            # 准确率对比
            plt.subplot(1, 2, 1)
            bars = plt.bar(performance_df.index, performance_df['Accuracy'], color='skyblue')
            plt.title('各模型准确率比较(含集成)')
            plt.ylabel('准确率')
            plt.ylim(0.8, 1.0)
            plt.grid(axis='y', alpha=0.3)
            
            # 在条形上添加准确率值
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom')
            
            # AUC对比
            plt.subplot(1, 2, 2)
            bars = plt.bar(performance_df.index, performance_df['AUC'], color='lightgreen')
            plt.title('各模型AUC比较(含集成)')
            plt.ylabel('AUC')
            plt.ylim(0.8, 1.0)
            plt.grid(axis='y', alpha=0.3)
            
            # 在条形上添加AUC值
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('final_model_performance_comparison.png')
            
            print("\n最终性能比较结果:")
            print(performance_df)
        
        print("\n模型评估完成！")
    
    except Exception as e:
        import traceback
        print(f"程序执行过程中发生错误: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # 设置TensorFlow日志级别
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main() 