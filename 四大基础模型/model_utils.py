# model_utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score,
                             roc_auc_score, precision_recall_curve, confusion_matrix,
                             classification_report)
import seaborn as sns


def find_optimal_threshold(y_true, y_scores):
    """寻找最佳阈值，平衡精确率和召回率"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    # 计算F1分数
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    # 计算精确率与召回率的几何平均，对极端值更敏感
    geo_mean = np.sqrt(precision * recall)

    # 找到F1和几何平均最大值对应的索引
    f1_idx = np.argmax(f1_scores)
    geo_idx = np.argmax(geo_mean)

    # 考虑两者，偏向于平衡模型
    best_idx = geo_idx  # 使用几何平均作为最终标准
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    return best_threshold, precision, recall, f1_scores, geo_mean, thresholds


def visualize_threshold_optimization(precision, recall, f1_scores, geo_mean, thresholds, best_threshold, model_name):
    """可视化阈值优化过程"""
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision[:-1], 'b--', label='精确率')
    plt.plot(thresholds, recall[:-1], 'g-', label='召回率')
    plt.plot(thresholds, f1_scores[:-1], 'r-', label='F1分数')
    plt.plot(thresholds, geo_mean[:-1], 'c-', label='几何平均')
    plt.axvline(x=best_threshold, color='purple', linestyle='--',
                label=f'最佳阈值 ({best_threshold:.3f})')
    plt.xlabel('阈值')
    plt.ylabel('分数')
    plt.title(f'{model_name}阈值优化')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_threshold_optimization.png')


def visualize_confusion_matrix(y_test, y_pred, model_name):
    """可视化混淆矩阵"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['正常', '攻击'],
                yticklabels=['正常', '攻击'])
    plt.xlabel('预测')
    plt.ylabel('实际')
    plt.title(f'{model_name}混淆矩阵')
    plt.savefig(f'{model_name}_confusion_matrix.png')
    return cm


def visualize_probability_distribution(y_test, y_probs, best_threshold, model_name):
    """可视化预测概率分布"""
    plt.figure(figsize=(10, 6))
    plt.hist(y_probs[y_test == 0], bins=50, alpha=0.5, label='正常流量')
    plt.hist(y_probs[y_test == 1], bins=50, alpha=0.5, label='攻击流量')
    plt.axvline(x=best_threshold, color='red', linestyle='--',
                label=f'阈值 ({best_threshold:.3f})')
    plt.xlabel('预测概率')
    plt.ylabel('样本数量')
    plt.title(f'{model_name}预测概率分布')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{model_name}_probability_distribution.png')


def visualize_roc_curve(y_test, y_probs, auc, model_name):
    """可视化ROC曲线"""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC曲线 (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title(f'{model_name} ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f'{model_name}_roc_curve.png')


def add_performance_table(y_test, y_pred, y_probs, model_name):
    """创建并保存详细的性能指标表格"""
    # 计算各项指标
    metrics = {
        '准确率 (Accuracy)': accuracy_score(y_test, y_pred),
        '精确率 (Precision)': precision_score(y_test, y_pred),
        '召回率 (Recall)': recall_score(y_test, y_pred),
        'F1分数': f1_score(y_test, y_pred),
        'AUC分数': roc_auc_score(y_test, y_probs),
        '正样本数量': sum(y_test == 1),
        '负样本数量': sum(y_test == 0),
        '预测正例': sum(y_pred == 1),
        '预测负例': sum(y_pred == 0)
    }

    # 转换为DataFrame并保存
    df_metrics = pd.DataFrame(list(metrics.items()), columns=['指标', '值'])
    df_metrics['值'] = df_metrics['值'].round(4)

    # 保存为CSV
    df_metrics.to_csv(f'{model_name}_performance_metrics.csv', index=False)

    # 可视化为条形图
    plt.figure(figsize=(10, 6))
    plt.bar(df_metrics['指标'][:5], df_metrics['值'][:5], color='teal')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'{model_name}模型性能指标')
    plt.tight_layout()
    plt.savefig(f'{model_name}_metrics_bar.png')

    # 返回各个指标值
    return (metrics['准确率 (Accuracy)'], metrics['AUC分数'],
            metrics['精确率 (Precision)'], metrics['召回率 (Recall)'],
            metrics['F1分数'])


def analyze_predictions(y_test, y_pred, y_probs, model_name):
    """详细分析预测结果并生成可视化"""
    # 1. 错误分析
    false_pos = np.where((y_pred == 1) & (y_test == 0))[0]
    false_neg = np.where((y_pred == 0) & (y_test == 1))[0]

    error_analysis = {
        '正确预测': sum(y_pred == y_test),
        '错误预测': sum(y_pred != y_test),
        '假阳性(FP)': len(false_pos),
        '假阴性(FN)': len(false_neg),
        '假阳性比例': len(false_pos) / max(sum(y_test == 0), 1),
        '假阴性比例': len(false_neg) / max(sum(y_test == 1), 1),
    }

    # 创建DataFrame
    df_errors = pd.DataFrame(list(error_analysis.items()),
                             columns=['错误类型', '数量/比例'])

    # 保存为CSV
    df_errors.to_csv(f'{model_name}_error_analysis.csv', index=False)

    # 2. 预测置信度分布
    # 预测概率分布柱状图
    plt.figure(figsize=(12, 6))

    # 为正确和错误预测创建不同的直方图
    correct_probs = y_probs[y_pred == y_test]
    wrong_probs = y_probs[y_pred != y_test]

    bins = np.linspace(0, 1, 21)  # 20个区间，从0到1

    plt.hist(correct_probs, bins=bins, alpha=0.5, label='正确预测', color='green')
    plt.hist(wrong_probs, bins=bins, alpha=0.5, label='错误预测', color='red')

    plt.xlabel('预测概率')
    plt.ylabel('样本数量')
    plt.title(f'{model_name}预测置信度分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{model_name}_prediction_confidence.png')

    # 3. 概率区间的准确率
    interval_accuracy = []
    intervals = np.linspace(0, 1, 11)  # 0.0, 0.1, 0.2, ..., 1.0

    for i in range(len(intervals) - 1):
        lower = intervals[i]
        upper = intervals[i + 1]
        mask = (y_probs >= lower) & (y_probs < upper)

        if sum(mask) > 0:
            acc = accuracy_score(y_test[mask], y_pred[mask])
            interval_accuracy.append((f"{lower:.1f}-{upper:.1f}", acc, sum(mask)))

    df_intervals = pd.DataFrame(interval_accuracy,
                                columns=['概率区间', '准确率', '样本数'])

    # 保存为CSV
    df_intervals.to_csv(f'{model_name}_interval_accuracy.csv', index=False)

    # 可视化概率区间准确率
    plt.figure(figsize=(12, 6))
    plt.bar(df_intervals['概率区间'], df_intervals['准确率'], color='purple')

    # 添加样本数量标签
    for i, (_, row) in enumerate(df_intervals.iterrows()):
        plt.text(i, row['准确率'] + 0.02, f"n={row['样本数']}",
                 ha='center', va='bottom', fontsize=9)

    plt.ylim(0, 1.1)
    plt.title(f'{model_name}不同概率区间的准确率')
    plt.ylabel('准确率')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{model_name}_interval_accuracy.png')

    return df_errors, df_intervals


def compare_models(models_results, save_path='model_comparison.csv'):
    """比较多个模型的性能指标

    Args:
        models_results: 字典，键为模型名称，值为(accuracy, auc, precision, recall, f1)元组
    """
    # 创建比较DataFrame
    df_comparison = pd.DataFrame(columns=['模型', '准确率', 'AUC', '精确率', '召回率', 'F1分数'])

    for i, (model_name, metrics) in enumerate(models_results.items()):
        df_comparison.loc[i] = [model_name] + list(metrics)

    # 保存为CSV
    df_comparison.to_csv(save_path, index=False)

    # 创建比较条形图
    metrics = ['准确率', 'AUC', '精确率', '召回率', 'F1分数']
    plt.figure(figsize=(15, 10))

    x = np.arange(len(metrics))
    width = 0.15
    n_models = len(models_results)

    for i, (model_name, _) in enumerate(models_results.items()):
        offset = width * (i - n_models / 2 + 0.5)
        values = df_comparison.iloc[i, 1:].values
        plt.bar(x + offset, values, width, label=model_name)

    plt.title('模型性能比较')
    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('model_comparison.png')

    return df_comparison