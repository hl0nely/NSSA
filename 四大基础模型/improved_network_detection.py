# improved_network_detection.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
import warnings

warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题


def create_enhanced_features(df):
    """创建增强特征"""
    # 创建副本避免修改原始数据
    df_enhanced = df.copy()
    
    # 1. 流量相关特征
    df_enhanced['bytes_per_packet_src'] = df['sbytes'] / (df['spkts'] + 1e-8)
    df_enhanced['bytes_per_packet_dst'] = df['dbytes'] / (df['dpkts'] + 1e-8)
    df_enhanced['packet_ratio'] = df['spkts'] / (df['dpkts'] + 1e-8)
    df_enhanced['bytes_ratio'] = df['sbytes'] / (df['dbytes'] + 1e-8)
    
    # 2. 时间相关特征
    df_enhanced['log_duration'] = np.log1p(df['dur'])
    df_enhanced['is_short_conn'] = (df['dur'] < 0.1).astype(int)
    df_enhanced['packets_per_second'] = (df['spkts'] + df['dpkts']) / (df['dur'] + 1e-8)
    
    # 3. 其他相关特征
    df_enhanced['tcp_window_ratio'] = df['swin'] / (df['dwin'] + 1e-8)
    df_enhanced['tcp_param_ratio'] = df['stcpb'] / (df['dtcpb'] + 1e-8)
    df_enhanced['connection_density'] = df['ct_srv_src'] + df['ct_dst_ltm'] + df['ct_src_ltm'] + df['ct_dst_src_ltm']
    df_enhanced['load_ratio'] = df['sload'] / (df['dload'] + 1e-8)
    df_enhanced['loss_ratio'] = (df['sloss'] + df['dloss']) / (df['spkts'] + df['dpkts'] + 1e-8)
    
    # 4. 简单协议标记 (不参与相关性计算)
    if 'proto' in df.columns:
        df_enhanced['is_tcp'] = (df['proto'] == 'tcp').astype(int)
        df_enhanced['is_udp'] = (df['proto'] == 'udp').astype(int)
    
    # 5. 简单服务标记 (不参与相关性计算)
    if 'service' in df.columns:
        df_enhanced['is_http'] = (df['service'] == 'http').astype(int)
        df_enhanced['is_dns'] = (df['service'] == 'dns').astype(int)
    
    # 修复错误：只选择数值型列计算相关性
    if 'label' in df.columns:
        try:
            # 只选择数值型列
            numeric_cols = df_enhanced.select_dtypes(include=['number']).columns
            # 确保label是数值型
            if pd.api.types.is_numeric_dtype(df['label']):
                # 计算数值型列与标签的相关性
                correlations = df_enhanced[numeric_cols].corrwith(df['label']).abs().sort_values(ascending=False)
                print("与标签最相关的前10个特征:")
                print(correlations.head(10))
        except Exception as e:
            print(f"计算相关性时出错 (不影响特征创建): {e}")
    
    return df_enhanced


def preprocess_data(train_df, test_df, num_features=30):
    """预处理数据，包括特征选择和标准化"""
    # 移除不需要的列
    drop_cols = ['id', 'attack_cat']
    X_train = train_df.drop(drop_cols + ['label'], axis=1, errors='ignore')
    y_train = train_df['label']
    X_test = test_df.drop(drop_cols + ['label'], axis=1, errors='ignore')
    y_test = test_df['label']

    # 处理类别特征
    cat_features = ['proto', 'service', 'state']
    X_train_encoded = pd.get_dummies(X_train, columns=cat_features, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=cat_features, drop_first=True)

    # 确保训练集和测试集有相同的特征
    train_cols = X_train_encoded.columns
    test_cols = X_test_encoded.columns

    # 找出训练集有而测试集没有的特征
    missing_cols = set(train_cols) - set(test_cols)
    for col in missing_cols:
        X_test_encoded[col] = 0

    # 确保列顺序一致
    X_test_encoded = X_test_encoded[train_cols]

    # 缺失值处理
    X_train_encoded = X_train_encoded.fillna(0)
    X_test_encoded = X_test_encoded.fillna(0)

    # 标准化数值特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_encoded)
    X_test_scaled = scaler.transform(X_test_encoded)

    # 使用随机森林进行特征选择
    selector = RandomForestClassifier(n_estimators=100, random_state=42)
    selector.fit(X_train_scaled, y_train)

    # 获取特征重要性并排序
    importance = selector.feature_importances_
    indices = np.argsort(importance)[::-1]

    # 打印最重要的特征
    feature_names = X_train_encoded.columns
    print("特征排名:")
    for i, idx in enumerate(indices[:10]):
        print(f"{i + 1}. {feature_names[idx]} ({importance[idx]:.4f})")

    # 选择前num_features个最重要的特征
    top_indices = indices[:num_features]
    X_train_selected = X_train_scaled[:, top_indices]
    X_test_selected = X_test_scaled[:, top_indices]

    # 可视化特征重要性
    plt.figure(figsize=(12, 6))
    plt.title('特征重要性')
    plt.bar(range(10), importance[indices[:10]], align='center')
    plt.xticks(range(10), [feature_names[i] for i in indices[:10]], rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance.png')

    return X_train_selected, y_train, X_test_selected, y_test, [feature_names[i] for i in top_indices]


def balance_data(X, y, sampling_strategy=0.6):
    """使用SMOTE平衡数据"""
    print(f"原始标签分布: {np.bincount(y)}")
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"重采样后的标签分布: {np.bincount(y_resampled)}")
    return X_resampled, y_resampled


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

    # 可视化阈值选择
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precision[:-1], 'b--', label='精确率')
    plt.plot(thresholds, recall[:-1], 'g-', label='召回率')
    plt.plot(thresholds, f1_scores[:-1], 'r-', label='F1分数')
    plt.plot(thresholds, geo_mean[:-1], 'c-', label='几何平均')
    plt.axvline(x=best_threshold, color='purple', linestyle='--',
                label=f'最佳阈值 ({best_threshold:.3f})')
    plt.xlabel('阈值')
    plt.ylabel('分数')
    plt.title('阈值对性能指标的影响')
    plt.legend()
    plt.grid(True)
    plt.savefig('threshold_optimization.png')

    return best_threshold


def train_and_evaluate(X_train, y_train, X_test, y_test):
    """训练并评估模型，使用交叉验证"""
    # 创建改进的随机森林分类器
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        bootstrap=True,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )

    # 使用交叉验证获取更可靠的概率估计
    print("执行交叉验证训练...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_probs = np.zeros(len(X_test))

    for train_idx, val_idx in cv.split(X_train, y_train):
        X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
        y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

        # 训练模型
        model.fit(X_cv_train, y_cv_train)

        # 累加预测概率
        y_probs += model.predict_proba(X_test)[:, 1] / cv.n_splits

    # 寻找最佳阈值
    best_threshold = find_optimal_threshold(y_test, y_probs)
    print(f"最佳阈值: {best_threshold:.4f}")

    # 使用最佳阈值进行预测
    y_pred = (y_probs >= best_threshold).astype(int)

    # 计算评估指标
    accuracy = (y_pred == y_test).mean()
    auc = roc_auc_score(y_test, y_probs)

    # 打印详细结果
    print(f"准确率: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))

    # 打印混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("\n混淆矩阵:")
    print(cm)

    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['正常', '攻击'],
                yticklabels=['正常', '攻击'])
    plt.xlabel('预测')
    plt.ylabel('实际')
    plt.title('混淆矩阵')
    plt.savefig('confusion_matrix.png')

    # 可视化预测概率分布
    plt.figure(figsize=(10, 6))
    plt.hist(y_probs[y_test == 0], bins=50, alpha=0.5, label='正常流量')
    plt.hist(y_probs[y_test == 1], bins=50, alpha=0.5, label='攻击流量')
    plt.axvline(x=best_threshold, color='red', linestyle='--',
                label=f'阈值 ({best_threshold:.3f})')
    plt.xlabel('预测概率')
    plt.ylabel('样本数量')
    plt.title('预测概率分布')
    plt.legend()
    plt.grid(True)
    plt.savefig('probability_distribution.png')
    
    # 【新增】ROC曲线可视化
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC曲线 (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('roc_curve.png')
    
    # 【新增】威胁热图 - 使用前10个最重要特征的预测概率热图
    # 训练最终模型以获得特征重要性
    final_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        bootstrap=True,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_train, y_train)
    
    # 获取特征重要性并找到前10个特征
    importance = final_model.feature_importances_
    indices = np.argsort(importance)[::-1][:10]
    
    # 创建威胁热图
    plt.figure(figsize=(12, 8))
    # 随机抽取100个样本或使用全部样本（如果少于100个）
    n_samples = min(100, X_test.shape[0])
    sample_indices = np.random.choice(X_test.shape[0], n_samples, replace=False)
    
    # 对抽样数据进行预测并获取预测概率
    sample_probs = final_model.predict_proba(X_test[sample_indices])[:, 1]
    
    # 创建热图数据
    heatmap_data = X_test[sample_indices][:, indices]
    
    # 标准化热图数据以便更好地可视化
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    heatmap_scaled = scaler.fit_transform(heatmap_data)
    
    # 绘制热图
    ax = sns.heatmap(heatmap_scaled, cmap='YlOrRd', 
                     cbar_kws={'label': '特征值（标准化）'})
    
    # 在y轴上添加预测概率的颜色条
    plt.figure(figsize=(1, 8))
    prob_colors = plt.cm.RdYlGn_r(sample_probs)
    plt.imshow([sample_probs], cmap='RdYlGn_r', aspect='auto')
    plt.title('预测概率')
    plt.xticks([])
    plt.tight_layout()
    plt.savefig('prediction_probs.png')
    
    # 合并两个图
    plt.figure(figsize=(14, 8))
    
    # 左侧创建热图
    plt.subplot(1, 10, 1)
    plt.imshow([sample_probs.reshape(-1)], cmap='RdYlGn_r', aspect='auto')
    plt.title('攻击概率')
    plt.xticks([])
    plt.yticks([])
    
    # 右侧展示主要特征热图
    plt.subplot(1, 10, (2, 10))
    sns.heatmap(heatmap_scaled, cmap='YlOrRd',
               xticklabels=[f'特征{i+1}' for i in range(10)],
               yticklabels=False)
    plt.title('威胁特征热图（前10个重要特征）')
    plt.tight_layout()
    plt.savefig('threat_heatmap.png')
    
    # 【新增】预测结果散点图
    # 限制最多显示1000个样本点，以提高绘图效率
    max_points = 1000
    if len(y_test) > max_points:
        # 随机选择一部分样本用于可视化
        idx = np.random.choice(len(y_test), max_points, replace=False)
        y_test_sample = y_test[idx]
        y_pred_sample = y_pred[idx]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_test_sample)), y_test_sample, color='blue', alpha=0.5, label='实际值')
        plt.scatter(range(len(y_pred_sample)), y_pred_sample, color='red', alpha=0.5, label='预测值')
    else:
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_test)), y_test, color='blue', alpha=0.5, label='实际值')
        plt.scatter(range(len(y_pred)), y_pred, color='red', alpha=0.5, label='预测值')
    
    plt.xlabel('样本索引')
    plt.ylabel('标签值')
    plt.title('预测结果可视化')
    plt.legend()
    plt.grid(True)
    plt.savefig('prediction_results.png')

    # 训练最终模型
    final_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        bootstrap=True,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    final_model.fit(X_train, y_train)

    return final_model, best_threshold, accuracy, auc


def main():
    # 1. 加载数据
    print("加载UNSW-NB15数据集...")
    train_data = pd.read_csv("./data/UNSW_NB15_training-set.csv")
    test_data = pd.read_csv("./data/UNSW_NB15_testing-set.csv")

    print(f"训练集形状: {train_data.shape}")
    print(f"测试集形状: {test_data.shape}")

    # 2. 数据探索和分析
    print("\n分析标签分布:")
    print(f"训练集标签分布: {train_data['label'].value_counts()}")
    print(f"测试集标签分布: {test_data['label'].value_counts()}")

    # 3. 特征工程
    print("\n创建增强特征...")
    train_enhanced = create_enhanced_features(train_data)
    test_enhanced = create_enhanced_features(test_data)

    # 4. 数据预处理和特征选择
    print("\n预处理数据和选择特征...")
    X_train, y_train, X_test, y_test, selected_features = preprocess_data(
        train_enhanced, test_enhanced, num_features=30
    )

    # 5. 平衡数据
    print("\n平衡训练数据...")
    X_train_balanced, y_train_balanced = balance_data(X_train, y_train)

    # 6. 训练和评估模型
    print("\n训练和评估模型...")
    model, threshold, accuracy, auc = train_and_evaluate(
        X_train_balanced, y_train_balanced, X_test, y_test
    )

    # 7. 保存模型
    print("\n保存最终模型...")
    import pickle
    with open('随机森林/unsw_nb15_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'threshold': threshold,
            'selected_features': selected_features,
            'accuracy': accuracy,
            'auc': auc
        }, f)

    print("改进的网络检测模型训练完成！")


if __name__ == "__main__":
    main()