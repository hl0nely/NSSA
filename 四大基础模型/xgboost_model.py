# xgboost_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve, \
    f1_score, recall_score, precision_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings
import pickle

from 多模型.model_utils import add_performance_table, analyze_predictions

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

    # 使用XGBoost进行特征选择
    selector = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
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
    plt.title('XGBoost特征重要性')
    plt.bar(range(10), importance[indices[:10]], align='center')
    plt.xticks(range(10), [feature_names[i] for i in indices[:10]], rotation=90)
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png')

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
    plt.title('XGBoost阈值优化')
    plt.legend()
    plt.grid(True)
    plt.savefig('xgboost_threshold_optimization.png')

    return best_threshold


def train_and_evaluate(X_train, X_test, y_train, y_test, random_state=42):
    """训练和评估XGBoost模型"""
    # 确保输入是numpy数组
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # 分割验证集用于阈值优化(避免数据泄露)
    X_train_main, X_val, y_train_main, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # 设置XGBoost参数
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'n_estimators': 100,
        'random_state': random_state
    }

    # 创建XGBoost分类器
    model = xgb.XGBClassifier(**params)

    try:
        # 使用交叉验证获取更可靠的概率估计
        print("执行交叉验证训练...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        y_probs = np.zeros(len(X_test))

        # 记录每次交叉验证的训练和验证性能
        cv_results = []

        for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_main, y_train_main)):
            # 正确使用数组切片
            X_cv_train, X_cv_val = X_train_main[train_idx], X_train_main[val_idx]
            y_cv_train, y_cv_val = y_train_main[train_idx], y_train_main[val_idx]

            # 创建评估集
            eval_set = [(X_cv_train, y_cv_train), (X_cv_val, y_cv_val)]

            # 训练模型
            model.fit(
                X_cv_train, y_cv_train,
                eval_set=eval_set,
                verbose=False
            )

            # 记录训练过程
            cv_results.append(model.evals_result())

            # 累加预测概率
            y_probs += model.predict_proba(X_test)[:, 1] / cv.n_splits

        # 训练最终模型
        final_model = xgb.XGBClassifier(**params)
        final_model.fit(X_train, y_train)

        # 在验证集上寻找最佳阈值(避免数据泄露)
        val_probs = final_model.predict_proba(X_val)[:, 1]
        best_threshold = find_optimal_threshold(y_val, val_probs, "XGBoost")
        print(f"最佳阈值: {best_threshold:.4f}")

        # 使用最佳阈值进行测试集预测
        test_probs = final_model.predict_proba(X_test)[:, 1]
        y_pred = (test_probs >= best_threshold).astype(int)

        # 计算评估指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, test_probs)

        # 打印详细结果
        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1分数: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))

        # 打印混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        print("\n混淆矩阵:")
        print(cm)

        # 添加数值化结果分析
        metrics_df = add_performance_table(y_test, y_pred, test_probs, "XGBoost")
        error_df, interval_df = analyze_predictions(y_test, y_pred, test_probs, "XGBoost")

        print("\n性能指标表:")
        print(metrics_df)
        print("\n错误分析:")
        print(error_df)
        print("\n概率区间准确率:")
        print(interval_df)

        # 可视化混淆矩阵
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['正常', '攻击'],
                    yticklabels=['正常', '攻击'])
        plt.xlabel('预测')
        plt.ylabel('实际')
        plt.title('XGBoost混淆矩阵')
        plt.savefig('xgboost_confusion_matrix.png')

        # 可视化预测概率分布
        plt.figure(figsize=(10, 6))
        plt.hist(test_probs[y_test == 0], bins=50, alpha=0.5, label='正常流量')
        plt.hist(test_probs[y_test == 1], bins=50, alpha=0.5, label='攻击流量')
        plt.axvline(x=best_threshold, color='red', linestyle='--',
                    label=f'阈值 ({best_threshold:.3f})')
        plt.xlabel('预测概率')
        plt.ylabel('样本数量')
        plt.title('XGBoost预测概率分布')
        plt.legend()
        plt.grid(True)
        plt.savefig('xgboost_probability_distribution.png')

        # ROC曲线可视化
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC曲线 (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('XGBoost ROC曲线')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('xgboost_roc_curve.png')

        return final_model, best_threshold, accuracy, auc, precision, recall, f1, cv_results

    except Exception as e:
        print(f"XGBoost模型训练失败: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细的堆栈跟踪
        return None, 0.5, 0, 0, 0, 0, 0, None


# def train_and_evaluate(X_train, X_test, y_train, y_test, random_state=42):
#     """
#     训练和评估XGBoost模型
#     """
#     # 设置XGBoost参数
#     params = {
#         'objective': 'binary:logistic',
#         'eval_metric': 'auc',  # 使用AUC作为评估指标
#         'max_depth': 6,
#         'learning_rate': 0.1,
#         'subsample': 0.8,
#         'colsample_bytree': 0.8,
#         'min_child_weight': 1,
#         'n_estimators': 100,
#         'random_state': random_state
#     }
#
#     # 创建XGBoost分类器
#     model = xgb.XGBClassifier(**params)
#
#     # 使用交叉验证获取更可靠的概率估计
#     print("执行交叉验证训练...")
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     y_probs = np.zeros(len(X_test))
#
#     # 记录每次交叉验证的训练和验证性能
#     cv_results = []
#
#     for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train)):
#         X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
#         y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
#
#         # 创建评估集
#         eval_set = [(X_cv_train, y_cv_train), (X_cv_val, y_cv_val)]
#
#         # 训练模型
#         model.fit(
#             X_cv_train, y_cv_train,
#             eval_set=eval_set,
#             verbose=False
#         )
#
#         # 记录训练过程
#         cv_results.append(model.evals_result())
#
#         # 累加预测概率
#         y_probs += model.predict_proba(X_test)[:, 1] / cv.n_splits
#
#     # 寻找最佳阈值
#     best_threshold = find_optimal_threshold(y_test, y_probs)
#     print(f"最佳阈值: {best_threshold:.4f}")
#
#     # 使用最佳阈值进行预测
#     y_pred = (y_probs >= best_threshold).astype(int)
#
#     # 计算评估指标
#     accuracy = (y_pred == y_test).mean()
#     auc = roc_auc_score(y_test, y_probs)
#
#     # 打印详细结果
#     print(f"准确率: {accuracy:.4f}")
#     print(f"AUC: {auc:.4f}")
#     print("\n分类报告:")
#     print(classification_report(y_test, y_pred))
#
#     # 打印混淆矩阵
#     cm = confusion_matrix(y_test, y_pred)
#     print("\n混淆矩阵:")
#     print(cm)
#
#     # 可视化混淆矩阵
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['正常', '攻击'],
#                 yticklabels=['正常', '攻击'])
#     plt.xlabel('预测')
#     plt.ylabel('实际')
#     plt.title('XGBoost混淆矩阵')
#     plt.savefig('xgboost_confusion_matrix.png')
#
#     # 可视化预测概率分布
#     plt.figure(figsize=(10, 6))
#     plt.hist(y_probs[y_test == 0], bins=50, alpha=0.5, label='正常流量')
#     plt.hist(y_probs[y_test == 1], bins=50, alpha=0.5, label='攻击流量')
#     plt.axvline(x=best_threshold, color='red', linestyle='--',
#                 label=f'阈值 ({best_threshold:.3f})')
#     plt.xlabel('预测概率')
#     plt.ylabel('样本数量')
#     plt.title('XGBoost预测概率分布')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('xgboost_probability_distribution.png')
#
#     # ROC曲线可视化
#     fpr, tpr, _ = roc_curve(y_test, y_probs)
#     plt.figure(figsize=(10, 6))
#     plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC曲线 (AUC = {auc:.4f})')
#     plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('假阳性率')
#     plt.ylabel('真阳性率')
#     plt.title('XGBoost ROC曲线')
#     plt.legend(loc="lower right")
#     plt.grid(True)
#     plt.savefig('xgboost_roc_curve.png')
#
#     # 可视化学习曲线
#     plt.figure(figsize=(10, 6))
#
#     # 使用第一个交叉验证结果进行可视化
#     train_metric = cv_results[0]['validation_0']['auc']
#     val_metric = cv_results[0]['validation_1']['auc']
#
#     epochs = len(train_metric)
#     x_axis = range(0, epochs)
#
#     plt.plot(x_axis, train_metric, 'b-', label='训练 AUC')
#     plt.plot(x_axis, val_metric, 'r-', label='验证 AUC')
#
#     plt.legend()
#     plt.xlabel('迭代次数')
#     plt.ylabel('AUC')
#     plt.title('XGBoost学习曲线')
#     plt.grid()
#     plt.savefig('xgboost_learning_curve.png')
#
#     # SHAP值分析 (如果安装了shap库)
#     try:
#         import shap
#
#         # 训练最终模型
#         final_model = xgb.XGBClassifier(**params)
#
#         # 检查是否支持GPU
#         try:
#             final_model.fit(X_train, y_train)
#         except Exception as e:
#             print(f"GPU训练失败，回退到CPU训练: {e}")
#             # 回退到CPU训练
#             final_model = xgb.XGBClassifier(**params)
#             final_model.fit(X_train, y_train)
#
#         # 计算SHAP值
#         explainer = shap.TreeExplainer(final_model)
#         shap_values = explainer.shap_values(X_test[:100])  # 只使用前100个样本计算SHAP值
#
#         # 汇总图
#         plt.figure(figsize=(10, 8))
#         shap.summary_plot(shap_values, X_test[:100], plot_type="bar", show=False)
#         plt.title('XGBoost SHAP特征重要性')
#         plt.tight_layout()
#         plt.savefig('xgboost_shap_importance.png')
#
#         # 详细SHAP图
#         plt.figure(figsize=(10, 10))
#         shap.summary_plot(shap_values, X_test[:100], show=False)
#         plt.title('XGBoost SHAP特征影响')
#         plt.tight_layout()
#         plt.savefig('xgboost_shap_summary.png')
#
#     except ImportError:
#         print("未安装SHAP库，跳过SHAP值分析")
#
#     # 预测结果散点图
#     # 限制最多显示1000个样本点，以提高绘图效率
#     max_points = 1000
#     if len(y_test) > max_points:
#         # 随机选择一部分样本用于可视化
#         idx = np.random.choice(len(y_test), max_points, replace=False)
#         y_test_sample = y_test[idx]
#         y_pred_sample = y_pred[idx]
#
#         plt.figure(figsize=(10, 6))
#         plt.scatter(range(len(y_test_sample)), y_test_sample, color='blue', alpha=0.5, label='实际值')
#         plt.scatter(range(len(y_pred_sample)), y_pred_sample, color='red', alpha=0.5, label='预测值')
#     else:
#         plt.figure(figsize=(10, 6))
#         plt.scatter(range(len(y_test)), y_test, color='blue', alpha=0.5, label='实际值')
#         plt.scatter(range(len(y_pred)), y_pred, color='red', alpha=0.5, label='预测值')
#
#     plt.xlabel('样本索引')
#     plt.ylabel('标签值')
#     plt.title('XGBoost预测结果可视化')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('xgboost_prediction_results.png')
#
#     return final_model, best_threshold, accuracy, auc


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
    print("\n训练和评估XGBoost模型...")
    model, threshold, accuracy, auc = train_and_evaluate(
        X_train_balanced, X_test, y_train_balanced, y_test
    )

    # 7. 保存模型
    print("\n保存最终模型...")
    with open('XGBOOST/unsw_nb15_xgboost_model.pkl', 'wb') as f:
        pickle.dump({
            'model': model,
            'threshold': threshold,
            'selected_features': selected_features,
            'accuracy': accuracy,
            'auc': auc
        }, f)

    print("XGBoost网络检测模型训练完成！")


if __name__ == "__main__":
    main() 