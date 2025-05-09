# lstm_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import warnings
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os

from 多模型.model_utils import add_performance_table, analyze_predictions

warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)
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


def preprocess_data_for_lstm(train_df, test_df):
    """特别为LSTM准备数据，包括序列化处理"""
    # 移除不需要的列
    drop_cols = ['id', 'attack_cat']
    X_train = train_df.drop(drop_cols + ['label'], axis=1, errors='ignore')
    y_train = train_df['label']
    X_test = test_df.drop(drop_cols + ['label'], axis=1, errors='ignore')
    y_test = test_df['label']

    # 处理类别特征 - 对于LSTM，我们可以使用标签编码而不是独热编码
    cat_features = ['proto', 'service', 'state']

    # 为每个类别特征创建标签编码器
    encoders = {}
    for feature in cat_features:
        if feature in X_train.columns:
            # 修改：合并训练集和测试集中的所有可能类别
            all_categories = set(X_train[feature].unique()).union(set(X_test[feature].unique()))
            print(f"特征 {feature} 中的类别数量: {len(all_categories)}")

            le = LabelEncoder()
            le.fit(list(all_categories))  # 使用所有可能类别拟合编码器

            X_train[feature] = le.transform(X_train[feature])
            X_test[feature] = le.transform(X_test[feature])

            encoders[feature] = le

    # 缺失值处理
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # 标准化数值特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 对于LSTM，我们需要重塑数据为3D格式 [samples, time_steps, features]
    X_train_lstm = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
    X_test_lstm = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

    print(f"LSTM输入形状: {X_train_lstm.shape}")

    return X_train_lstm, y_train, X_test_lstm, y_test, X_train.columns.tolist(), scaler


def balance_data_for_lstm(X, y, sampling_strategy=0.6):
    """为LSTM准备平衡数据集"""
    print(f"原始标签分布: {np.bincount(y)}")
    
    # 对于LSTM数据，我们需要先将3D数据展平，应用SMOTE，然后再重塑回3D
    n_samples, time_steps, n_features = X.shape
    X_reshaped = X.reshape(n_samples, time_steps * n_features)
    
    # 应用SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_reshaped, y)
    
    # 重塑回3D
    X_resampled = X_resampled.reshape(X_resampled.shape[0], time_steps, n_features)
    
    print(f"重采样后的标签分布: {np.bincount(y_resampled)}")
    print(f"重采样后LSTM输入形状: {X_resampled.shape}")
    
    return X_resampled, y_resampled


def create_lstm_model(input_shape):
    """创建LSTM模型"""
    model = Sequential([
        # 双向LSTM层
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        
        # 第二个LSTM层
        Bidirectional(LSTM(64)),
        Dropout(0.3),
        
        # 全连接层
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        # 输出层
        Dense(1, activation='sigmoid')
    ])
    
    # 编译模型
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    # 打印模型摘要
    model.summary()
    
    return model


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
    plt.title('LSTM阈值优化')
    plt.legend()
    plt.grid(True)
    plt.savefig('lstm_threshold_optimization.png')

    return best_threshold


# def train_and_evaluate_lstm(X_train, y_train, X_test, y_test):
#     """训练并评估LSTM模型"""
#     # 从训练集中拆分出验证集
#     X_train_split, X_val, y_train_split, y_val = train_test_split(
#         X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
#     )
#
#     # 创建模型
#     input_shape = (X_train.shape[1], X_train.shape[2])
#     model = create_lstm_model(input_shape)
#
#     # 设置回调函数
#     callbacks = [
#         EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True),
#         ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001),
#         ModelCheckpoint('best_lstm_model.h5', monitor='val_auc', save_best_only=True, mode='max')
#     ]
#
#     # 训练模型
#     print("训练LSTM模型...")
#     history = model.fit(
#         X_train_split, y_train_split,
#         validation_data=(X_val, y_val),
#         epochs=30,
#         batch_size=128,
#         callbacks=callbacks,
#         verbose=1
#     )
#
#     # 加载最佳模型
#     model.load_weights('best_lstm_model.h5')
#
#     # 预测概率
#     y_probs = model.predict(X_test).flatten()
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
#     # 可视化训练历史
#     plt.figure(figsize=(12, 5))
#
#     # 绘制精度
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'], label='训练集准确率')
#     plt.plot(history.history['val_accuracy'], label='验证集准确率')
#     plt.title('LSTM模型准确率')
#     plt.xlabel('Epoch')
#     plt.ylabel('准确率')
#     plt.legend()
#     plt.grid(True)
#
#     # 绘制损失
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'], label='训练集损失')
#     plt.plot(history.history['val_loss'], label='验证集损失')
#     plt.title('LSTM模型损失')
#     plt.xlabel('Epoch')
#     plt.ylabel('损失')
#     plt.legend()
#     plt.grid(True)
#
#     plt.tight_layout()
#     plt.savefig('lstm_training_history.png')
#
#     # 可视化混淆矩阵
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=['正常', '攻击'],
#                 yticklabels=['正常', '攻击'])
#     plt.xlabel('预测')
#     plt.ylabel('实际')
#     plt.title('LSTM混淆矩阵')
#     plt.savefig('lstm_confusion_matrix.png')
#
#     # 可视化预测概率分布
#     plt.figure(figsize=(10, 6))
#     plt.hist(y_probs[y_test == 0], bins=50, alpha=0.5, label='正常流量')
#     plt.hist(y_probs[y_test == 1], bins=50, alpha=0.5, label='攻击流量')
#     plt.axvline(x=best_threshold, color='red', linestyle='--',
#                 label=f'阈值 ({best_threshold:.3f})')
#     plt.xlabel('预测概率')
#     plt.ylabel('样本数量')
#     plt.title('LSTM预测概率分布')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('lstm_probability_distribution.png')
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
#     plt.title('LSTM ROC曲线')
#     plt.legend(loc="lower right")
#     plt.grid(True)
#     plt.savefig('lstm_roc_curve.png')
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
#     plt.title('LSTM预测结果可视化')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('lstm_prediction_results.png')
#
#     # 绘制AUC曲线
#     plt.figure(figsize=(10, 6))
#     plt.plot(history.history['auc'], label='训练集AUC')
#     plt.plot(history.history['val_auc'], label='验证集AUC')
#     plt.title('LSTM模型AUC曲线')
#     plt.xlabel('Epoch')
#     plt.ylabel('AUC')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('lstm_auc_curve.png')
#
#     return model, best_threshold, accuracy, auc, history.history


def train_and_evaluate_lstm(X_train, y_train, X_test, y_test):
    """训练并评估LSTM模型"""
    # 确保输入是numpy数组
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    try:
        # 从训练集中拆分出验证集
        X_train_main, X_val, y_train_main, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # 创建模型
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = create_lstm_model(input_shape)

        # 设置回调函数
        callbacks = [
            EarlyStopping(monitor='val_auc', patience=10, mode='max', restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001),
            ModelCheckpoint('best_lstm_model.keras', monitor='val_auc', save_best_only=True, mode='max')
        ]

        # 训练模型
        print("训练LSTM模型...")
        history = model.fit(
            X_train_main, y_train_main,
            validation_data=(X_val, y_val),
            epochs=30,
            batch_size=128,
            callbacks=callbacks,
            verbose=1
        )

        # 加载最佳模型
        model.load_weights('best_lstm_model.keras')

        # 在验证集上寻找最佳阈值(避免数据泄露)
        val_probs = model.predict(X_val).flatten()
        best_threshold = find_optimal_threshold(y_val, val_probs, "LSTM")
        print(f"最佳阈值: {best_threshold:.4f}")

        # 使用最佳阈值进行测试集预测
        test_probs = model.predict(X_test).flatten()
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

        # 添加数值化结果分析
        metrics_df = add_performance_table(y_test, y_pred, test_probs, "LSTM")
        error_df, interval_df = analyze_predictions(y_test, y_pred, test_probs, "LSTM")

        print("\n性能指标表:")
        print(metrics_df)
        print("\n错误分析:")
        print(error_df)
        print("\n概率区间准确率:")
        print(interval_df)

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
        plt.title('LSTM混淆矩阵')
        plt.savefig('lstm_confusion_matrix.png')

        # 可视化训练历史
        plt.figure(figsize=(12, 5))

        # 绘制精度
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='训练集准确率')
        plt.plot(history.history['val_accuracy'], label='验证集准确率')
        plt.title('LSTM模型准确率')
        plt.xlabel('Epoch')
        plt.ylabel('准确率')
        plt.legend()
        plt.grid(True)

        # 绘制损失
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='训练集损失')
        plt.plot(history.history['val_loss'], label='验证集损失')
        plt.title('LSTM模型损失')
        plt.xlabel('Epoch')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('lstm_training_history.png')

        # 可视化预测概率分布
        plt.figure(figsize=(10, 6))
        plt.hist(test_probs[y_test == 0], bins=50, alpha=0.5, label='正常流量')
        plt.hist(test_probs[y_test == 1], bins=50, alpha=0.5, label='攻击流量')
        plt.axvline(x=best_threshold, color='red', linestyle='--',
                    label=f'阈值 ({best_threshold:.3f})')
        plt.xlabel('预测概率')
        plt.ylabel('样本数量')
        plt.title('LSTM预测概率分布')
        plt.legend()
        plt.grid(True)
        plt.savefig('lstm_probability_distribution.png')

        # ROC曲线可视化
        fpr, tpr, _ = roc_curve(y_test, test_probs)
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC曲线 (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率')
        plt.ylabel('真阳性率')
        plt.title('LSTM ROC曲线')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig('lstm_roc_curve.png')

        # 预测结果散点图
        try:
            max_points = 1000
            if len(y_test) > max_points:
                idx = np.random.choice(len(y_test), max_points, replace=False)
                y_test_sample = y_test[idx]
                y_pred_sample = y_pred[idx]

                # 确保长度一致
                indices = np.arange(len(y_test_sample))

                plt.figure(figsize=(10, 6))
                plt.scatter(indices, y_test_sample, color='blue', alpha=0.5, label='实际值')
                plt.scatter(indices, y_pred_sample, color='red', alpha=0.5, label='预测值')
            else:
                indices = np.arange(len(y_test))

                plt.figure(figsize=(10, 6))
                plt.scatter(indices, y_test, color='blue', alpha=0.5, label='实际值')
                plt.scatter(indices, y_pred, color='red', alpha=0.5, label='预测值')

            plt.xlabel('样本索引')
            plt.ylabel('标签值')
            plt.title('LSTM预测结果可视化')
            plt.legend()
            plt.grid(True)
            plt.savefig('lstm_prediction_results.png')
        except Exception as e:
            print(f"绘制预测结果散点图时出错 (不影响模型): {e}")

        # 绘制AUC曲线
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['auc'], label='训练集AUC')
        plt.plot(history.history['val_auc'], label='验证集AUC')
        plt.title('LSTM模型AUC曲线')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.grid(True)
        plt.savefig('lstm_auc_curve.png')

        return model, best_threshold, accuracy, auc, precision, recall, f1, history.history

    except Exception as e:
        print(f"LSTM模型训练失败: {str(e)}")
        import traceback
        traceback.print_exc()  # 打印详细的堆栈跟踪
        return None, 0.5, 0, 0, 0, 0, 0, None




def main():
    # 设置GPU内存增长，避免占用全部GPU内存
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 限制GPU内存使用，让TensorFlow按需分配内存
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"发现 {len(gpus)} 个GPU设备，已启用内存增长模式")
        except RuntimeError as e:
            print(f"GPU配置失败: {e}")
    else:
        print("未检测到GPU设备，将使用CPU训练")

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

    # 4. 数据预处理
    print("\n预处理数据...")
    X_train, y_train, X_test, y_test, feature_names, scaler = preprocess_data_for_lstm(
        train_enhanced, test_enhanced
    )

    # 5. 平衡数据
    print("\n平衡训练数据...")
    X_train_balanced, y_train_balanced = balance_data_for_lstm(X_train, y_train)

    # 6. 训练和评估模型
    print("\n训练和评估LSTM模型...")
    model, threshold, accuracy, auc, history = train_and_evaluate_lstm(
        X_train_balanced, y_train_balanced, X_test, y_test
    )

    # 7. 保存模型和相关信息
    print("\n保存最终模型...")
    # 保存Keras模型
    model.save('lstm_model.h5')
    
    # 保存阈值和评估指标
    results = {
        'threshold': threshold,
        'accuracy': accuracy,
        'auc': auc,
        'feature_names': feature_names
    }
    
    with open('LSTM/lstm_model_info.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    # 保存标准化器
    with open('LSTM/lstm_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    print("LSTM网络检测模型训练完成！")


if __name__ == "__main__":
    # 设置TensorFlow日志级别
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=默认, 1=不显示INFO, 2=不显示INFO/WARNING, 3=不显示所有
    main() 