import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import itertools
import numpy as np
import tensorflow as tf

# 定义技能列表
skills = ['java', 'go', 'python', 'spring', 'rpc', 'gin', 'ai', 'mysql', 'linux', '消息队列']

# 生成模型样本数据
# 生成模型样本数据
def generate_samples(num_samples):
    samples = []
    for age in range(22, 46):
        for num_skills in range(len(skills) + 1):  # 循环不同技能数量
            for skill_combination in itertools.combinations(skills, num_skills):  # 组合技能
                pass_rate = 100 - age * 1.5 - num_skills * 2
                if 'ai' in skill_combination:
                    pass_rate = 100
                pass_rate = max(min(pass_rate, 100), 0)  # 将通过率截断为不超过100
                feature = [age] + [1 if skill in skill_combination else 0 for skill in skills]  # 将年龄和技能转换为特征向量
                samples.append([feature, pass_rate])
    return samples

def forecast(age, input_array):
    # 生成样本数据
    samples = generate_samples(num_samples=500)

    # 提取特征和目标
    X_train = np.array([sample[0] for sample in samples], dtype=float)
    Y_train = np.array([sample[1] for sample in samples], dtype=float)

    # 构建 TensorFlow 模型
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(skills) + 1,)),  # 输入层，特征数量为技能数量加上年龄
        tf.keras.layers.Dense(1)  # 输出层，输出单个值，即通过率
    ])

    # 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error')

    # 训练模型
    model.fit(X_train, Y_train, epochs=50, batch_size=32, verbose=0)

    # 预测新样本
    # 新样本：年龄为30岁，具有java、go和ai技能
    new_sample = np.array([[age] + [1 if skill in input_array else 0 for skill in skills]], dtype=float)
    prediction = model.predict(new_sample)

    # 如果预测通过率超过100，则取100
    prediction = np.minimum(prediction, 100)

    # 保留两位小数
    prediction = np.around(prediction, decimals=2)

    print("\n预测通过率:")
    print(str(prediction[0][0]) + '%')


if __name__ == '__main__':
    # 输入你的年龄和掌握的技能组，预测你的面试通过率
    forecast(28, ['java', 'python', 'go', 'ai'])
