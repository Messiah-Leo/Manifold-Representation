import numpy as np


def random_image(initial_vector, num_per_class, num_of_training, num_of_class, norm_type):
    '''
    读取数据并划分
    :param initial_vector: 数据矩阵(行为特征，列为样本)
    :param num_per_class: 每类样本数
    :param num_of_training: 训练样本数
    :param num_of_class: 类数
    :param norm_type: 归一化处理（1.mean-zero vector normalization; 2.std normalization; 3.max-min normalization; 4.L2 norm normalization）
    :return: TrainingVector, TestingVector, ID_Train(int), ID_Test(int)
    '''
    # initial_vector is the OriginalImagevector matrix
    dim, num_of_all_image = initial_vector.shape

    ID = np.arange(1, num_of_all_image + 1)
    ID_Train = np.array([])
    ID_Test = np.array([])

    for i in range(1, num_of_class + 1):
        flag = np.random.permutation(num_per_class)
        ID[((i - 1) * num_per_class): (i * num_per_class)] = flag + (i - 1) * num_per_class

        ID_Train = np.concatenate((ID_Train, ID[((i - 1) * num_per_class):((i - 1) * num_per_class + num_of_training)]))
        ID_Test = np.concatenate((ID_Test, ID[((i - 1) * num_per_class + num_of_training): (i * num_per_class)]))

    TrainingVector = initial_vector[:, ID_Train.astype(int) - 1]
    TestingVector = initial_vector[:, ID_Test.astype(int) - 1]

    # Data normalizations
    if norm_type == 1:  # mean-zero vector normalization
        TrV = np.mean(TrainingVector, axis=1)
        TeV = np.mean(TestingVector, axis=1)
        MTr = np.tile(TrV.reshape(dim, 1), (1, TrainingVector.shape[1]))
        MTe = np.tile(TeV.reshape(dim, 1), (1, TestingVector.shape[1]))

        TrainingVector = TrainingVector - MTr
        TestingVector = TestingVector - MTe

    elif norm_type == 2:  # std normalization
        stdTrV = np.std(TrainingVector, axis=1)
        stdTeV = np.std(TestingVector, axis=1)
        stdMTr = np.tile(stdTrV.reshape(dim, 1), (1, TrainingVector.shape[1]))
        stdMTe = np.tile(stdTeV.reshape(dim, 1), (1, TestingVector.shape[1]))

        TrainingVector = TrainingVector / stdMTr
        TestingVector = TestingVector / stdMTe

    elif norm_type == 3:  # max-min normalization
        RTrV = np.max(TrainingVector, axis=1) - np.min(TrainingVector, axis=1)
        RTeV = np.max(TestingVector, axis=1) - np.min(TestingVector, axis=1)
        RTr = np.tile(RTrV.reshape(dim, 1), (1, TrainingVector.shape[1]))
        RTe = np.tile(RTeV.reshape(dim, 1), (1, TestingVector.shape[1]))

        TrainingVector = TrainingVector / RTr
        TestingVector = TestingVector / RTe

    elif norm_type == 4:  # L2 norm normalization
        N2TrV = np.sqrt(np.sum(TrainingVector * TrainingVector, axis=1))
        N2TeV = np.sqrt(np.sum(TestingVector * TestingVector, axis=1))
        N2Tr = np.tile(N2TrV.reshape(dim, 1), (1, TrainingVector.shape[1]))
        N2Te = np.tile(N2TeV.reshape(dim, 1), (1, TestingVector.shape[1]))

        TrainingVector = TrainingVector / N2Tr
        TestingVector = TestingVector / N2Te

    return TrainingVector.T, TestingVector.T, ID_Train.astype(int), ID_Test.astype(int)


if __name__ == "__main__":
    import scipy.io

    # 读取MAT文件
    mat_data = scipy.io.loadmat('./数据汇总/Segment.mat')
    # 查看MAT文件中的变量
    print(mat_data.keys())
    # 获取特定变量的值
    variable_value = mat_data['data']
    TrainingVector, TestingVector, ID_Train, ID_Test = random_image(variable_value.T, 330, 200, 7, 1)
    print(ID_Train)
