import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
from pandas_confusion import ConfusionMatrix

class neural_net:

    def __init__(self, num_feature, num_class ):
        self.num_feature = num_feature
        self.num_class = num_class
        self.listFeature = ['i' for i in range(75)]
        self.feature_dict = {i:label for i, label in zip(
                    range(75), tuple(self.listFeature))}
        self.neural_struct = None
        self.weight_in = None
        self.weight_out = None
        self.data_placeholder = tf.placeholder("float", [None, self.num_feature])
        self.answer_placeholder = tf.placeholder("float", [None, self.num_class])
        self.train_op = None
        self.predict_op = None
        self.training_data = None
        self.training_answer = None
        self.testing_data = None
        self.testing_answer = None
        self.sess = None

    def init_weights(self,shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    def model(self, X, w_h, w_o):
        h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
        return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

    def create_struct(self, num_neural):
        self.weight_in = self.init_weights([self.num_feature, num_neural])
        self.weight_out = self.init_weights([num_neural, self.num_class])
        self.neural_struct = self.model(self.data_placeholder, self.weight_in, self.weight_out)
        cross = tf.nn.softmax_cross_entropy_with_logits(self.neural_struct, self.answer_placeholder)
        cost = tf.reduce_mean(cross)
        self.train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
        self.predict_op = tf.argmax(self.neural_struct, 1)

    def training(self, epoch):
        self.sess = tf.Session()
        init = tf.initialize_all_variables()
        self.sess.run(init)
        for i in range(epoch):
            for start, end in zip(range(0, len(self.training_data), 128), range(128, len(self.training_data), 128)):
                self.sess.run(self.train_op, feed_dict={self.data_placeholder: self.training_data[start:end],
                                                        self.answer_placeholder: self.training_answer[start:end]})
            print i, np.mean(np.argmax(self.testing_answer, axis=1) ==
                             self.sess.run(self.predict_op, feed_dict={self.data_placeholder: self.testing_data,
                                                                       self.answer_placeholder: self.testing_answer}))

    def data_input(self, data_file, type_set="train"):
        df = pd.io.parsers.read_csv(
            filepath_or_buffer=data_file,
            header=None,
            sep=',',
            )
        df.columns = [l for i, l in sorted(self.feature_dict.items())] + ['class label']
        df.dropna(how="all", inplace=True) # to drop the empty line at file-end
        df.tail()
        data = df[[i for i in range(self.num_feature)]].values

        answer = df['class label'].values
        answerList = [[0]*3 for x in range(len(answer))]
        for x in range(len(answerList)):
            answerList[x][int(answer[x])] = 1
        if type_set == "train":
            self.training_data = np.array(data.tolist(), dtype=np.float32)
            self.training_answer = np.array(answerList)
        elif type_set == "test" :
            self.testing_data = np.array(data.tolist(), dtype=np.float32)
            self.testing_answer = np.array(answerList)

    def accuracy_info(self):
        answerListB = self.testing_answer.tolist()
        answerList = [answer.index(1) for answer in answerListB]
        cm = ConfusionMatrix(answerList,
                             self.sess.run(self.predict_op,
                                           feed_dict={self.data_placeholder: self.testing_data,
                                                      self.answer_placeholder: self.testing_answer}))
        cmData = cm.to_array('a')
        acc = [cmData[0][0], cmData[1][1], cmData[2][2]]
        print cm
        print acc



# def init_weights(shape):
#     return tf.Variable(tf.random_normal(shape, stddev=0.01))
#
# def model(X, w_h, w_o):
#     h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
#     return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us
#
# listFeature = ['i' for i in range(75)]
# feature_dict = {i:label for i,label in zip(
#             range(75), tuple(listFeature))}
#
# df = pd.io.parsers.read_csv(
#     filepath_or_buffer='/home/sayong/Project/AVCS/Car-counter-using-python-opencv/list_data_raw.csv',
#     header=None,
#     sep=',',
#     )
# df.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
# df.dropna(how="all", inplace=True) # to drop the empty line at file-end
# df.tail()
#
# df_test = pd.io.parsers.read_csv(
#     filepath_or_buffer='/home/sayong/Project/AVCS/Car-counter-using-python-opencv/list_test_raw.csv',
#     header=None,
#     sep=',',
#     )
# df_test.columns = [l for i,l in sorted(feature_dict.items())] + ['class label']
# df_test.dropna(how="all", inplace=True) # to drop the empty line at file-end
# df_test.tail()
#
# data = df[[i for i in range(75)]].values
# trX = np.array(data.tolist(), dtype=np.float32)
#
# data_test = df_test[[i for i in range(75)]].values
# teX = np.array(data_test.tolist(), dtype=np.float32)
#
# answer = df['class label'].values
# answerList = [[0]*3 for x in range(len(answer))]
# for x in range(len(answerList)):
#     answerList[x][int(answer[x])] = 1
# trY = np.array(answerList)
#
# answer_test = df_test['class label'].values
# answerTestList = [[0]*3 for x in range(len(answer_test))]
# for x in range(len(answerTestList)):
#     answerTestList[x][int(answer_test[x])] = 1
# teY = np.array(answerTestList)
#
# preprocessing.scale(data, axis=0, with_mean=True, with_std=True, copy=False)
# preprocessing.scale(data_test, axis=0, with_mean=True, with_std=True, copy=False)
#
# listAcc = []
#
# # trX: training data, trY: answer data, teX: testing data trX: testing answer
# X = tf.placeholder("float", [None, 75])
# Y = tf.placeholder("float", [None, 3])
#
# for neuron in range(150, 151):
#     w_h = init_weights([75, neuron]) # create symbolic variables
#     w_o = init_weights([neuron, 3])
#
#     py_x = model(X, w_h, w_o)
#
#     cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute costs
#     #cost  = tf.reduce_mean(tf.pow(py_x-Y , 2))*0.5
#     #cost = -tf.reduce_sum(Y * tf.log(py_x))
#     #train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
#     train_op = tf.train.AdamOptimizer(0.001).minimize(cost)
#     predict_op = tf.argmax(py_x, 1)
#
#     sess = tf.Session()
#     init = tf.initialize_all_variables()
#     sess.run(init)
#
#     for i in range(5000):
#         for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
#             sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
#         print i, np.mean(np.argmax(teY, axis=1) ==
#                          sess.run(predict_op, feed_dict={X: teX, Y: teY}))
#     from pandas_confusion import ConfusionMatrix
#     cm = ConfusionMatrix(answer_test, sess.run(predict_op, feed_dict={X: teX, Y: teY}))
#     cmData = cm.to_array('a')
#     listAcc.append([cmData[0][0], cmData[1][1], cmData[2][2]])
# import ipdb;ipdb.set_trace()