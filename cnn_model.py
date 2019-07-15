import tensorflow as tf
import datetime
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import import_data
import math
from tqdm import tqdm

def weight_variable(shape):#权重正态分布初始化
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
    
# tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差。
def bias_variable(shape):#偏置量初始化
    initial=tf.constant(0.0,shape=shape)#value=0.1,shape是生成的维度
    return tf.Variable(initial)

class model_cnn(object):
    def __init__(self, image_size, num_classes,learning_rate,max_gradient_norm):
        self.image_input = tf.placeholder(tf.float32, [None,image_size,image_size,1])
        self.label_input = tf.placeholder(tf.int32, [None,1])
        self.keep_prob = tf.placeholder(tf.float32)
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.maximum(tf.train.exponential_decay(learning_rate, self.global_step,200,0.97,staircase=True),1e-7)
        #first conv2d 53*53*32
        self.W_first = weight_variable([15,15,1,32]) 
        self.b_first = bias_variable([32])
        label_onehot = tf.reshape(tf.one_hot(self.label_input,depth=num_classes,dtype = tf.float32),(-1,num_classes))
        first_out = tf.nn.relu(tf.nn.conv2d(self.image_input,self.W_first,strides=[1,2,2,1], padding='VALID')+self.b_first)
        first_out_pool = tf.nn.max_pool(first_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        #second conv2d 12*12*64
        self.W_second = weight_variable([7,7,32,64]) 
        self.b_second = bias_variable([64])
        second_out = tf.nn.relu(tf.nn.conv2d(first_out_pool,self.W_second,strides=[1,2,2,1], padding='VALID')+self.b_second)
        second_out_pool = tf.nn.max_pool(second_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        #third conv2d 
        self.W_third = weight_variable([3,3,64,128]) 
        self.b_third = bias_variable([128])
        third_out = tf.nn.relu(tf.nn.conv2d(second_out_pool,self.W_third,strides=[1,1,1,1], padding='SAME')+self.b_third)
        third_out_pool = tf.nn.max_pool(third_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        # 全连接
        self.W_fc1 = weight_variable([6*6*128, 1024])
        self.b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(third_out_pool, [-1, 6*6*128])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)
        
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)   

#         # second fc2
#         self.W_fc2 = weight_variable([1024, 128])
#         self.b_fc2 = bias_variable([128])
#         h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, self.W_fc2) + self.b_fc2)

        # out
        self.W_fc3 = weight_variable([1024, num_classes])
        self.b_fc3 = bias_variable([num_classes])
        output_logit = tf.add(tf.matmul(h_fc1_drop, self.W_fc3) ,self.b_fc3)

        self.out_predict = tf.nn.softmax(output_logit)
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output_logit, labels = label_onehot))
        params = tf.trainable_variables()  # return variables which needed train
        gradients = tf.gradients(self.cross_entropy, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)  # prevent gradient boom
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(clipped_gradients, params), global_step = self.global_step)   


class run_main():
    def __init__(self):
        sign_handle = import_data.dealsign()
        trainData,self.trainFlag,testData,self.testFlag = sign_handle.readFile()
        self.trainData = self.two_dimension_graph(trainData)
        self.testData = self.two_dimension_graph(testData)
        self.image_size = 225
        self.num_classes = 5
        self.max_gradient_norm = 10
        self.learning_rate = 5e-4
        self.batch_size = 32
        self.cnn = model_cnn(self.image_size,  self.num_classes, self.learning_rate,self.max_gradient_norm)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        self.sess.run(tf.global_variables_initializer())

    def two_dimension_graph(self,feature):
        one_feature = np.ones((feature.shape[0],1))
        feature = np.hstack((feature,one_feature))
        feature_graph = []
        for i in range(feature.shape[0]):
            single_use = feature[i].reshape(-1,len(feature[i]))
            # single_graph = self.sigmoid(0.5*np.sqrt(single_use.T*single_use))
            single_graph = 0.5*np.sqrt(single_use.T*single_use)
#             single_graph = single_use.T*single_use
            single_graph = single_graph.reshape(-1,single_graph.shape[0]*single_graph.shape[1])
            feature_graph.append(single_graph)
        feature_graph = np.squeeze(np.array(feature_graph))
        return feature_graph

    def train(self,iteration):
        seed = 3
        m = self.trainData.shape[0]
        for step in tqdm(range(iteration), total=iteration, ncols=70, leave=False, unit='b'):
            epoch_cost = 0
            seed += 1
            num_minibatches = int(m / self.batch_size)
            minibatches = self.random_mini_batches(self.batch_size,seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                minibatch_X = minibatch_X.reshape((-1,self.image_size,self.image_size,1))
                output_feed = [self.cnn.train_op,self.cnn.cross_entropy,self.cnn.learning_rate]
                _, loss ,learning_rate_now = self.sess.run(output_feed, feed_dict={self.cnn.image_input:minibatch_X,self.cnn.label_input:minibatch_Y,self.cnn.keep_prob:0.8})
                epoch_cost = epoch_cost + loss / num_minibatches
            if step % 10 == 0 or step == (iteration-1):
                print('step {}: learing_rate={:3.10f}\t loss = {:3.10f}\t '.format(step, learning_rate_now,epoch_cost))
        self.save()

    def predict(self):
        m = self.testData.shape[0]
        num = int(m/self.batch_size)
        correct = 0
        for i in range(num):
            datafortest = self.testData[i*self.batch_size:(i+1)*self.batch_size,:].reshape((-1,self.image_size,self.image_size,1))
            answer = self.sess.run(self.cnn.out_predict, feed_dict={self.cnn.image_input:datafortest,self.cnn.keep_prob:1})
            prediction = np.argmax(answer,axis=1)
            correct += np.sum((prediction.reshape((-1,1)) == self.testFlag[i*self.batch_size:(i+1)*self.batch_size,:])+0)
        datafortest = self.testData[num*self.batch_size:,:].reshape((-1,self.image_size,self.image_size,1))
        answer = self.sess.run(self.cnn.out_predict, feed_dict={self.cnn.image_input:datafortest,self.cnn.keep_prob:1})
        prediction = np.argmax(answer,axis=1)
        correct += np.sum((prediction.reshape((-1,1)) == self.testFlag[num*self.batch_size:,:])+0)
        accuracy = correct/m
        print(accuracy)

    def random_mini_batches(self, mini_batch_size=64, seed=0):
        X = self.trainData
        Y = self.trainFlag
        m = X.shape[0]                  # number of training examples
        mini_batches = []
        np.random.seed(seed)
        permutation = list(np.random.permutation(m))  # 随机排序0-(m-1)
        shuffled_X = X[permutation, :]
        shuffled_Y = Y[permutation, :].reshape((m,Y.shape[1]))
        mini_batch_X = shuffled_X[0 * mini_batch_size: 0 *mini_batch_size + mini_batch_size,:]
        mini_batch_Y = shuffled_Y[0 * mini_batch_size: 0 *mini_batch_size + mini_batch_size,:]
        num_complete_minibatches = math.floor(m/mini_batch_size)
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[k * mini_batch_size: k *mini_batch_size + mini_batch_size, :]
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k *mini_batch_size + mini_batch_size,:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size: m,:]
            mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size: m,:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        return mini_batches

    def save(self):
        saver = tf.train.Saver(max_to_keep = 5)
        saver.save(self.sess,'saver_cnn/muscle.ckpt')

    def load(self):
        saver = tf.train.Saver()
        saver.restore(self.sess,'saver_cnn/muscle.ckpt')

if __name__ == "__main__":
    ram_better = run_main()
    ram_better.train(100)
    ram_better.predict()