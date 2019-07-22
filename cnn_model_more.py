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
    def __init__(self, image_size,channal, num_classes,learning_rate,max_gradient_norm):
        self.image_input = tf.placeholder(tf.float32, [None,image_size,image_size,channal])
        self.label_input = tf.placeholder(tf.int32, [None,1])
        self.keep_prob = tf.placeholder(tf.float32)
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.maximum(tf.train.exponential_decay(learning_rate, self.global_step,100,0.97,staircase=True),1e-7)
        self.m_plus = 0.9
        self.m_minus = 0.1
        self.lambda_val = 0.5
        label_onehot = tf.reshape(tf.one_hot(self.label_input,depth=num_classes,dtype = tf.float32),(-1,num_classes))


        #first conv2d 8*8*64
        self.W_first = weight_variable([5,5,16,64]) 
        self.b_first = bias_variable([64])
        first_out = tf.nn.relu(tf.nn.conv2d(self.image_input,self.W_first,strides=[1,1,1,1], padding='SAME')+self.b_first)
        first_out_pool = tf.nn.max_pool(first_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

        #second conv2d 4*4*128
        self.W_second = weight_variable([3,3,64,128]) 
        self.b_second = bias_variable([128])
        second_out = tf.nn.relu(tf.nn.conv2d(first_out_pool,self.W_second,strides=[1,1,1,1], padding='SAME')+self.b_second)
        second_out_pool = tf.nn.max_pool(second_out, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

    
        # 全连接
        self.W_fc1 = weight_variable([4*4*128, 512])
        self.b_fc1 = bias_variable([512])
        h_pool2_flat = tf.reshape(second_out_pool, [-1, 4*4*128])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, self.W_fc1) + self.b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)   

        # out
        self.W_fc2 = weight_variable([512, num_classes])
        self.b_fc2 = bias_variable([num_classes])
        self.output_logit = tf.add(tf.matmul(h_fc1_drop, self.W_fc2) ,self.b_fc2)

        #loss 1
        # self.out_predict = tf.nn.softmax(output_logit)
        # self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output_logit, labels = label_onehot))

        #loss 2
        self.out_predict = tf.nn.softmax(self.output_logit)
        max_l = tf.reshape(tf.square(tf.maximum(0., self.m_plus - self.output_logit)),(-1,num_classes))
        max_r = tf.reshape(tf.square(tf.maximum(0., self.output_logit - self.m_minus)),(-1,num_classes))
        L_c = label_onehot * max_l + self.lambda_val * (1 - label_onehot) * max_r
        self.cross_entropy = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        params = tf.trainable_variables()  # return variables which needed train
        gradients = tf.gradients(self.cross_entropy, params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)  # prevent gradient boom
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(clipped_gradients, params), global_step = self.global_step)   


class run_main():
    def __init__(self):
        sign_handle = import_data.dealsign()
        trainData,self.trainFlag,testData,self.testFlag = sign_handle.readFile()
        self.image_size = 15
        self.channal = 16
        self.num_classes = 5
        self.max_gradient_norm = 10
        self.learning_rate = 5e-4
        self.batch_size = 16
        self.trainData = self.two_dimension_graph(trainData)
        self.testData = self.two_dimension_graph(testData)
        self.cnn = model_cnn(self.image_size,self.channal, self.num_classes, self.learning_rate,self.max_gradient_norm)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        self.sess.run(tf.global_variables_initializer())

    def two_dimension_graph(self,feature):
        feature = feature.reshape((-1,self.image_size-1,self.channal))
        feature = feature.transpose(0,2,1)
        feature_graph = np.zeros((feature.shape[0],self.image_size,self.image_size,self.channal))
        for i in range(feature.shape[0]):
            for j in range(feature.shape[1]):
                single_use = np.hstack((feature[i,j,:],1)).reshape(-1,self.image_size)
                # single_graph = 0.5*np.sqrt(single_use.T*single_use)
                single_graph = self.sigmoid(0.5*np.sqrt(single_use.T*single_use))
                feature_graph[i,:,:,j] = single_graph
        feature_graph = feature_graph.reshape((feature.shape[0],-1))
        return feature_graph

    def sigmoid(self,single_feature):
        output_s = 1/(1+np.exp(-single_feature))
        return output_s

    def train(self,iteration):
        seed = 3
        m = self.trainData.shape[0]
        i = 0
        num_minibatches = int(m / self.batch_size)
        self.record_epoch_loss = np.zeros(iteration)
        self.record_loss = np.zeros(num_minibatches*iteration)
        for step in tqdm(range(iteration), total=iteration, ncols=70, leave=False, unit='b'):
            epoch_cost = 0
            seed += 1
            j = 0
            minibatches = self.random_mini_batches(self.batch_size,seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                minibatch_X = minibatch_X.reshape((-1,self.image_size,self.image_size,self.channal))
                output_feed = [self.cnn.train_op,self.cnn.cross_entropy,self.cnn.learning_rate]
                _, loss ,learning_rate_now = self.sess.run(output_feed, feed_dict={self.cnn.image_input:minibatch_X,self.cnn.label_input:minibatch_Y,self.cnn.keep_prob:0.8})
                epoch_cost = epoch_cost + loss / num_minibatches
                self.record_loss[i*num_minibatches+j] = loss
                j = j + 1
            if step % 1 == 0 or step == (iteration-1):
                print('step {}: learing_rate={:3.10f}\t loss = {:3.10f}\t '.format(step, learning_rate_now,epoch_cost))
            self.record_epoch_loss[i] = epoch_cost
            i = i + 1
        self.save()

    def save_epoch_loss(self,accuracy):
        m = self.record_epoch_loss.shape[0]
        with open('saver_cnn_best/saver_cnn_'+str(accuracy)+'/epoch_loss.txt','w') as f:
            for i in range(m):
                f.write(str(self.record_epoch_loss[i]))
                f.write('\n')

    def save_loss(self,accuracy):
        m = self.record_loss.shape[0]
        with open('saver_cnn_best/saver_cnn_'+str(accuracy)+'/loss.txt','w') as f:
            for i in range(m):
                f.write(str(self.record_loss[i]))
                f.write('\n')
    
    def save_best(self,accuracy):
        saver = tf.train.Saver(max_to_keep = 5)
        saver.save(self.sess,'saver_cnn_best/saver_cnn_'+str(accuracy)+'/muscle.ckpt')
        
    def predict(self):
        m = self.testData.shape[0]
        num = int(m/self.batch_size)
        correct = 0
        for i in range(num):
            datafortest = self.testData[i*self.batch_size:(i+1)*self.batch_size,:].reshape((-1,self.image_size,self.image_size,self.channal))
            answer = self.sess.run(self.cnn.out_predict, feed_dict={self.cnn.image_input:datafortest,self.cnn.keep_prob:1})
            prediction = np.argmax(answer,axis=1)
            correct += np.sum((prediction.reshape((-1,1)) == self.testFlag[i*self.batch_size:(i+1)*self.batch_size,:])+0)
        print(correct)
        accuracy = correct/(self.batch_size*num)
        print(accuracy)
        return accuracy

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
        return mini_batches

    def save(self):
        saver = tf.train.Saver(max_to_keep = 5)
        saver.save(self.sess,'saver_cnn/muscle.ckpt')

    def load(self):
        saver = tf.train.Saver()
        saver.restore(self.sess,'saver_cnn/muscle.ckpt')

if __name__ == "__main__":
    for i in range(5):
        print('the time:'+str(i+1))
        ram_better = run_main()
        ram_better.train(40) 
        accuracy = ram_better.predict()
        if accuracy >0.7:
            ram_better.save_best(accuracy)
            ram_better.save_loss(accuracy)
            ram_better.save_epoch_loss(accuracy)
        tf.reset_default_graph()