import numpy as np
import tensorflow as tf
import math
import import_data
from tqdm import tqdm

class AnnModel():
    def __init__(self, input_size, num_classes,learning_rate,max_gradient_norm):
        W1 = tf.get_variable("W1", [224, 512], initializer=tf.contrib.layers.xavier_initializer())
        b1 = tf.get_variable("b1", [512], initializer=tf.zeros_initializer())
        W2 = tf.get_variable("W2", [512, 1024], initializer=tf.contrib.layers.xavier_initializer())
        b2 = tf.get_variable("b2", [1024], initializer=tf.zeros_initializer())
        W3 = tf.get_variable("W3", [1024, 128], initializer=tf.contrib.layers.xavier_initializer())
        b3 = tf.get_variable("b3", [128], initializer=tf.zeros_initializer())
        W4 = tf.get_variable("W4", [128, num_classes], initializer=tf.contrib.layers.xavier_initializer())
        b4 = tf.get_variable("b4", [num_classes], initializer=tf.zeros_initializer())
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = learning_rate
        self.learning_rate = tf.maximum(tf.train.exponential_decay(learning_rate, self.global_step,100,0.97,staircase=True),1e-8)
        self.input = tf.placeholder(tf.float32,[None,input_size])
        self.label = tf.placeholder(tf.int32,[None,1])
        self.m_plus = 0.9
        self.m_minus = 0.1
        self.lambda_val = 0.5
        label_onehot = tf.reshape(tf.one_hot(self.label,depth=num_classes,dtype = tf.float32),(-1,num_classes))

        # forward_propagation
        fc1 = tf.nn.relu(tf.matmul(self.input, W1) + b1)
        fc2 = tf.nn.relu(tf.matmul(fc1, W2) + b2)
        fc3 = tf.nn.relu(tf.matmul(fc2, W3) + b3)
        self.output_logit = tf.matmul(fc3, W4) + b4

        #loss1
        # self.out_predict = tf.nn.softmax(out_logit)
        # self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = out_logit, labels = label_onehot))

        #loss2
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
        self.trainData,self.trainFlag,self.testData,self.testFlag = sign_handle.readFile()
        self.image_size = 224
        self.num_classes = 5
        self.max_gradient_norm = 10
        self.learning_rate = 1e-5
        self.batch_size = 16
        self.Ann_model = AnnModel(self.image_size,  self.num_classes, self.learning_rate,self.max_gradient_norm)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        self.sess.run(tf.global_variables_initializer())

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
                output_feed = [self.Ann_model.train_op,self.Ann_model.cross_entropy,self.Ann_model.learning_rate]
                _, loss ,learning_rate_now = self.sess.run(output_feed, feed_dict={self.Ann_model.input:minibatch_X,self.Ann_model.label:minibatch_Y})
                epoch_cost = epoch_cost + loss / num_minibatches
                self.record_loss[i*num_minibatches+j] = loss
                j = j + 1
            if step % 1 == 0 or step == (iteration-1):
                print('step {}: learing_rate={:3.10f}\t loss = {:3.6f}\t '.format(step, learning_rate_now,epoch_cost))
            self.record_epoch_loss[i] = epoch_cost
            i = i + 1
        self.save()

    def save_epoch_loss(self,accuracy):
        m = self.record_epoch_loss.shape[0]
        with open('saver_bp_best/saver_bp_'+str(accuracy)+'/epoch_loss.txt','w') as f:
            for i in range(m):
                f.write(str(self.record_epoch_loss[i]))
                f.write('\n')

    def save_loss(self,accuracy):
        m = self.record_loss.shape[0]
        with open('saver_bp_best/saver_bp_'+str(accuracy)+'/loss.txt','w') as f:
            for i in range(m):
                f.write(str(self.record_loss[i]))
                f.write('\n')
    
    def save_best(self,accuracy):
        saver = tf.train.Saver(max_to_keep = 5)
        saver.save(self.sess,'saver_bp_best/saver_bp_'+str(accuracy)+'/muscle.ckpt')

    def predict(self):
        m = self.testData.shape[0]
        num = int(m/self.batch_size)
        correct = 0
        for i in range(num):
            datafortest = self.testData[i*self.batch_size:(i+1)*self.batch_size,:]
            answer = self.sess.run(self.Ann_model.out_predict, feed_dict={self.Ann_model.input: datafortest})
            prediction = np.argmax(answer,axis=1)
            correct += np.sum((prediction.reshape((-1,1)) == self.testFlag[i*self.batch_size:(i+1)*self.batch_size,:])+0)
        print(correct)
        accuracy = correct/(self.batch_size*num)
        print('test accuracy = {:3.6f}'.format(accuracy))
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
            mini_batch_X = shuffled_X[k * mini_batch_size: k *mini_batch_size + mini_batch_size,:]
            mini_batch_Y = shuffled_Y[k * mini_batch_size: k *mini_batch_size + mini_batch_size,:]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)
        return mini_batches
    
    def save(self):
        saver = tf.train.Saver(max_to_keep = 5)
        saver.save(self.sess,'saver_bp/muscle.ckpt')

    def load(self):
        saver = tf.train.Saver()
        saver.restore(self.sess,'saver_bp/muscle.ckpt')

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
