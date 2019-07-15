import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
import import_data
import math
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg' #高画质图

class CapsLayer(object):
    def __init__(self, epsilon,iter_routing,num_outputs, vec_len, with_routing=True, layer_type='FC'):
        self.epsilon = epsilon
        self.iter_routing = iter_routing
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type

    def __call__(self, input, kernel_size=None, stride=None):
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride
            if not self.with_routing:
                capsules = tf.contrib.layers.conv2d(input, self.num_outputs * self.vec_len,self.kernel_size, self.stride, padding="VALID")
                capsules = tf.reshape(capsules, (-1, capsules.shape[1].value*capsules.shape[2].value*self.num_outputs, self.vec_len, 1))
                capsules = self.squash(capsules)
                return (capsules)
        if self.layer_type == 'FC':
            if self.with_routing:
                self.input = tf.reshape(input, shape=(-1, input.shape[1].value, 1, input.shape[-2].value, 1))
                with tf.variable_scope('routing'):
                    b_IJ = tf.constant(np.zeros([1, input.shape[1].value, self.num_outputs, 1, 1], dtype=np.float32))
                    capsules = self.routing(self.input, b_IJ)
                    capsules = tf.squeeze(capsules, axis=1)
            return(capsules)

    def routing(self,input, b_IJ):
        input_shape = input.get_shape()
        W = tf.get_variable('Weight', shape=(1,input_shape[1],self.num_outputs*self.vec_len,input_shape[3],input_shape[4]),dtype=tf.float32, initializer=tf.random_normal_initializer(stddev=0.01))
        # W = tf.get_variable('Weight', shape=(1, input_shape[1], input_shape[2], input_shape[3], self.vec_len), dtype=tf.float32,initializer=tf.random_normal_initializer(stddev=0.01))
        input = tf.tile(input, [1, 1,self.num_outputs* self.vec_len, 1, 1])
        u_hat = tf.reduce_sum(W * input, axis=3, keepdims=True)
        u_hat = tf.reshape(u_hat, shape=[-1, input_shape[1], self.num_outputs, self.vec_len, 1])
        u_hat_stopped = tf.stop_gradient(u_hat,name='stop_gradient')
        # u_hat = tf.matmul(W, input, transpose_a=True) # W在相乘前转置
        for r_iter in range(self.iter_routing):
            with tf.variable_scope('iter_' + str(r_iter)):
                c_IJ = tf.nn.softmax(b_IJ, axis=2)
                if r_iter == self.iter_routing-1:
                    s_J = tf.multiply(c_IJ, u_hat)
                    s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                    v_J = self.squash(s_J)
                elif r_iter <self.iter_routing-1:
                    s_J = tf.multiply(c_IJ, u_hat_stopped)
                    s_J = tf.reduce_sum(s_J, axis=1, keep_dims=True)
                    v_J = self.squash(s_J)
                    v_J_tiled = tf.tile(v_J, [1, input_shape[1], 1, 1, 1])
                    u_produce_v = tf.matmul(u_hat_stopped, v_J_tiled, transpose_a=True)
                    b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
        return(v_J)

    def squash(self,vector):
        vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keep_dims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + self.epsilon)
        vec_squashed = scalar_factor * vector  
        return(vec_squashed)

class spatial_attention():
    def __init__(self,input_shape):
        self.w_tanh = self.weight_variable((1,input_shape[1].value,input_shape[2].value,input_shape[3].value))
        self.w_sigmoid = self.weight_variable((1,1,1,input_shape[3].value))
        # self.b_tanh = self.bias_variable((1,input_shape[1].value,input_shape[2].value,input_shape[3].value))
        self.b_tanh = self.bias_variable((1,input_shape[1].value,input_shape[2].value,1))
        self.b_sigmoid = self.bias_variable([1])
        
    def __call__(self,input):
        fc_first = tf.nn.tanh(tf.multiply(input,self.w_tanh) + self.b_tanh) 
        fc_second = tf.reduce_sum(tf.multiply(fc_first,self.w_sigmoid),axis=3,keep_dims=True) + self.b_sigmoid
        self.attention = tf.nn.sigmoid(fc_second)
        output = tf.multiply(input,self.attention)
        return output

    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape = shape, stddev = 0.01)  # define weight
        return tf.Variable(initial)

    def bias_variable(self,shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

class Capsnet():
    def __init__(self,image_size,num_classes,lambda_val,m_plus,m_minus,epsilon,iter_routing,num_outputs_decode,num_dims_decode,learning_rate):
        self.image_size = image_size
        self.lambda_val = lambda_val
        self.m_plus = m_plus
        self.m_minus = m_minus
        self.epsilon = epsilon
        self.global_step = tf.Variable(0, name='global_step', trainable = False)
        # self.learning_rate = tf.maximum(tf.train.exponential_decay(learning_rate, self.global_step,1000,0.97,staircase=True),1e-5)
        self.iter_routing = iter_routing
        self.max_gradient_norm = 10
        self.num_outputs_layer_conv2d1 = 64
        self.num_outputs_layer_conv2d2 = 128
        self.num_outputs_layer_conv2d3 = 256
        self.num_outputs_layer_PrimaryCaps = 32
        self.num_dims_layer_PrimaryCaps = 8
        self.num_outputs_decode = num_outputs_decode
        self.num_dims_decode = num_dims_decode
        self.image = tf.placeholder(tf.float32,[None,image_size,image_size,1])
        self.label = tf.placeholder(tf.int64,[None,1])
        label_onehot = tf.one_hot(self.label,depth=num_classes,axis=1,dtype = tf.float32)
        # transform to 
        with tf.variable_scope('Conv1_layer'):
            conv1 = tf.contrib.layers.conv2d(self.image, num_outputs = self.num_outputs_layer_conv2d1,kernel_size = 13, stride = 2,padding='VALID')
        with tf.variable_scope('Conv2_layer'):
            conv2 = tf.contrib.layers.conv2d(conv1, num_outputs = self.num_outputs_layer_conv2d2,kernel_size = 11, stride = 2,padding='VALID')
        with tf.variable_scope('Conv3_layer'):
            conv3 = tf.contrib.layers.conv2d(conv2, num_outputs = self.num_outputs_layer_conv2d3,kernel_size = 11, stride = 2,padding='VALID')
        attention_shape = conv3.get_shape()
        with tf.variable_scope('soft_attention'):
            spatialAtt = spatial_attention(attention_shape)
            attention1 = spatialAtt(conv3)
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(self.epsilon,self.iter_routing,num_outputs=self.num_outputs_layer_PrimaryCaps, vec_len=self.num_dims_layer_PrimaryCaps, with_routing=False, layer_type='CONV')
            caps1 = primaryCaps(attention1, kernel_size=9, stride=2)
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(self.epsilon,self.iter_routing, num_outputs = self.num_outputs_decode, vec_len=self.num_dims_decode, with_routing=True, layer_type='FC')
            self.caps2 = digitCaps(caps1)
        with tf.variable_scope('Masking'):
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keep_dims=True) + self.epsilon)
        
        # loss function1
        # label_onehot_out = tf.reshape(label_onehot,(-1,num_classes))
        # output_logit = tf.reshape(self.v_length,(-1,num_classes))
        # self.total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = output_logit, labels = label_onehot_out))
        # params = tf.trainable_variables()  # return variables which needed train
        # gradients = tf.gradients(self.total_loss, params)
        # clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)  # prevent gradient boom
        # self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(clipped_gradients, params), global_step = self.global_step) 

        # loss function2
        max_l = tf.square(tf.maximum(0., self.m_plus - self.v_length))
        max_r = tf.square(tf.maximum(0., self.v_length - self.m_minus))
        max_l_out = tf.reshape(max_l, shape=(-1,self.num_outputs_decode))
        max_r_out = tf.reshape(max_r, shape=(-1,self.num_outputs_decode))
        label_onehot_out = tf.squeeze(label_onehot,axis=2)
        L_c = label_onehot_out * max_l_out + self.lambda_val * (1 - label_onehot_out) * max_r_out
        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))
        # self.weight_err = tf.sqrt(tf.reduce_sum(tf.square(spatialAtt.attention))) # L2正则项
        # self.total_loss = self.margin_loss + (5e-10)*self.weight_err
        self.total_loss = self.margin_loss

        # Adam optimization
        self.train_op = tf.train.AdamOptimizer().minimize(self.total_loss, global_step = self.global_step)


class run_main():
    def __init__(self):
        sign_handle = import_data.dealsign()
        trainData,self.trainFlag,testData,self.testFlag = sign_handle.readFile()
        self.trainData = self.two_dimension_graph(trainData)
        self.testData = self.two_dimension_graph(testData)
        self.image_size = 225
        self.num_classes = 5
        self.batch_size = 4
        self.lambda_val = 0.5
        self.m_plus = 0.9
        self.m_minus = 0.1
        self.epsilon = 1e-9
        self.iter_routing = 3
        self.num_outputs_decode = 5
        self.num_dims_decode = 16
        self.learning_rate = 1e-3
        self.capsnet_model = Capsnet(self.image_size,self.num_classes,self.lambda_val,self.m_plus,self.m_minus,self.epsilon,self.iter_routing,self.num_outputs_decode,self.num_dims_decode,self.learning_rate)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
        self.sess.run(tf.global_variables_initializer())
    
    def two_dimension_graph(self,feature):
        one_feature = np.ones((feature.shape[0],1))
        feature = np.hstack((feature,one_feature))
        feature_graph = []
        for i in range(feature.shape[0]):
            single_use = feature[i].reshape(-1,len(feature[i]))
            # single_graph = self.sigmoid(0.5*np.sqrt(single_use.T*single_use))
            single_graph = np.sqrt(single_use.T*single_use)
            # single_graph = single_use.T*single_use
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
                minibatch_Y = minibatch_Y.reshape((-1,1))
                output_feed = [self.capsnet_model.train_op, self.capsnet_model.total_loss,self.capsnet_model.v_length]
                _, loss ,v_length= self.sess.run(output_feed, feed_dict={self.capsnet_model.image: minibatch_X, self.capsnet_model.label: minibatch_Y})
                epoch_cost = epoch_cost + loss / num_minibatches
            if step % 1 == 0 or step == iteration-1:
                print('step {}: loss = {:3.10f}'.format(step,epoch_cost))
                print(v_length.reshape((-1,5)))
        self.save()

    def predict(self):
        m = self.testData.shape[0]
        num = int(m/self.batch_size)
        correct = 0
        for i in range(num):
            datafortest = self.testData[i*self.batch_size:(i+1)*self.batch_size,:].reshape((-1,self.image_size,self.image_size,1))
            answer = self.sess.run(self.capsnet_model.v_length,feed_dict={self.capsnet_model.image: datafortest})
            prediction = np.argmax(np.squeeze(answer),axis=1)
            correct += np.sum((prediction.reshape((-1,1)) == self.testFlag[i*self.batch_size:(i+1)*self.batch_size,:])+0)
            print(correct)
        datafortest = self.testData[num*self.batch_size:,:].reshape((-1,self.image_size,self.image_size,1))
        answer = self.sess.run(self.capsnet_model.v_length,feed_dict={self.capsnet_model.image: datafortest})
        prediction = np.argmax(np.squeeze(answer),axis=1)
        correct += np.sum((prediction.reshape((-1,1)) == self.testFlag[num*self.batch_size:,:])+0)
        print(correct)
        accuracy = correct/m
        print('test accuracy = {:3.6f}'.format(accuracy))

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
        saver.save(self.sess,'saver_caps/muscle.ckpt')

    def load(self):
        saver = tf.train.Saver()
        saver.restore(self.sess,'saver_caps/muscle.ckpt')

if __name__ == "__main__":
    ram_better = run_main()
    ram_better.train(10) 
    ram_better.predict()