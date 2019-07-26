import numpy as np
# import tensorflow as tf

# a = tf.placeholder(tf.float32,[1,3,2])
# b = tf.placeholder(tf.float32,[4,3,2])
# c = tf.multiply(a,b)
# a1 = np.arange(1,7,1).reshape([1,3,2])
# a2 = np.arange(1,25,1).reshape([4,3,2])
# print(a1)
# print(a2)
# with tf.Session() as sess:
#     answer = sess.run(c,feed_dict={a:a1,b:a2})
#     print(answer)
a = np.random.randint(4,9,(5))

print(a)
b= np.sum(a>=6)
print(b)