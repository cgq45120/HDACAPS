import numpy as np
import tensorflow as tf
import math
import import_data
import bp_model
import caps_att_15_more
import cnn_model
import caps_noatt

if __name__ == "__main__":
    for i in range(70):
        print('caps the time:'+str(i+1))
        ram_better = caps_att_15_more.run_main()
        ram_better.train(40) 
        accuracy = ram_better.predict()
        if accuracy >0.75:
            ram_better.save_best(accuracy)
        tf.reset_default_graph()
    for i in range(50):
        print('caps noatt the time:'+str(i+1))
        ram_better = caps_noatt.run_main()
        ram_better.train(40) 
        accuracy = ram_better.predict()
        if accuracy >0.7:
            ram_better.save_best(accuracy)
        tf.reset_default_graph()
    for i in range(50):
        print('cnn the time:'+str(i+1))
        ram_better = cnn_model.run_main()
        ram_better.train(40) 
        accuracy = ram_better.predict()
        if accuracy >0.7:
            ram_better.save_best(accuracy)
        tf.reset_default_graph()
    for i in range(50):
        print('bp the time:'+str(i+1))
        ram_better = bp_model.run_main()
        ram_better.train(40) 
        accuracy = ram_better.predict()
        if accuracy >0.7:
            ram_better.save_best(accuracy)
        tf.reset_default_graph()