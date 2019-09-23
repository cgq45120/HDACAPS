import numpy as np
import tensorflow as tf
import math
import import_data
import bp_model
import TDACAPS
import caps_cbam
import caps_sae
import caps_se
import caps_noatt
import svm_model
import cnn_feature_no2d
import cnn_model
import cnn_no2d


if __name__ == "__main__":
    data_class = "person"
    # data_class = "people"
    iteration = 40 if data_class == "person" else 30
    for i in range(10):
        print('TDACAPS the time:'+str(i+1))
        TDACAPS = TDACAPS.RunMain(data_class)
        TDACAPS.train(iteration) 
        accuracy = TDACAPS.predict()
        TDACAPS.save_best(accuracy)
        tf.reset_default_graph()

    for i in range(10):
        print('caps_cbam_model the time:'+str(i+1))
        caps_se_model = caps_cbam.RunMain(data_class)
        caps_se_model.train(iteration) 
        accuracy = caps_se_model.predict()
        caps_se_model.save_best(accuracy)
        tf.reset_default_graph()

    for i in range(10):
        print('caps_sae_model the time:'+str(i+1))
        caps_sae_model = caps_sae.RunMain(data_class)
        caps_sae_model.train(iteration) 
        accuracy = caps_sae_model.predict()
        caps_sae_model.save_best(accuracy)
        tf.reset_default_graph()

    for i in range(10):
        print('caps_se_model the time:'+str(i+1))
        caps_se_model = caps_se_model.RunMain(data_class)
        caps_se_model.train(iteration) 
        accuracy = caps_se_model.predict()
        caps_se_model.save_best(accuracy)
        tf.reset_default_graph()
