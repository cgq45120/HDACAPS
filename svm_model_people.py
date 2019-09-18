from sklearn import svm
import numpy as np
import import_data_people
class svmclassify():
    def __init__(self,trainData,trainFlag,testData,testFlag):
        self.trainData = trainData
        self.trainFlag = trainFlag
        self.testData = testData
        self.testFlag = testFlag

    def classify(self):
        clf = svm.SVC(decision_function_shape='ovo', gamma='auto')
        clf.fit(self.trainData,self.trainFlag.reshape(-1,))
        predictTag = clf.predict(self.testData)
        predict_action = (predictTag.reshape(-1,1) == self.testFlag)+0
        action_batch = int(self.testFlag.shape[0]/5)
        action_num = np.zeros(5)
        for i in range(5):
            action_num[i] = np.sum(predict_action[i*action_batch:(i+1)*action_batch])
        print(np.sum(predict_action[182*32:5*action_batch]))
        print(action_num)
        action_acc = action_num/action_batch
        print(action_acc)
        accuracy = np.sum(action_num)/self.testFlag.shape[0] 
        return accuracy

if __name__ == "__main__":
    sign_handle = import_data_people.dealsign()
    trainData,trainFlag,testData,testFlag = sign_handle.readFile()
    print('train and predict')
    svm_model = svmclassify(trainData,trainFlag,testData,testFlag)
    accuracy = svm_model.classify()
    print(accuracy)