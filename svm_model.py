from sklearn import svm
import numpy as np
import import_data
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
        print(action_num)
        action_acc = action_num/action_batch
        print(action_acc)
        accuracy = np.sum(action_num)/self.testFlag.shape[0] 
        return accuracy

        # variousTag = np.unique(self.trainFlag)
        # rowacc = variousTag.shape[0]   # rowacc表示动作数目
        # accNumb = np.zeros((1, rowacc))
        # for i in range(self.testData.shape[0]):  # 判断测试集的分类
        #     k = np.matrix(self.testData[i, :])
        #     y = clf.predict(k)
        #     if y == self.testFlag[i]:
        #         accNumb[0, y-1] += 1
        # accNumb = accNumb / self.testData.shape[0]*rowacc
        # avg = np.sum(accNumb) / rowacc
        # std = np.sqrt(np.sum((accNumb - avg) ** 2)/rowacc)
        # print(avg,std)

if __name__ == "__main__":
    data_class = "person"
    # data_class = "people"
    sign_handle = import_data.DealSign()
    trainData,trainFlag,testData,testFlag = sign_handle.readFile(data_class)
    print('train and predict')
    svm_model = svmclassify(trainData,trainFlag,testData,testFlag)
    accuracy = svm_model.classify()
    print(accuracy)