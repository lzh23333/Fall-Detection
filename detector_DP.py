import numpy as np
import pickle

class detector:
    def __init__(self):
        self.featureList = []
        self.winLen = 5

        self.clf = []
        cats = ['back', 'front', 'side', 'endUpSit']
        for c in cats:
            self.clf.append(pickle.load(open(c + '.model', 'rb')))
        self.f4 = []
        for i in range(4):
            inf = 999
            f = np.zeros((1, 4))
            f[0, 0] = 0
            f[0, 1] = -inf
            f[0, 2] = -inf
            f[0, 3] = -inf
            self.f4.append(f)

    def check(self):
        if len(self.featureList) < 5:
            return "None"

        outStr = ''

        for iclf in range(4):
            pro = np.log(np.reshape(self.clf[iclf].predict_proba(np.reshape(np.array(self.featureList[-5:]), (1, -1))), -1))

            self.f4[iclf] = np.concatenate((self.f4[iclf], np.zeros(1, 4)))
            f = self.f4[iclf]
            i = f.shape[0]
            for j in range(4):
                if j == 0:
                    f[i, j] = f[i - 1, j] + pro[i][j]
                else:
                    f[i, j] = max(f[i - 1, j], f[i - 1, j - 1]) + pro[i][j]
            outStr += str(f[-1, 3]) + '---'
        return outStr


    def input(self, skeleton):
        self.featureList.append(skeleton)



if __name__ == '__main__':

    d = detector()
