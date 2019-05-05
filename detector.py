import numpy as np
import pickle

class detector:
    def __init__(self):
        self.featureList = []
        self.winLen = 5
        self.clf = pickle.load(open('model.pickle', 'rb'))

    def check(self):
        if len(self.featureList) < self.winLen: return False
        x = np.array(self.featureList[-5:]).reshape((1, 5 * 24))
        return self.clf.predict(x)[0]

    def input(self, skeleton):
        indices = [4, 8, 5, 9, 6, 10, 12, 16, 13, 17, 14, 18]
        tmp = []
        for i in indices:
            tmp.append(skeleton[i][0])
        for i in indices:
            tmp.append(skeleton[i][1])
        self.featureList.append(tmp)


if __name__ == '__main__':

    d = detector()

    s = np.zeros((25, 3))
    d.input(s)
    d.input(s)
    d.input(s)
    d.input(s)
    d.input(s)

    print(d.check())
