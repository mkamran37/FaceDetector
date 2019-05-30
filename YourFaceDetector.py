import cv2
import numpy as np
import sys
import os
import pickle 
import math
from nms import non_max_suppression
import json
from sklearn.feature_selection import SelectPercentile, f_classif

json_list = []
class Features:
    def __init__(self,x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def compute_feature(self, ii):
        a,b,c,d = self.return_vals(ii)
        return a + b - c - d

    def return_vals(self,ii):
        if self.height + self.y >= ii.shape[0] or self.x + self.width >= ii.shape[1]:
            return 0,0,0,0            
        a = ii[self.height + self.y][self.x + self.width]
        b = ii[self.y][self.x]
        c = ii[self.height + self.y][self.x]
        d = ii[self.y][self.x + self.width]
        return a,b,c,d

class CascadeClassifier:
    def __init__(self):
        self.layers = [10,15,20,30]
        self.clfs = []   
        
    def train_classifier(self, weights, features, X, y, training_data):
        for feature_num in self.layers:
            clf = ViolaJones(T=feature_num)
            clf.train(weights, features, X, y, training_data)
            self.clfs.append(clf)
        self.save('classifier')

    def classify(self, image):
        for clf in self.clfs:
            op,total = clf.classify(image)
            if op == 0:
                return 0,total
        return 1, total
          
    def save(self, filename):
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self,f)
    
    def load(self,filename):
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)

class ViolaJones:
    def __init__(self, T = 10):
        self.T = T
        self.alphas = []
        self.clfs = []
    
    def train(self, weights, features, X, y, training_data):
        for _ in range(self.T):
            weights = weights / len(weights)
            weak_classifiers = self.weak(X, y, features,weights)
            clf, error, accuracy = self.optimal_classifier(weak_classifiers, weights, training_data)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0/beta)
            self.alphas.append(alpha)
            self.clfs.append(clf)
    
    def get_initial_weights(self,y,weights):
        total_pos, total_neg = 0, 0
        for w, label in zip(weights, y):
            if label == 1:
                total_pos += w
            else:
                total_neg += w
        return total_pos, total_neg    
    
    def weak(self, X, y, features, weights):
        classifiers = []
        total_pos, total_neg = self.get_initial_weights(y,weights)
        for index, feature in enumerate(X):
            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])
            pos_seen, neg_seen = 0, 0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            for w, f, label in applied_feature:
                error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
                if error < min_error:
                    min_error = error
                    best_feature = features[index]
                    best_threshold = f
                    if pos_seen > neg_seen:
                        best_polarity = 1
                    else:
                        best_polarity = -1
                if label == 1:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w
            clf = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
            classifiers.append(clf)
        return classifiers        
    
    def optimal_classifier(self, classifiers, weights, training_data):
        best_clf, best_error, best_accuracy = None, float('inf'), None
        for clf in classifiers:
            error, accuracy = 0, []
            for data, w in zip(training_data, weights):
                correctness = abs(clf.classify(data[0]) - data[1])
                accuracy.append(correctness)
                error += w * correctness
            best_error = error / len(accuracy)
        best_accuracy = accuracy/len(classifiers)
        return best_clf, best_error, best_accuracy
    
    def classify(self, image):
        total = 0
        ii = findIntegralImage(image)
        for alpha, clf in zip(self.alphas, self.clfs):
            total += alpha * clf.classify(ii)
        if total >= 0.58 * sum(self.alphas):
            return 1,total
        else:
            return 0,0
    
class WeakClassifier:
    def __init__(self, positive_regions, negative_regions, threshold, polarity):
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
        self.threshold = threshold
        self.polarity = polarity

    def classify(self, x):
        intensity = self.fval(x)
        if self.polarity * intensity < self.polarity * self.threshold:
            return 1
        else:
            return 0
    
    def fval(self,ii):
        a = sum([pos.compute_feature(ii) for pos in self.positive_regions])
        b = sum([neg.compute_feature(ii) for neg in self.negative_regions])
        return a - b

def init(path, clsf):
    images = []
    if clsf == "positive":
        classification = 1
    else:
        classification = 0
    names = []
    for img in os.listdir(path):
        if img.endswith('jpg'):
            names.append(img)
    for img in names:
        image = cv2.imread(path+"/"+img, cv2.IMREAD_GRAYSCALE)
        images.append((image,classification))
        if len(images) >= 4000:
            break
    return images

def findIntegralImage(image):
    ii = np.zeros(image.shape)
    s = np.zeros(image.shape)
    for y in range(len(image)):
        for x in range(len(image[y])):
            if y-1 >= 0:
                s[y][x] = s[y-1][x] + image[y][x]
            else:
                image[y][x]
            if x-1 >= 0:
                ii[y][x] = ii[y][x-1]+s[y][x]
            else:
                s[y][x]
    return ii

def get_y(pos, neg):
    y = []
    for _ in range(len(pos)):
        y.append(1)
    for _ in range(len(neg)):
        y.append(0)
    return y

def build_features(height, width):
    features = []
    features = features + type1_ftrs(width,height)
    features = features + type2_ftrs(width,height)
    features = features + type3_ftrs(width,height)
    return features

def type1_ftrs(width, height):
        features = []
        for x in range(1,width):
            w = 0
            h = 0
            for y in range(1, height):
                if (x + w) * 2 < width:
                    white_region = Features(x, y, x + w, y)
                    black_region = Features(x+w, y, x + w, y)
                    features.append(([white_region], [black_region]))
                if y + h < height and x + w < width:
                    white_region = Features(x,y,w, y+h)
                    black_region = Features(x+w, y,w, y+h)
                    features.append(([white_region], [black_region]))
                w+=1
                h+=1
        return features

def type2_ftrs(width, height):
        features = []
        for y in range(1, height):
            w = 0
            h = 0
            for x in range(1,width):
                if x + w < width and y + h < height:
                    white_region = Features(x, y, x + w, y + h)
                    black_region = Features(x, y + h, (x + w), y + h)
                    features.append(([white_region], [black_region]))
                w+=1
                h+=1
        return features

def type3_ftrs(width, height):
        features = []
        for x in range(1,width):
            w = 0
            h = 0
            for y in range(1, height):
                if (x + w) * 2 < width:
                    white_region = Features(x, y, x + w, y)
                    black_region = Features(x+w, y, x + w, y)
                    white_region2 = Features((x+w)*2, y, x+w, y)
                    features.append(([white_region], [black_region, white_region2]))
                if (y + h) < height and (x+w)*2 < width:
                    white_region = Features(x,y, x, y+h)
                    black_region = Features(x+w, y, x, y+h)
                    white_region2 = Features((x+w)*2, y, x, y+h)
                    features.append(([white_region], [black_region, white_region2]))
                w+=1
                h+=1
        return features

def get_X(features, training_data):
        X = np.zeros((len(features), len(training_data)))
        i = 0
        for negative_regions, positive_regions in features:
            X[i] = list(map(lambda data: findIntensity(data[0], positive_regions, negative_regions), training_data))
            i += 1
        return X

def findIntensity(ii, positive_regions, negative_regions):
    a = sum([pos.compute_feature(ii) for pos in positive_regions])
    b = sum([neg.compute_feature(ii) for neg in negative_regions])
    return a - b

def faceOrNot(name, image, clf):
    h = w = 100
    results = []
    while h < image.shape[0] and w < image.shape[1]:
        for i in range(0, image.shape[0] - h, 10):
            for j in range(0, image.shape[1] - w, 10):
                b = cv2.resize(image[i:i+h, j:j+w], (19,19))
                op, total = clf.classify(b)
                if op == 1:
                    b = (i,j,i+h,j+w)
                    results.append((total,b))
        h = int(h*1.25)
        w = int(w*1.25)
    results.sort(key = lambda tup: tup[0], reverse = True)
    results = np.array(results)
    result = []
    i = 0
    for x in results:
        result.append(x[1])
    pick = non_max_suppression(np.array(result), 0.1)
    pick = non_max_suppression(np.array(pick), 0.02)
    for (sx,sy,ex,ey) in pick:
        ex = int(ex)
        ey = int(ey)
        sx = int(sx)
        sy = int(sy)
        elem = {
            "iname": name,
            "bbox": [sy,sx,ey,ex]
        }
        json_list.append(elem)

def test_init(path):
    images = []
    names = []
    for img in os.listdir(path):
        if img.endswith('jpg'):
            names.append(img)
            image = cv2.imread(path+"/"+img)
            images.append(image)
    return names,images

def train():
    pospath = sys.argv[1]
    negpath = sys.argv[2]
    posimages = init(pospath, "positive")
    negimages = init(negpath, "negative")
    trainingimages = posimages+negimages
    clsf = CascadeClassifier()
    features = build_features(19,19)
    pos, neg = [], []
    for ex in trainingimages: 
        if ex[1] == 1:
            pos.append(ex)
        else:
            neg.append(ex)
    training_data = []
    weights = np.zeros(len(trainingimages))
    print("Creating integral images")
    for x in range(len(trainingimages)):
        training_data.append((findIntegralImage(trainingimages[x][0]), trainingimages[x][1]))
        if trainingimages[x][1] == 1:
            weights[x] = 1.0 / len(pos)
        else:
            weights[x] = 1.0 / len(neg)
    print("Getting the value of X")
    X = get_X(features, training_data)
    print("Getting the value of y")
    y = get_y(posimages,negimages)
    print("Starting the training")
    clsf = CascadeClassifier()
    clsf.train_classifier(weights, features, X, y, training_data)

def main():
    clf = CascadeClassifier()
    clsf = clf.load("classifier")
    names, color = test_init(sys.argv[1])
    for name, img in zip(names, color):
        bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faceOrNot(name, bw, clsf)
    with open("results.json", 'w') as f:
        json.dump(json_list,f,indent=1)
    
if __name__ == '__main__':
    main()