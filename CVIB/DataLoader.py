import numpy as np
import os
def load_data(data_dir):
    train_file = os.path.join(data_dir,"ydata-ymusic-rating-study-v1_0-train.txt")
    test_file = os.path.join(data_dir,"ydata-ymusic-rating-study-v1_0-test.txt")
    x_train = []
    # <user_id> <song id> <rating>
    with open(train_file, "r") as f:
        for line in f:
            x_train.append(line.strip().split())
    x_train = np.array(x_train).astype(int)
    x_test = []
    # <user_id> <song id> <rating>
    with open(test_file, "r") as f:
        for line in f:
            x_test.append(line.strip().split())
    x_test = np.array(x_test).astype(int)
    print("===>Load from yahoo data set<===")
    print("[train] num data:", x_train.shape[0])
    print("[test]  num data:", x_test.shape[0])
    return x_train[:,:-1], x_train[:,-1], x_test[:, :-1], x_test[:,-1]