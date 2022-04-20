import numpy as np
import torch
from sklearn.metrics import roc_auc_score
np.random.seed(2020)
torch.manual_seed(2020)
import pdb

from DataLoader import load_data
from matrix_factorization import MF_CVIB
from utils import gini_index, ndcg_func, get_user_wise_ctr, binarize, shuffle
mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)

data_dir = "./data"
x_train, y_train, x_test, y_test = load_data(data_dir)

print("***"*3,"before shuffle","***"*3)
print("the first 10 x_train:")
print(x_train[:10]) #[[user,item]*311704个]
print("the first 10 y_train:")
print(y_train[:10]) #[rating*311704个]
print("the first 10 x_test:")
print(x_test[:10]) #[[user,item]*54000个]
print("the first 10 y_test:")
print(y_test[:10]) #[rating*54000个]
x_train, y_train = shuffle(x_train, y_train)
num_user = x_train[:,0].max() + 1
num_item = x_train[:,1].max() + 1

print("# user: {}, # item: {}".format(num_user, num_item))
# binarize
# if rating>=3,y=1;if rating<3,y=0
y_train = binarize(y_train)
y_test = binarize(y_test)
print("***"*3,"After shuffle and binarize","***"*3)
print("the first 10 y_train:")
print(y_train[:10]) #[rating*311704个]
print("the first 10 y_test:")
print(y_test[:10]) #[rating*54000个]

#MF CVIB TODO:

p_list=[]
for lr_search in [0.1,0.05,0.01,0.005,0.001]:
    for wd_search in [1e-3,1e-4,1e-5]:
        for bs_search in [128,256,512,1024,2048]:
            #for alpha_search in [2,1,0.5,0.1]:
            #for gamma_search in [1,0.1,1e-2,1e-3]:
            mf_cvib = MF_CVIB(num_user, num_item)
            mf_cvib.fit(x_train, y_train,
            num_epoch=10000,
            lr=lr_search,
            batch_size=bs_search,
            lamb=wd_search,
            alpha=1,
            gamma=0.001,
            tol=1e-5,
            verbose=False)
            test_pred = mf_cvib.predict(x_test)
            mse_mf = mse_func(y_test, test_pred)
            auc_mf = roc_auc_score(y_test, test_pred)
            ndcg_res = ndcg_func(mf_cvib, x_test, y_test)
            #print("***"*5 + "[MF-CVIB]" + "***"*5)
            #print("***"*5 + "[MF-CVIB]" + "***"*5)#########
            #print("[MF-CVIB] lr:{},wd:{},bs:{},alpha:{},gamma:{}".format(lr_search,wd_search,bs_search,alpha_search,gamma_search))#######
            #print("[MF-CVIB] test mse:", mse_mf)
            #print("[MF-CVIB] test auc:", auc_mf)
            #if mse_mf<30 and auc_mf>60:
            print("[MF-CVIB] lr:{},wd:{},bs:{},mse_mf:{},auc_mf:{}".format(lr_search,wd_search,bs_search,mse_mf,auc_mf))#######
            #print("[MF] ndcg@5:{:.6f}, ndcg@10:{:.6f}".format(
            #        np.mean(ndcg_res["ndcg_5"]), np.mean(ndcg_res["ndcg_10"])))
            #user_wise_ctr = get_user_wise_ctr(x_test,y_test,test_pred)
            #gi,gu = gini_index(user_wise_ctr)
            #print("***"*5 + "[MF-CVIB]" + "***"*5)