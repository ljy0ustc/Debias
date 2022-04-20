import numpy as np
import torch
torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F

def generate_total_sample(num_user, num_item):
    # generate all user-item pairs
    sample = []
    for i in range(num_user):
        sample.extend([[i,j] for j in range(num_item)])
    return np.array(sample)

class MF_CVIB(nn.Module):
    def __init__(self, num_users, num_items, embedding_k=4, *args, **kwargs):
        super(MF_CVIB, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)

        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0])
        item_idx = torch.LongTensor(x[:,1])
        zz_user_emb = self.W(user_idx)
        cc_user_emb = self.W(user_idx)
        z_user_emb = torch.cat((zz_user_emb,torch.zeros_like(zz_user_emb)),1)
        c_user_emb = torch.cat((torch.zeros_like(cc_user_emb),cc_user_emb),1)

        zz_item_emb = self.H(item_idx)
        cc_item_emb = self.H(item_idx)
        z_item_emb = torch.cat((zz_item_emb,torch.zeros_like(zz_item_emb)),1)
        c_item_emb = torch.cat((torch.zeros_like(cc_item_emb),cc_item_emb),1)

        z_out = torch.sum(z_user_emb.mul(z_item_emb), 1)
        c_out = torch.sum(c_user_emb.mul(c_item_emb), 1)
        zc_out = torch.sum((z_user_emb+c_user_emb).mul(z_item_emb+c_item_emb), 1)

        z_out = self.sigmoid(z_out)
        c_out = self.sigmoid(c_out)
        zc_out = self.sigmoid(zc_out)

        if is_training:
            return z_out, c_out, zc_out
        else:
            return z_out

    def fit(self, x, y, 
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        alpha=0.1, gamma=0.01,
        tol=1e-4, verbose=True):

        self.alpha = alpha
        self.gamma = gamma

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        # generate all counterfactuals and factuals for info reg
        #x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // batch_size #整数除法，向下取整
        early_stop = 0

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            #ul_idxs = np.arange(x_all.shape[0])
            #np.random.shuffle(ul_idxs)

            epoch_loss = 0
            for idx in range(total_batch):
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y)

                z_pred, c_pred, zc_pred = self.forward(sub_x, True)
                xent_loss = self.xent_func(z_pred,sub_y)
                mf_loss = (1-self.gamma)*self.xent_func(z_pred,sub_y)-(self.gamma-self.alpha)*self.xent_func(c_pred,sub_y)+self.gamma*self.xent_func(zc_pred,sub_y)

                loss = mf_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.detach().numpy()


            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DIB] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DIB] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DIB] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().numpy()

class NCF_CVIB(nn.Module):
    """The neural collaborative filtering method.
    """
    def __init__(self, num_users, num_items, embedding_k=4):
        super(NCF_CVIB, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.linear_1 = torch.nn.Linear(self.embedding_k*4, self.embedding_k)
        self.relu = torch.nn.ReLU()
        self.linear_2 = torch.nn.Linear(self.embedding_k, 1, bias=False)
        self.sigmoid = torch.nn.Sigmoid()

        self.xent_func = torch.nn.BCELoss()


    def forward(self, x, is_training=False):
        user_idx = torch.LongTensor(x[:,0])
        item_idx = torch.LongTensor(x[:,1])
        zz_user_emb = self.W(user_idx)
        cc_user_emb = self.W(user_idx)
        z_user_emb = torch.cat((zz_user_emb,torch.zeros_like(zz_user_emb)),1)
        c_user_emb = torch.cat((torch.zeros_like(cc_user_emb),cc_user_emb),1)

        zz_item_emb = self.H(item_idx)
        cc_item_emb = self.H(item_idx)
        z_item_emb = torch.cat((zz_item_emb,torch.zeros_like(zz_item_emb)),1)
        c_item_emb = torch.cat((torch.zeros_like(cc_item_emb),cc_item_emb),1)

        zc_user_emb=z_user_emb+c_user_emb
        zc_item_emb=z_item_emb+c_item_emb

        z_emb=torch.cat([z_user_emb, z_item_emb], axis=1)
        c_emb=torch.cat([c_user_emb, c_item_emb], axis=1)
        zc_emb=torch.cat([zc_user_emb, zc_item_emb], axis=1)

        z_h1 = self.linear_1(z_emb)
        z_h1 = self.relu(z_h1)
        z_out = self.linear_2(z_h1)

        c_h1 = self.linear_1(c_emb)
        c_h1 = self.relu(c_h1)
        c_out = self.linear_2(c_h1)

        zc_h1 = self.linear_1(zc_emb)
        zc_h1 = self.relu(zc_h1)
        zc_out = self.linear_2(zc_h1)

        z_out = self.sigmoid(z_out)
        c_out = self.sigmoid(c_out)
        zc_out = self.sigmoid(zc_out)

        if is_training:
            return z_out, c_out, zc_out
        else:
            return z_out

    def fit(self, x, y, 
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, 
        alpha=0.1, gamma=0.01,
        tol=1e-4, verbose=True):

        self.alpha = alpha
        self.gamma = gamma

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        # generate all counterfactuals and factuals for info reg
        #x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x)
        total_batch = num_sample // batch_size #整数除法，向下取整
        early_stop = 0

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            #ul_idxs = np.arange(x_all.shape[0])
            #np.random.shuffle(ul_idxs)

            epoch_loss = 0
            for idx in range(total_batch):
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y)

                z_pred, c_pred, zc_pred = self.forward(sub_x, True)
                xent_loss = self.xent_func(torch.squeeze(z_pred),sub_y)
                mf_loss = (1-self.gamma)*self.xent_func(torch.squeeze(z_pred),sub_y)-(self.gamma-self.alpha)*self.xent_func(torch.squeeze(c_pred),sub_y)+self.gamma*self.xent_func(torch.squeeze(zc_pred),sub_y)

                loss = mf_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += xent_loss.detach().numpy()


            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DIB] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DIB] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DIB] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().numpy().flatten()

    def predict_proba(self, x):
        pred = self.forward(x)
        pred = pred.reshape(-1,1)
        return np.concatenate([1-pred,pred],axis=1)