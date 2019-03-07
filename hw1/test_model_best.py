import argparse
import os
import pathlib
import pandas as pd
import numpy as np

def load_model(path):
    with open(path) as f:
        paras = f.read().split('\n')
        paras.pop()
        paras = [ float(x) for x in paras ] 

    paras = np.array(paras).astype(np.float)
    paras = paras.reshape(-1,1)
    return paras

def load_test(path):
    total_data = pd.read_csv(path,header=None)
    #print(total_data.head())
    #print(total_data.shape)

    test_X = []
    for i in range(240):
        data = total_data.iloc[i*18:(i+1)*18,2:].values
        data = data.reshape(-1)
        test_X.append(data)

    for dataidx in range(len(test_X)):
        test_X[dataidx] = np.append(test_X[dataidx],'1')
        for valueidx in range(len(test_X[0])):
            if test_X[dataidx][valueidx]=='NR':
                test_X[dataidx][valueidx]=0
            else:
                test_X[dataidx][valueidx] = float(test_X[dataidx][valueidx])

    test_X = np.array(test_X).astype(np.float)
    return test_X

def get_normalize(path):
    f = open(path)
    data = f.readlines()
    mean_str = data[0][:-1].split(' ')
    mean = [float(x) for x in mean_str]
    mean = np.array(mean).astype(np.float)
    std_str = data[1].split(' ')
    std = [float(x) for x in std_str]
    std = np.array(std).astype(np.float)
    return mean,std

def write_csv(test_Y,path):
    test_Y = np.ndarray.tolist(test_Y.reshape(-1))
    print(len(test_Y))
    idlist = [ 'id_'+ str(x) for x in range(240) ]
    df = pd.DataFrame({'id':idlist,'value':test_Y})
    if os.path.dirname(path)!='': 
        if not os.path.isdir(os.path.dirname(path)):
            dirname = os.path.dirname(path)
            odir = pathlib.Path(dirname)
            odir.mkdir(parents=True, exist_ok=True)
            #os.mkdir(os.path.dirname(path))
    df.to_csv(path,index=False)

class Linear_Regression():
    def __init__(self,paranum,paras=None):
        self.paranum = paranum
        if paras is not None:
            self.best_W = paras
        else:
            self.W = np.zeros(self.paranum).reshape(-1,1)
        self.ada = np.zeros(self.paranum).reshape(-1,1)
        self.epoch = 0

    def MSE(self,X,Y):
        pre_Y = np.matmul(X,self.W)
        L = (pre_Y - Y).reshape(-1)
        
        loss = np.square(L).mean()
        return loss
    
    def train_with_val(self,train_X,train_Y,val_X,val_Y,shuffle,epochs,lr,lamda,modelpath,bestmodelpath):
        assert train_X.shape[0]==train_Y.shape[0]

        if shuffle==True:
            total_train_X_tmp = train_X
            total_train_Y_tmp = train_Y
            train_batch_ind = [i for i in range(int(total_train_X_tmp.shape[0]))]
            import random as rd
            rd.shuffle(train_batch_ind)

        for epoch in range(epochs):

            if shuffle==True:
                rd.shuffle(train_batch_ind)
                train_X = total_train_X_tmp[train_batch_ind]
                train_Y = total_train_Y_tmp[train_batch_ind]

            predict_Y = np.matmul(train_X,self.W).reshape(-1,1)
            loss = predict_Y - train_Y
            gradient = 2* np.matmul(np.transpose(train_X),loss) + 2*lamda*self.W
            #print(gradient.shape)
            self.ada += gradient**2
            self.W = self.W - lr/np.sqrt(self.ada)*gradient
            val_loss = self.MSE(val_X,val_Y)
            train_loss = self.MSE(total_train_X_tmp,total_train_Y_tmp)

            if epoch%1000==999:
                print('epoch: {}, training loss: {}\t, validation loss: {}'.format(self.epoch,train_loss,val_loss))
            if self.epoch==0:
                self.best_train_loss = train_loss
                self.best_epoch = 0
                self.best_W = self.W
                self.save_model(bestmodelpath)
            else:
                if self.best_train_loss > train_loss:
                    self.best_train_loss = train_loss
                    self.val_loss_in_best_epoch = val_loss
                    self.best_epoch = self.epoch
                    self.best_W = self.W
                    self.save_model(bestmodelpath)

            self.epoch+=1
            self.save_model(modelpath)

        print('best epoch: {}'.format(self.best_epoch))
        print('best train loss: {}'.format(self.best_train_loss))
        print('val loss in best epoch: {}'.format(self.val_loss_in_best_epoch))

    def train(self,train_X,train_Y,epochs,lr,lamda,modelpath,bestmodelpath):
        assert train_X.shape[0]==train_Y.shape[0]

        for epoch in range(epochs):
            predict_Y = np.matmul(train_X,self.W).reshape(-1,1)
            loss = predict_Y - train_Y
            gradient = 2* np.matmul(np.transpose(train_X),loss) + 2*lamda*self.W
            #print(gradient.shape)
            self.ada += gradient**2
            self.W = self.W - lr/np.sqrt(self.ada)*gradient
            train_loss = self.MSE(train_X,train_Y)

            if epoch%1000==999:
                print('epoch: {}, training loss: {}'.format(self.epoch,train_loss))
            if self.epoch==0:
                self.best_train_loss = train_loss
                self.best_epoch = 0
                self.best_W = self.W
                self.save_model(bestmodelpath)
            else:
                if self.best_train_loss > train_loss:
                    self.best_train_loss = train_loss
                    self.best_epoch = self.epoch
                    self.best_W = self.W
                    self.save_model(bestmodelpath)

            self.epoch+=1
            self.save_model(modelpath)

        print('best epoch: {}'.format(self.best_epoch))
        print('best train loss: {}'.format(self.best_train_loss))

    def projection(self,train_X,train_Y,modelpath):
        W = np.matmul(train_X.transpose(),train_X)
        W = np.linalg.inv(W)
        W = np.matmul(W,train_X.transpose())
        W = np.matmul(W,train_Y)
        self.W = W
        print('loss: {}'.format(self.MSE(train_X,train_Y)))
        self.save_model(modelpath)

    def save_model(self,path):
        np.savetxt(path,self.W,delimiter='\n')

    def clear_ada(self):
        self.ada = np.zeros(self.paranum).reshape(-1,1)

    def test(self,test_X):
        test_Y = np.matmul(test_X,self.best_W)
        return test_Y

def extract_features(train_X,extract_features_list):
    attrs = ['AMB', 'CH4', 'CO', 'NMHC', 'NO', 'NO2',
            'NOx', 'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH',
            'SO2', 'THC', 'WD_HR', 'WIND_DIR', 'WIND_SPEED', 'WS_HR']
    tmp = [i for i in range(len(attrs))]
    attr_dict = {attrs[i]:tmp[i] for i in range(len(attrs))}

    extract_ind_list = [attr_dict[feature] for feature in extract_features_list]
    print(extract_ind_list)

    X_shape = train_X.shape
    part_train_X = []
    for data in train_X:
        data_part = data[:-1].reshape(18,-1)[extract_ind_list]
        data_part = data_part.reshape(-1)
        data_part = np.concatenate((data_part,np.ones(1)),axis=None)
        part_train_X.append(data_part)
    train_X = np.array(part_train_X)
    return train_X

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile", help="input file path")
    parser.add_argument("outputfile", help="output file path")
    parser.add_argument("modelpath", help="model path")
    args = parser.parse_args()
    print(args.inputfile)
    print(args.outputfile)

    paras = load_model(args.modelpath)
    model = Linear_Regression(paranum=64, paras=paras)

    test_X = load_test(args.inputfile)
    print('test_X raw {}'.format(test_X.shape))

    #mean,std = get_normalize('scale/normalize_hw1.txt')
    #test_X = (test_X-mean)/std

    

    extract_features_list = ['CH4', 'CO', 'NO2', 'O3', 'PM10', 'PM2.5', 'SO2']

    test_X = extract_features(test_X,extract_features_list)

    test_Y = model.test(test_X)

    write_csv(test_Y,args.outputfile)

