#
# Imports
from __future__ import print_function
import os
import mxnet as mx
from mxnet import nd, autograd, gluon
import matplotlib
import cv2
import matplotlib.pyplot as plt
import numpy as np
from imutils import paths
from mxnet.gluon import nn, Block
mx.random.seed(1)
from tables import *
from sklearn.decomposition import PCA
import sklearn.model_selection as sk
import matplotlib.pyplot as plt
from time import time
import datetime
now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

###########################
#  Speficy the context we'll be using
###########################

# there should be a function to return the results

###########################
#  Load up our dataset
###########################

class Regression_plt():
    def __init__(self):

        self.data_ctx = mx.cpu()
        self.model_ctx = mx.cpu()

        self.tr_input,  self.tr_target,  self.ts_input, \
        self.ts_target,  self.size_1,  self.size_2 = self.Reg_Data()

        self.num_outputs = 4
        self.filedir = "save_load/"


    def Reg_Data(self):

        # putting more functionality for the data
        # all data altoghether
        #
        MAIN_DIR = ''
        h5file = open_file(MAIN_DIR + 'h5_plt_data_file.h5',"a")
        DATA_01 = h5file.root.plate_data.datray[:]
        h5file.close()
        # defining batch_xs and batch_ys

        tr_target = DATA_01[:400, -5:-1]
        ts_target = nd.array(DATA_01[400:, -5:-1])

        # # not for keras learning app.
        pca = PCA(0.9999999).fit(DATA_01[:, :20000])
        # #pca.n_components_
        tr_input = nd.array(pca.fit_transform(DATA_01[:, :20000]))
        ts_input = tr_input[400:, :]
        tr_input = tr_input[:400, :]

        #input1 = inp_raw[:,:2000]
        #target = target[:,-5:-1]
        #tr_input = nd.array(tr_input)
        #tr_target = nd.array(tr_target)

        return tr_input, tr_target, ts_input, ts_target, tr_input.shape[0], tr_input.shape[1]

#for data, _ in inp_raw:
#    data = data.as_in_context(data_ctx)
#    break

    def Model(self):
        num_hidden = self.size_2
        net = gluon.nn.Sequential()
        with net.name_scope():
        ###########################
        # Adding first hidden layer
        ###########################
            net.add(gluon.nn.Dense(int(.6*num_hidden), in_units=self.size_2, activation= "tanh"))
            net.add(gluon.nn.Dropout(.5))

            net.add(gluon.nn.Dense(int(.4*num_hidden), in_units=int(.6*num_hidden), activation="tanh"))
            net.add(gluon.nn.Dropout(.3))

            net.add(gluon.nn.Dense(int(.2*num_hidden), in_units=int(.4*num_hidden), activation="softrelu"))
            ###########################
            # Adding dropout with rate .75 to the first hidden layer
            ###########################
            net.add(gluon.nn.Dropout(.1))

            ###########################
            # Adding second hidden layer
            ###########################
            net.add(gluon.nn.Dense(int(.1*num_hidden), in_units=int(.2*num_hidden), activation="softrelu"))
            ###########################
            # Adding dropout with rate .75 to the second hidden layer
            ###########################
            net.add(gluon.nn.Dropout(.1))

            ###########################
            # Adding the output layer
            ###########################
            net.add(gluon.nn.Dense(self.num_outputs, in_units=int(.1*num_hidden)))

            return net

    def Train_Reg_Model(self):
        #
        # define basic functions:
        saveit = 10
        epochs = 20  # Low number for testing, set higher when you run!
        batch_size = 100
        num_batch = int(self.size_1 / batch_size)
        lr = 0.005
        optimizer = ['sgd', 'adam']

        net = self.Model()

        loss = gluon.loss.L2Loss()
        net.collect_params().initialize(mx.init.Normal(sigma=.01), ctx=self.model_ctx)
        trainer = gluon.Trainer(net.collect_params(),\
                                optimizer=optimizer[1], optimizer_params={'learning_rate': lr})

        print('training initialize')
        start = time()
        initial_time = start
        for e in range(epochs):
            mn_l = 0

            for batch_i in range(batch_size):
                data = self.tr_input.as_in_context(self.data_ctx).reshape((-1, self.size_2))
                target = self.target.as_in_context(self.data_ctx)

                with autograd.record(train_mode=True):
                    output = net(data)
                    l = loss(output, target)
                    l.backward()

                trainer.step(num_batch)

                mn_l += nd.mean(l).asnumpy()
            # saving network
            if (time() - initial_time) % 2 * 3600 == 0:
                filename = "save_load/" + "testnet" + now + ".params"
                net.save_params(filename)
                print('file is saved on time')
            elif (e % saveit) == 0:
                filename = "save_load/" + "testnet" + now + ".params"
                net.save_params(filename)
                print('file is saved with mode')

                # print(l)
            # in place addition for nd arrays
            # nd.elemwise_add(sum(l) , mn_l, out=mn_l)
            comp_time = time() - start
            print('epoch: {}, loss: {}, comp time: {}'.format(e, mn_l, comp_time))

            return net.params


    # this part is for testing
    def mlp_tst(self):
        ## first we should define net the same as training, then load the saved parameters
        net = self.Model()
        net.load_params(self.filedir + "testnet2018-03-17 22:58:16.515013.params", ctx=self.model_ctx)

        # which image?
        # an integer between 0-453
        wi =410
        predict = net(self.ts_input[wi].reshape((1,-1))).asnumpy()
        # Showing the results
        image = self.ts_input[wi].reshape([100,200])
        #plt_points = [[pred[0] pred[1]],[pred[0] + pred[2], pred[1]],\
        #              [pred[0], pred[1] + pred[3]], [pred[0] + pred[2], pred[1] + pred[3]]]
        #cv2.convexHull(plt_points, dtype='np.float32')
        # after process, padding should be removed or taget also should be padded

        image = cv2.rectangle(image, (int(predict[wi][0]) , int(predict[wi][1])), \
                              (int(predict[wi][2]), int(predict[wi][3])), (0, 200, 0), 3)

        plt.imshow(image)

        #for j in range(inp_raw.shape[0]):
        #    plate = cv2.rectangle(image[j], (pred[0], pred[1]),(pred[3], pred[4]), (200,0,0), 0)
        #    plt.imshow(plate)


