import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def fc(input, num_output, name = 'fc'):
    with tf.variable_scope(name):
        num_input = input.get_shape()[1]
        W = tf.get_variable('w', [num_input, num_output], tf.float32, tf.random_normal_initializer(0.0, 0.02))
        b = tf.get_variable('b', [num_output], initializer = tf.constant_initializer(0.0))
        return tf.matmul(input, W) + b
    
def dropout(input,keep_prob,is_train):
    return tf.layers.dropout(input,keep_prob,training=is_train)

class NN_force(object):
    def __init__(self,input_dim,output_dim,num_epoch=5,batch_size=64,log_step=50,
                 drop_prob=0.50,decay_rate =5e-4):
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.log_step = log_step
        self.input_dim = input_dim
        self.output_dim = 1
        self.drop_prob = drop_prob
        self.decay_rate = decay_rate
        self._build_model()
    
    def _init_variable(self):
        self.X = tf.placeholder(tf.float32, [None, self.input_dim])
        self.Y = tf.placeholder(tf.float32, [None,self.output_dim])
        self.is_train  = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32)
    
    def _model(self):
        print('intput layer: ' + str(self.X.get_shape()))
        with tf.variable_scope('fc1'):
            self.fc1 = fc(self.X,512)
            self.relufc1 = tf.nn.relu(self.fc1)
            self.dropfc1 = dropout(self.relufc1,self.keep_prob,self.is_train)
            print('fc1 layer: ' + str(self.dropfc1.get_shape()))
        
        with tf.variable_scope('fc2'):
            self.fc2 = fc(self.dropfc1,1024)
            self.relufc2 = tf.nn.relu(self.fc2)
            self.dropfc2 = dropout(self.relufc2,self.keep_prob,self.is_train)
            print('f2 layer: ' + str(self.dropfc2.get_shape()))

        with tf.variable_scope('fc3'):
            self.fc3 = fc(self.dropfc2,512)
            self.relufc3 = tf.nn.relu(self.fc3)
            self.dropfc3 = dropout(self.relufc3,self.keep_prob,self.is_train)
            print('f3 layer: ' + str(self.dropfc3.get_shape()))

        #with tf.variable_scope('fc3a'):
        #    self.fc3a = fc(self.dropfc3,256)
        #    self.relufc3a = tf.nn.relu(self.fc3a)
        #    self.dropfc3a = dropout(self.relufc3a,self.keep_prob,self.is_train)
        #    print('f3 layer: ' + str(self.dropfc3a.get_shape()))            

        with tf.variable_scope('fc4'):
            self.fc4 = fc(self.dropfc3,1)            
            print('fc layer: ' + str(self.fc4.get_shape()))
    
        return self.fc4
    
    def _build_optimizer(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(self.decay_rate, self.global_step, 500, 0.96)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(loss=self.loss_op,global_step=self.global_step) 
    
    def lossfunction(self, predict_Y, true_Y):
        mm=tf.square(predict_Y-true_Y)
        #mm=tf.abs(predict_Y-true_Y)
        print("output shape:",mm.shape,predict_Y.shape,true_Y.shape)
        self.loss_op = tf.reduce_mean(mm)  
    
    def _build_model(self):
        self._init_variable()
        predict_Y = self._model()
        self.lossfunction(predict_Y, self.Y)
        self._build_optimizer()
        self.predicted = predict_Y


class trainNN():
    def __init__(self,nnmodel):
        self.step = 0
        self.losses = []
        self.accuracies = []
        self.train_accuracy=[]
        self.test_accuracy=[]
        self.nnmodel = nnmodel
    
    def initilize(self,tag=0):
        graph1_init_op = tf.global_variables_initializer()
        return graph1_init_op
    
    def train(self,sess,num_training,X_train, Y_train,fname='NN_Ar.ckpt'):

        print('-' * 5 + '  Start training  ' + '-' * 5)
        for epoch in range(self.nnmodel.num_epoch):
            print('train for epoch %d' % epoch)
            for i in range(num_training // self.nnmodel.batch_size):
                X_ = X_train[i * self.nnmodel.batch_size:(i + 1) * self.nnmodel.batch_size][:]
                Y_ = Y_train[i * self.nnmodel.batch_size:(i + 1) * self.nnmodel.batch_size]

                feed_dict = {self.nnmodel.X : X_, self.nnmodel.Y : Y_, self.nnmodel.is_train : True,
                             self.nnmodel.keep_prob: self.nnmodel.drop_prob}                
                fetches = [self.nnmodel.train_op, self.nnmodel.loss_op]

                _, loss = sess.run(fetches, feed_dict=feed_dict)
                self.losses.append(loss)

                if self.step % self.nnmodel.log_step == 0:
                    print('iteration (%d): loss = %.3f,' % (self.step, loss))
                self.step += 1

            #############################################################################
            # TODO: Plot training curves                                                #
            #############################################################################
            if epoch % 5 == 0 :
                loss_hist_ = self.losses[1::50]
        saver = tf.train.Saver()
        save_path = saver.save(sess, "NN_Training/"+fname)
        print("Model saved in path: %s" % save_path)
        plt.subplot(2, 1, 1)
        plt.title('Training loss')
        print('final iteration (%d): loss = %.3f,' % (self.step, loss))
        plt.plot(loss_hist_, '-o')
        plt.show()
        return loss_hist_

class NN_ML_model():
    def __init__(self,nnmodel):
        self.nnmodel = nnmodel
    
    def initilize(self,sess,path):
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, path)
        print("Model restored.")
    
    def predict_force(self,sess,X_eval):
        Y_=np.empty((X_eval.shape[0],self.nnmodel.output_dim),dtype=float)   
        for i in range(X_eval.shape[0] // self.nnmodel.batch_size):    
            X_ = X_eval[i * self.nnmodel.batch_size:(i + 1) * self.nnmodel.batch_size][:]            
            feed_dict = {self.nnmodel.X : X_, self.nnmodel.is_train : False, self.nnmodel.keep_prob: self.nnmodel.drop_prob}
            tt = sess.run(self.nnmodel.predicted, feed_dict=feed_dict)
            Y_[i * self.nnmodel.batch_size:(i + 1) * self.nnmodel.batch_size]=tt 
        return Y_
    
    def predict_force_single(self,sess,X_eval):
        X_ = np.expand_dims(X_eval, axis=0)
        #print(X_.shape)
        feed_dict = {self.nnmodel.X : X_, self.nnmodel.is_train : 
                     False, self.nnmodel.keep_prob: self.nnmodel.drop_prob}
        Y_ = sess.run(self.nnmodel.predicted, feed_dict=feed_dict)
        return Y_[0][0]

class create_sess:
    def __init__(self,graph,model_param,fname='Ar.ckpt'):
        self.graph = graph
        self.input_dim = model_param['input_dim']
        self.output_dim = model_param['output_dim']
        self.num_epoch = model_param['num_epoch']
        self.batch_size = model_param['batch_size']
        self.drop_prob = model_param['drop_prob']
        self.decay_rate = model_param['decay_rate']   
        self.fname = fname
    def create_model(self,num_training,train_XX,train_YY):
        with self.graph.as_default():
            Force_model = NN_force(input_dim = self.input_dim,output_dim = self.output_dim,num_epoch = self.num_epoch,
                                   batch_size = self.batch_size,drop_prob = self.drop_prob,decay_rate = self.decay_rate)
            Train_F = trainNN(Force_model) 
            graph_init_op = Train_F.initilize()
            sess = tf.Session(graph=self.graph)
            sess.run(graph_init_op)
            loss_hist_ = Train_F.train(sess,num_training,train_XX,train_YY,fname=self.fname)
            sess.close()
            return loss_hist_
    def load_model(self):
        with self.graph.as_default():
            Force_model = NN_force(input_dim=self.input_dim,output_dim=self.output_dim,batch_size=self.batch_size)
            ML_Force_Cal = NN_ML_model(Force_model)
            sess = tf.Session(graph=self.graph)
            ML_Force_Cal.initilize(sess,'NN_Training/'+self.fname)
        return sess,ML_Force_Cal
            
                                    
