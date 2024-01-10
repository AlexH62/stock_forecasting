import os
import yfinance as yahooFinance
import numpy as np
import tensorflow as tf
import ltc_model as ltc
from ctrnn_model import CTRNN, NODE, CTGRU

tf.compat.v1.disable_eager_execution()

def get_data(ticker, period="max"):
    info = yahooFinance.Ticker(ticker)

    # Valid periods are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, and max.
    return info.history(period=period)

class Model:

    def __init__(self, model_type, model_size, sparsity_level=0.0, learning_rate = 0.001):
        self.model_type = model_type
        self.constrain_op = []
        self.sparsity_level = sparsity_level
        self.x = tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,None,5])
        self.target_y = tf.compat.v1.placeholder(dtype=tf.int32,shape=[None,None])

        self.model_size = model_size
        head = self.x
        if(model_type == "lstm"):
            self.fused_cell = tf.compat.v1.nn.rnn_cell.LSTMCell(model_size)

            head,_ = tf.compat.v1.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type.startswith("ltc")):
            learning_rate = 0.005 # LTC needs a higher learning rate
            self.wm = ltc.LTCCell(model_size)
            if(model_type.endswith("_rk")):
                self.wm._solver = ltc.ODESolver.RungeKutta
            elif(model_type.endswith("_ex")):
                self.wm._solver = ltc.ODESolver.Explicit
            else:
                self.wm._solver = ltc.ODESolver.SemiImplicit

            head,_ = tf.compat.v1.nn.dynamic_rnn(self.wm,head,dtype=tf.float32,time_major=True)
            self.constrain_op.extend(self.wm.get_param_constrain_op())
        elif(model_type == "node"):
            self.fused_cell = NODE(model_size,cell_clip=-1)
            head,_ = tf.compat.v1.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctgru"):
            self.fused_cell = CTGRU(model_size,cell_clip=-1)
            head,_ = tf.compat.v1.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        elif(model_type == "ctrnn"):
            self.fused_cell = CTRNN(model_size,cell_clip=-1,global_feedback=True)
            head,_ = tf.compat.v1.nn.dynamic_rnn(self.fused_cell,head,dtype=tf.float32,time_major=True)
        else:
            raise ValueError("Unknown model type '{}'".format(model_type))

        if(self.sparsity_level > 0):
            self.constrain_op.extend(self.get_sparsity_ops())

        self.y = tf.compat.v1.layers.Dense(2,activation=None)(head)
        print("logit shape: ",str(self.y.shape))
        self.loss = tf.reduce_mean(tf.compat.v1.losses.sparse_softmax_cross_entropy(
            labels = self.target_y,
            logits = self.y,
        ))
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        model_prediction = tf.argmax(input=self.y, axis=2)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(model_prediction, tf.cast(self.target_y,tf.int64)), tf.float32))

        self.sess = tf.compat.v1.InteractiveSession()
        self.sess.run(tf.compat.v1.global_variables_initializer())

        self.result_file = os.path.join("results","test","{}_{}_{:02d}.csv".format(model_type,model_size,int(100*self.sparsity_level)))
        if(not os.path.exists("results/test")):
            os.makedirs("results/test")
        if(not os.path.isfile(self.result_file)):
            with open(self.result_file,"w") as f:
                f.write("best epoch, train loss, train accuracy, valid loss, valid accuracy, test loss, test accuracy\n")

        self.checkpoint_path = os.path.join("tf_sessions","test","{}".format(model_type))
        if(not os.path.exists("tf_sessions/test")):
            os.makedirs("tf_sessions/test")
            
        self.saver = tf.compat.v1.train.Saver()

    def get_sparsity_ops(self):
        tf_vars = tf.compat.v1.trainable_variables()
        op_list = []
        for v in tf_vars:
            # print("Variable {}".format(str(v)))
            if(v.name.startswith("rnn")):
                if(len(v.shape)<2):
                    # Don't sparsity biases
                    continue
                if("ltc" in v.name and (not "W:0" in v.name)):
                    # LTC can be sparsified by only setting w[i,j] to 0
                    # both input and recurrent matrix will be sparsified
                    continue
                op_list.append(self.sparse_var(v,self.sparsity_level))
                
        return op_list
        
    def sparse_var(self,v,sparsity_level):
        mask = np.random.choice([0, 1], size=v.shape, p=[sparsity_level,1-sparsity_level]).astype(np.float32)
        v_assign_op = tf.compat.v1.assign(v,v*mask)
        print("Var[{}] will be sparsified with {:0.2f} sparsity level".format(
            v.name,sparsity_level
        ))
        return v_assign_op

    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)


    def fit(self,gesture_data,epochs,verbose=True,log_period=50):

        best_valid_accuracy = 0
        best_valid_stats = (0,0,0,0,0,0,0)
        self.save()
        for e in range(epochs):
            if(verbose and e%log_period == 0):
                test_acc,test_loss = self.sess.run([self.accuracy,self.loss],{self.x:gesture_data.test_x,self.target_y: gesture_data.test_y})
                valid_acc,valid_loss = self.sess.run([self.accuracy,self.loss],{self.x:gesture_data.valid_x,self.target_y: gesture_data.valid_y})
                # Accuracy metric -> higher is better
                if(valid_acc > best_valid_accuracy and e > 0):
                    best_valid_accuracy = valid_acc
                    best_valid_stats = (
                        e,
                        np.mean(losses),np.mean(accs)*100,
                        valid_loss,valid_acc*100,
                        test_loss,test_acc*100
                    )
                    self.save()

            losses = []
            accs = []
            for batch_x,batch_y in gesture_data.iterate_train(batch_size=16):
                acc,loss,_ = self.sess.run([self.accuracy,self.loss,self.train_step],{self.x:batch_x,self.target_y: batch_y})
                if(len(self.constrain_op) > 0):
                    self.sess.run(self.constrain_op)

                losses.append(loss)
                accs.append(acc)

            if(verbose and e%log_period == 0):
                print("Epochs {:03d}, train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test accuracy: {:0.2f}%".format(
                    e,
                    np.mean(losses),np.mean(accs)*100,
                    valid_loss,valid_acc*100,
                    test_loss,test_acc*100
                ))
            if(e > 0 and (not np.isfinite(np.mean(losses)))):
                break
        self.restore()
        best_epoch,train_loss,train_acc,valid_loss,valid_acc,test_loss,test_acc = best_valid_stats
        print("Best epoch {:03d}, train loss: {:0.2f}, train accuracy: {:0.2f}%, valid loss: {:0.2f}, valid accuracy: {:0.2f}%, test loss: {:0.2f}, test accuracy: {:0.2f}%".format(
            best_epoch,
            train_loss,train_acc,
            valid_loss,valid_acc,
            test_loss,test_acc
        ))
        with open(self.result_file,"a") as f:
            f.write("{:03d}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}, {:0.2f}\n".format(
            best_epoch,
            train_loss,train_acc,
            valid_loss,valid_acc,
            test_loss,test_acc
        ))



model = Model(model_type = "ltc_rk",model_size=32)
data = get_data("AMZN")
model.fit(data, epochs=200)

