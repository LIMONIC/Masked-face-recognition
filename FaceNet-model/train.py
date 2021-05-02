import os,cv2,json,time
import numpy as np
import tensorflow
import math
from tensorflow.python.framework import graph_util


if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.compat.v1.gfile as gfile

print("Tensorflow version: ",tf.__version__)


class Classification():
    def __init__(self,para_dict):
        #----var parsing
        train_img_dir = para_dict['train_img_dir']
        test_img_dir = para_dict['test_img_dir']
        label_dict = para_dict['label_dict']

        #----get label names to number dictionary
        if label_dict is None:
            label_dict = self.__get_label_dict(train_img_dir)
            print(label_dict)

        class_num = len(label_dict.keys())


        #----read training set paths and labels
        train_paths, train_labels = self.__get_paths_labels(train_img_dir,label_dict)
        print("train path shape:{}, train label shape:{}".format(train_paths.shape,train_labels.shape))

        #----read test set paths and labels
        if test_img_dir is not None:
            test_paths, test_labels = self.__get_paths_labels(test_img_dir,label_dict)
            print("test path shape:{}, test label shape:{}".format(test_paths.shape, test_labels.shape))
        #----local var to global
        self.train_img_dir = train_img_dir
        self.label_dict = label_dict
        self.train_paths = train_paths
        self.train_labels = train_labels
        self.class_num = class_num
        if test_img_dir is not None:
            self.test_img_dir = test_img_dir
            self.test_paths = test_paths
            self.test_labels = test_labels

    def model_init(self,para_dict):
        #----var parsing
        model_shape = para_dict['model_shape']#[N,H,W,C]
        infer_method = para_dict['infer_method']
        loss_method = para_dict['loss_method']
        opti_method = para_dict['opti_method']
        learning_rate = para_dict['learning_rate']
        save_dir = para_dict['save_dir']

        #----tf_placeholder declaration
        tf_input = tf.placeholder(shape=model_shape,dtype=tf.float32,name='input')
        tf_keep_prob = tf.placeholder(dtype=tf.float32,name="keep_prob")
        tf_label_batch = tf.placeholder(shape=[None],dtype=tf.int32,name="label_batch")

        #---inference selection
        if infer_method == "simple_resnet":
            output = self.simple_resnet(tf_input,tf_keep_prob,self.class_num)


        #---loss selection
        if loss_method == "cross_entropy":
            prediction = tf.nn.softmax(output,name="prediction")
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_label_batch,logits=output),
                                  name="loss")

        #----optimizer selection
        if opti_method == "adam":
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)


        #----create the dir to save model weights(CKPT, PB)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        out_dir_prefix = os.path.join(save_dir,"model")
        saver = tf.train.Saver(max_to_keep=5)


        #----appoint PB node names
        pb_save_path = os.path.join(save_dir,"pb_model.pb")
        pb_save_list = ['prediction']


        #----create the log(JSON)
        log_path = os.path.join(save_dir,"train_result_0.json")
        content = {"train_img_dir":self.train_img_dir, "test_img_dir":self.test_img_dir,"label_dict":self.label_dict}

        #----local var to global
        self.tf_input = tf_input
        self.tf_keep_prob = tf_keep_prob
        self.tf_label_batch = tf_label_batch
        self.optimizer = optimizer
        self.prediction = prediction
        self.out_dir_prefix = out_dir_prefix
        self.saver = saver
        self.pb_save_path = pb_save_path
        self.pb_save_list = pb_save_list
        self.log_path = log_path
        self.content = content
        self.save_dir = save_dir
        self.model_shape = model_shape
        self.loss = loss


    def train(self,para_dict):
        #----var parsing
        epochs = para_dict['epochs']
        GPU_ratio = para_dict['GPU_ratio']
        batch_size = para_dict['batch_size']


        #----local var
        train_loss_list = list()
        train_acc_list = list()
        test_loss_list = list()
        test_acc_list = list()

        #----calculate iterations of one epoch
        train_ites = math.ceil(self.train_paths.shape[0] / batch_size)
        if self.test_img_dir is not None:
            test_ites = math.ceil(self.test_paths.shape[0] / batch_size)

        #----GPU setting
        config = tf.ConfigProto(log_device_placement=True,
                                allow_soft_placement=True)
        if GPU_ratio is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio

        with tf.Session(config=config) as sess:
            #----tranfer learning check
            files = [file.path for file in os.scandir(self.save_dir) if file.name.split(".")[-1] == 'meta']
            if len(files) == 0:
                sess.run(tf.global_variables_initializer())
                print("no previous model param can be used!")
            else:
                check_name = files[-1].split("\\")[-1].split(".")[0]
                model_path = os.path.join(self.save_dir,check_name)
                self.saver.restore(sess,model_path)
                msg = "use previous model param:{}".format(model_path)
                print(msg)

            #----epoch training
            for epoch in range(epochs):
                #----record the start time
                d_t = time.time()

                train_loss = 0
                train_acc = 0
                test_loss = 0
                test_acc = 0

                #----shuffle
                indice = np.random.permutation(self.train_paths.shape[0])
                self.train_paths = self.train_paths[indice]
                self.train_labels = self.train_labels[indice]

                #----do optimizers(training by iteration)
                for index in range(train_ites):
                    #----get image start and end numbers
                    num_start = index * batch_size
                    num_end = np.minimum(num_start + batch_size, self.train_paths.shape[0])

                    batch_data = self.get_4D_data(self.train_paths[num_start:num_end],self.model_shape[1:])

                    #----put all data to tf placeholders
                    feed_dict = {self.tf_input:batch_data,
                                 self.tf_label_batch:self.train_labels[num_start:num_end],
                                 self.tf_keep_prob:0.8}

                    #----session run
                    sess.run(self.optimizer,feed_dict=feed_dict)


                #----evaluation(training set)
                for index in range(train_ites):
                    #----get image start and end numbers
                    num_start = index * batch_size
                    num_end = np.minimum(num_start + batch_size, self.train_paths.shape[0])

                    batch_data = self.get_4D_data(self.train_paths[num_start:num_end],self.model_shape[1:])

                    #----put all data to tf placeholders
                    feed_dict = {self.tf_input:batch_data,
                                 self.tf_label_batch:self.train_labels[num_start:num_end],
                                 self.tf_keep_prob:1.0}

                    #----session run
                    loss_temp, predict_temp = sess.run([self.loss,self.prediction],feed_dict=feed_dict)

                    #----calculate the loss and accuracy
                    train_loss += loss_temp
                    train_acc += self.evaluation(predict_temp,self.train_labels[num_start:num_end])

                train_loss /= train_ites
                train_acc /= self.train_paths.shape[0]
                #print("train_loss:{}, train_acc:{}".format(train_loss,train_acc))


                #----evaluation(test set)
                if self.test_img_dir is not None:
                    for index in range(test_ites):
                        # ----get image start and end numbers
                        num_start = index * batch_size
                        num_end = np.minimum(num_start + batch_size, self.test_paths.shape[0])

                        batch_data = self.get_4D_data(self.test_paths[num_start:num_end], self.model_shape[1:])

                        # ----put all data to tf placeholders
                        feed_dict = {self.tf_input: batch_data,
                                     self.tf_label_batch: self.test_labels[num_start:num_end],
                                     self.tf_keep_prob: 1.0}

                        # ----session run
                        loss_temp, predict_temp = sess.run([self.loss, self.prediction], feed_dict=feed_dict)

                        # ----calculate the loss and accuracy
                        test_loss += loss_temp
                        test_acc += self.evaluation(predict_temp, self.test_labels[num_start:num_end])

                    test_loss /= test_ites
                    test_acc /= self.test_paths.shape[0]
                    #print("test_loss:{}, test_acc:{}".format(test_loss, test_acc))

                #----save ckpt, pb files
                model_save_path = self.saver.save(sess,self.out_dir_prefix,global_step=epoch)
                print("save model CKPT to ",model_save_path)

                graph = tf.get_default_graph().as_graph_def()
                output_graph_def = graph_util.convert_variables_to_constants(sess,graph,self.pb_save_list)
                with gfile.GFile(self.pb_save_path,'wb')as f:
                    f.write(output_graph_def.SerializeToString())
                print("save PB file to ",self.pb_save_path)

                #----record the end time
                d_t = time.time() - d_t

                #----save results in the log file
                train_loss_list.append(float(train_loss))
                train_acc_list.append(float(train_acc))
                if self.test_img_dir is not None:
                    test_loss_list.append(float(test_loss))
                    test_acc_list.append(float(test_acc))

                self.content["train_loss_list"] = train_loss_list
                self.content["train_acc_list"] = train_acc_list
                if self.test_img_dir is not None:
                    self.content["test_loss_list"] = test_loss_list
                    self.content["test_acc_list"] = test_acc_list

                with open(self.log_path, 'w') as f:
                    json.dump(self.content,f)

                print("save the log file in ",self.log_path)



                #----display training results
                print("Epoch: ",epoch)
                print("training loss:{}, accuracy:{}".format(train_loss,train_acc))
                if self.test_img_dir is not None:
                    print("testing loss:{}, accuracy:{}".format(test_loss, test_acc))

                print("Epoch time consumption:",d_t)




    #----functions

    def evaluation(self,predictions,labels):
        count = 0
        for i in range(predictions.shape[0]):
            if np.argmax(predictions[i]) == labels[i]:
                count += 1

        return count

    def get_4D_data(self,paths,img_shape):
        re_array = []

        for path in paths:
            img = cv2.imread(path)
            if img is None:
                print("read failed:",path)
            else:
                img = cv2.resize(img,(img_shape[1],img_shape[0]))
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32)
                img /= 255
                re_array.append(img)

        re_array = np.array(re_array)

        return re_array



    def __get_label_dict(self,img_dir):
        label_dict = dict()
        count = 0
        for obj in os.scandir(img_dir):
            if obj.is_dir():
                label_dict[obj.name] = count
                count += 1
        if count == 0:
            print("No dir in the ",img_dir)
            return None
        else:
            return label_dict

    def __get_paths_labels(self,img_dir,label_dict):
        #----var
        img_format = {'png', 'jpg', 'bmp'}
        re_paths = list()
        re_labels = list()

        #----read dirs
        dirs = [obj.path for obj in os.scandir(img_dir) if obj.is_dir()]
        if len(dirs) == 0:
            print("No dirs in the ",img_dir)
        else:
            #-----read paths of each dir
            for dir_path in dirs:
                path_temp = [file.path for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format]
                if len(path_temp) == 0:
                    print("No images in the ",dir_path)
                else:
                    #----get the label number from class name
                    label_num = dir_path.split("\\")[-1]
                    label_num = label_dict[label_num]
                    #----create the label array
                    label_temp = np.ones(len(path_temp),dtype=np.int32) * label_num

                    #----collect paths and labels
                    re_paths.extend(path_temp)
                    re_labels.extend(label_temp)

            #----list to numpy array
            re_paths = np.array(re_paths)
            re_labels = np.array(re_labels)

            #----shuffle
            indice = np.random.permutation(re_paths.shape[0])
            re_paths = re_paths[indice]
            re_labels = re_labels[indice]

        return re_paths, re_labels

    #----models
    def resnet_block(self,input_x, k_size=3,filters=32):
        net = tf.layers.conv2d(
            inputs=input_x,
            filters = filters,
            kernel_size=[k_size,k_size],
            kernel_regularizer=tf.keras.regularizers.l2(0.1),
            padding="same",
            activation=tf.nn.relu
        )
        net = tf.layers.conv2d(
            inputs=net,
            filters=filters,
            kernel_size=[k_size, k_size],
            kernel_regularizer=tf.keras.regularizers.l2(0.1),
            padding="same",
            activation=tf.nn.relu
        )

        net_1 = tf.layers.conv2d(
            inputs=input_x,
            filters=filters,
            kernel_size=[k_size, k_size],
            kernel_regularizer=tf.keras.regularizers.l2(0.1),
            padding="same",
            activation=tf.nn.relu
        )

        add = tf.add(net,net_1)

        add_result = tf.nn.relu(add)

        return add_result


    def simple_resnet(self,tf_input,tf_keep_prob,class_num):
        net = self.resnet_block(tf_input,k_size=3,filters=16)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2,2], strides=2)
        print("pool_1 shape:",net.shape)

        net = self.resnet_block(net, k_size=3, filters=32)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        print("pool_2 shape:", net.shape)

        net = self.resnet_block(net, k_size=3, filters=48)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        print("pool_3 shape:", net.shape)

        net = self.resnet_block(net, k_size=3, filters=64)
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
        print("pool_4 shape:", net.shape)

        #----flatten
        net = tf.layers.flatten(net)
        print("flatten shape:",net.shape)

        #----dropout
        net = tf.nn.dropout(net,keep_prob=tf_keep_prob)

        #----FC
        net = tf.layers.dense(inputs=net,units=128,activation=tf.nn.relu)
        print("FC shape:",net.shape)

        #----output
        output = tf.layers.dense(inputs=net,units=class_num,activation=None)
        print("output shape:",output.shape)

        return output










if __name__ == "__main__":
    train_img_dir = r"D:\dataset\mnist\train"
    test_img_dir = r"D:\dataset\mnist\val"
    label_dict = None

    para_dict = {"train_img_dir":train_img_dir,"test_img_dir":test_img_dir,"label_dict":label_dict}

    cls = Classification(para_dict)

    model_shape = [None,28,28,3]
    infer_method = "simple_resnet"
    loss_method = "cross_entropy"
    opti_method = "adam"
    learning_rate = 1e-4
    save_dir = r"D:\code\model_saver\test"

    para_dict = {"model_shape":model_shape,"infer_method":infer_method,"loss_method":loss_method,
                 "opti_method":opti_method,'learning_rate':learning_rate,"save_dir":save_dir}
    cls.model_init(para_dict)

    epochs = 100
    GPU_ratio = None
    batch_size = 32
    para_dict = {'epochs':epochs, "GPU_ratio":GPU_ratio, "batch_size":batch_size}

    cls.train(para_dict)
