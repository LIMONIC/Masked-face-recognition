import os,math,cv2,shutil
import numpy as np
import tensorflow

#----tensorflow version check
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.compat.v1.gfile as gfile
print("Tensorflow version: ",tf.__version__)


img_format = {'png', 'jpg', 'bmp'}


def model_restore_from_pb(pb_path,node_dict,GPU_ratio=None):
    tf_dict = dict()
    with tf.Graph().as_default():
        config = tf.ConfigProto(log_device_placement=True,
                                allow_soft_placement=True,
                                )
        if GPU_ratio is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio
        sess = tf.Session(config=config)
        with gfile.FastGFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()

            for node in graph_def.node:
                if node.op == 'RefSwitch':
                    node.op = 'Switch'
                    for index in range(len(node.input)):
                        if 'moving_' in node.input[index]:
                            node.input[index] = node.input[index] + '/read'
                elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr: del node.attr['use_locking']

            tf.import_graph_def(graph_def, name='')

        sess.run(tf.global_variables_initializer())
        for key,value in node_dict.items():
            node = sess.graph.get_tensor_by_name(value)
            tf_dict[key] = node
        return sess,tf_dict

def get_embeddings(sess,paths,tf_dict,batch_size=128):
    #----
    len_path = len(paths)
    tf_input = tf_dict['input']
    tf_phase_train = tf_dict['phase_train']
    tf_embeddings = tf_dict['embeddings']

    feed_dict = {tf_phase_train: False}
    if 'keep_prob' in tf_dict.keys():
        tf_keep_prob = tf_dict['keep_prob']
        feed_dict[tf_keep_prob] = 1.0

    model_shape = tf_input.shape
    # model_shape = [None,160,160,3]
    print("tf_input shape:",model_shape)

    #----
    ites = math.ceil(len_path / batch_size)
    embeddings = np.zeros([len_path, tf_embeddings.shape[-1]], dtype=np.float32)
    for idx in range(ites):
        num_start = idx * batch_size
        num_end = np.minimum(num_start + batch_size, len_path)
        # ----read batch data
        batch_dim = [num_end - num_start] #normally num_end - num_start = batch_size, but not in the last batch
        batch_dim.extend(model_shape[1:])
        batch_data = np.zeros(batch_dim, dtype=np.float32)
        for idx_path, path in enumerate(paths[num_start:num_end]):
            img = cv2.imread(path)
            if img is None:
                print("Read failed:", path)
            else:
                img = cv2.resize(img, (model_shape[2], model_shape[1]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                batch_data[idx_path] = img
        batch_data /= 255  # norm
        feed_dict[tf_input] = batch_data
        embeddings[num_start:num_end] = sess.run(tf_embeddings, feed_dict=feed_dict)


    return embeddings

def evaluation(test_dir,face_databse_dir,pb_path,GPU_ratio=None):
    #----var
    paths_test = list()
    node_dict = {'input': 'input:0',
                 'phase_train': 'phase_train:0',
                 'embeddings': 'embeddings:0',
                 'keep_prob': 'keep_prob:0'
                 }
    batch_size = 128
    arg_dis = list()
    dis_list = list()
    count_o = 0
    count_unknown = 0
    threshold = 0.8

    #----get test images
    for dir_name, subdir_names, filenames in os.walk(test_dir):
        if len(filenames):
            for file in filenames:
                if file[-3:] in img_format:
                    paths_test.append(os.path.join(dir_name,file))

    if len(paths_test) == 0:
        print("No images in ",test_dir)
        raise ValueError

    #----get images of face_databse_dir
    paths_ref = [file.path for file in os.scandir(face_databse_dir) if file.name[-3:] in img_format]
    len_path_ref = len(paths_ref)
    if len_path_ref == 0:
        print("No images in ", face_databse_dir)
        raise ValueError

    #----model init
    sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=GPU_ratio)
    tf_embeddings = tf_dict['embeddings']

    #----tf setting for calculating distance
    with tf.Graph().as_default():
        tf_tar = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape[-1])
        tf_ref = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape)
        tf_dis = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf_ref, tf_tar)), axis=1))
        # ----GPU setting
        config = tf.ConfigProto(log_device_placement=True,
                                allow_soft_placement=True,
                                )
        config.gpu_options.allow_growth = True
        sess_cal = tf.Session(config=config)
        sess_cal.run(tf.global_variables_initializer())

    #----get embeddings
    embed_ref = get_embeddings(sess, paths_ref, tf_dict, batch_size=batch_size)
    embed_tar = get_embeddings(sess, paths_test, tf_dict, batch_size=batch_size)
    print("embed_ref shape: ", embed_ref.shape)
    print("embed_tar shape: ", embed_tar.shape)

    #----calculate distance and get the minimum index
    feed_dict_2 = {tf_ref: embed_ref}
    for idx, embedding in enumerate(embed_tar):
        feed_dict_2[tf_tar] = embedding
        distance = sess_cal.run(tf_dis, feed_dict=feed_dict_2)
        arg_temp = np.argsort(distance)[0]
        arg_dis.append(arg_temp)
        dis_list.append(distance[arg_temp])

    for idx, path in enumerate(paths_test):
        answer = path.split("\\")[-1].split("_")[0]

        arg = arg_dis[idx]
        prediction = paths_ref[arg].split("\\")[-1].split("_")[0]

        dis = dis_list[idx]

        if dis < threshold:
            if prediction == answer:
                count_o += 1
            else:
                print("\nIncorrect:",path)
                print("prediction:{}, answer:{}".format(prediction,answer))
        else:
            count_unknown += 1
            print("\nunknown:", path)
            print("prediction:{}, answer:{}, distance:{}".format(prediction,answer,dis))


    #----statistics
    print("accuracy: ", count_o / len(paths_test))
    print("unknown: ", count_unknown / len(paths_test))

if __name__ == "__main__":
    #----face matching evaluation
    root_dir = r"C:\Users\ztyam\3D Objects\DataSet\test_database_3_10000\with_mask"
    face_databse_dir = r"C:\Users\ztyam\3D Objects\DataSet\test_database_3_10000\no_mask"
    # pb_path = r"G:\我的雲端硬碟\Python\Code\model_saver\face_reg_models\FLW_0.98\pb_model.pb"
    # pb_path = r"G:\我的雲端硬碟\Python\Code\model_saver\face_reg_models\FLW_0.9918\pb_model.pb"
    pb_path = r"C:\Users\ztyam\OneDrive - University of Florida\0_UF Class\2021 Spring\03_Pattern Recognition\Project\CASIA-WebFace-20180408-102900\20180408-102900.pb"
    evaluation(root_dir, face_databse_dir, pb_path, GPU_ratio=None)