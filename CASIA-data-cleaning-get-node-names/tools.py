import os,math,cv2,shutil
import numpy as np
import tensorflow

if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.compat.v1.gfile as gfile

print("Tensorflow version: ",tf.__version__)


img_format = {'png','jpg','bmp'}


def model_restore_from_pb(pb_path, node_dict,GPU_ratio=None):
    tf_dict = dict()
    with tf.Graph().as_default():
        config = tf.ConfigProto(log_device_placement=True,  # 印出目前的運算是使用CPU或GPU
                                allow_soft_placement=True,  # 當設備不存在時允許tf選擇一个存在且可用的設備來繼續執行程式
                                )
        if GPU_ratio is None:
            config.gpu_options.allow_growth = True  # 依照程式執行所需要的資料來自動調整
        else:
            config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio  # 手動限制GPU資源的使用
        sess = tf.Session(config=config)
        with gfile.FastGFile(pb_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')  # 匯入計算圖
        sess.run(tf.global_variables_initializer())
        for key, value in node_dict.items():
            node = sess.graph.get_tensor_by_name(value)
            tf_dict[key] = node
        return sess, tf_dict

def img_removal_by_embed(root_dir,output_dir,pb_path,node_dict,threshold=0.7,type='copy',GPU_ratio=None, dataset_range=None):
    # ----var
    img_format = {"png", 'jpg', 'bmp'}
    batch_size = 64

    # ----collect all folders
    dirs = [obj.path for obj in os.scandir(root_dir) if obj.is_dir()]
    if len(dirs) == 0:
        print("No sub-dirs in ", root_dir)
    else:
        #----dataset range
        if dataset_range is not None:
            dirs = dirs[dataset_range[0]:dataset_range[1]]

        # ----model init
        sess, tf_dict = model_restore_from_pb(pb_path, node_dict, GPU_ratio=GPU_ratio)
        tf_input = tf_dict['input']
        tf_phase_train = tf_dict['phase_train']
        tf_embeddings = tf_dict['embeddings']
        model_shape = [None, 160, 160, 3]
        feed_dict = {tf_phase_train: False}

        # ----tf setting for calculating distance
        with tf.Graph().as_default():
            tf_tar = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape[-1])
            tf_ref = tf.placeholder(dtype=tf.float32, shape=tf_embeddings.shape)
            tf_dis = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(tf_ref, tf_tar)), axis=1))
            # ----GPU setting
            config = tf.ConfigProto(log_device_placement=True,
                                    allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                                    )
            config.gpu_options.allow_growth = True
            sess_cal = tf.Session(config=config)
            sess_cal.run(tf.global_variables_initializer())



        #----process each folder
        for dir_path in dirs:
            paths = [file.path for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format]
            len_path = len(paths)
            if len_path == 0:
                print("No images in ",dir_path)
            else:
                # ----create the sub folder in the output folder
                save_dir = os.path.join(output_dir, dir_path.split("\\")[-1])
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                # ----calculate embeddings
                ites = math.ceil(len_path / batch_size)
                embeddings = np.zeros([len_path, tf_embeddings.shape[-1]], dtype=np.float32)
                for idx in range(ites):
                    num_start = idx * batch_size
                    num_end = np.minimum(num_start + batch_size, len_path)
                    # ----read batch data
                    batch_dim = [num_end - num_start]#[64]
                    batch_dim.extend(model_shape[1:])#[64,160, 160, 3]
                    batch_data = np.zeros(batch_dim, dtype=np.float32)
                    for idx_path,path in enumerate(paths[num_start:num_end]):
                        img = cv2.imread(path)
                        if img is None:
                            print("Read failed:",path)
                        else:
                            img = cv2.resize(img, (model_shape[2], model_shape[1]))
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            batch_data[idx_path] = img
                    batch_data /= 255  # norm
                    feed_dict[tf_input] = batch_data
                    embeddings[num_start:num_end] = sess.run(tf_embeddings, feed_dict=feed_dict)

                # ----calculate ave distance of each image
                feed_dict_2 = {tf_ref: embeddings}
                ave_dis = np.zeros(embeddings.shape[0], dtype=np.float32)
                for idx, embedding in enumerate(embeddings):
                    feed_dict_2[tf_tar] = embedding
                    distance = sess_cal.run(tf_dis, feed_dict=feed_dict_2)
                    ave_dis[idx] = np.sum(distance) / (embeddings.shape[0] - 1)
                # ----remove or copy images
                for idx,path in enumerate(paths):
                    if ave_dis[idx] > threshold:
                        print("path:{}, ave_distance:{}".format(path,ave_dis[idx]))
                        if type == "copy":
                            save_path = os.path.join(save_dir,path.split("\\")[-1])
                            shutil.copy(path,save_path)
                        elif type == "move":
                            save_path = os.path.join(save_dir,path.split("\\")[-1])
                            shutil.move(path,save_path)

def check_path_length(root_dir,output_dir,threshold=5):
    # ----var
    img_format = {"png", 'jpg'}

    # ----collect all dirs
    dirs = [obj.path for obj in os.scandir(root_dir) if obj.is_dir()]

    if len(dirs) == 0:
        print("No dirs in ",root_dir)
    else:
        # ----process each dir
        for dir_path in dirs:
            leng = len([file.name for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format])
            if leng <= threshold:
                corresponding_dir = os.path.join(output_dir,dir_path.split("\\")[-1])
                leng_corre = len([file.name for file in os.scandir(corresponding_dir) if file.name.split(".")[-1] in img_format])
                print("dir name:{}, quantity of origin:{}, quantity of removal:{}".format(dir_path.split("\\")[-1],leng,leng_corre))

def delete_dir_with_no_img(root_dir):
    # ----collect all dirs
    dirs = [obj.path for obj in os.scandir(root_dir) if obj.is_dir()]
    if len(dirs) == 0:
        print("No dirs in ",root_dir)
    else:
        # ----process each dir
        for dir_path in dirs:
            leng = len([file.name for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format])
            if leng == 0:
                shutil.rmtree(dir_path)
                print("Deleted:",dir_path)







if __name__ == "__main__":
    root_dir = r"C:\Users\ztyam\3D Objects\DataSet\CASIA-WebFace_aligned"
    output_dir = r"C:\Users\ztyam\3D Objects\DataSet\CASIA-WebFace-mislabeled"
    pb_path = r"C:\Users\ztyam\OneDrive - University of Florida\0_UF Class\2021 Spring\03_Pattern Recognition\Project\VGGFace2-20180402-114759\20180402-114759.pb"
    node_dict = {'input': 'input:0',
                 'phase_train': 'phase_train:0',
                 'embeddings': 'embeddings:0',
                 }
    dataset_range = [0,100]
    img_removal_by_embed(root_dir, output_dir, pb_path, node_dict, threshold=1.25, type='move', GPU_ratio=0.25,
                         dataset_range=dataset_range)

    # ----check_path_length
    # root_dir = r"D:\CASIA\CASIA-WebFace_aligned"
    # output_dir = r"D:\CASIA\mislabeled"
    # check_path_length(root_dir, output_dir, threshold=3)

    #----delete_dir_with_no_img
    # root_dir = r"D:\CASIA\CASIA-WebFace_aligned"
    # delete_dir_with_no_img(root_dir)


