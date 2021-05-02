import tensorflow
import os

#----tensorflow version check
if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
print("Tensorflow version: ",tf.__version__)


def get_pb_all_nodes(pb_path,is_save = False):
    '''
    :param pb_path:
    :param is_save: when True, save all nodes in a txt file
    :return: None
    '''
    txt_filename = r"get_all_pb_names.txt"

    if os.path.exists(pb_path):
        config = tf.ConfigProto(log_device_placement=True,
                                allow_soft_placement=True,  # 允許當找不到設備時自動轉換成有支援的設備
                                )
        config.gpu_options.per_process_gpu_memory_fraction = 0.6
        with tf.Session(config = config) as sess:
            with tf.gfile.FastGFile(pb_path, 'rb') as f:
                frozen_graph_def = tf.GraphDef()
                frozen_graph_def.ParseFromString(f.read())
                sess.graph.as_default()
                tf.import_graph_def(frozen_graph_def, name='')

                tensor_name_list = [tensor.name for tensor in tf.get_default_graph().as_graph_def().node]
                for tensor_name in tensor_name_list:
                    print(tensor_name, '\n')
                if is_save:
                    with open(txt_filename, "w") as f:
                        for tensor_name in tensor_name_list:
                            f.write(tensor_name)
                            f.write("\n")
                    print("txt is saved in ",os.path.join(os.getcwd(),txt_filename))

    else:
        print("{} doesn't exist".format(pb_path))

if __name__ == "__main__":
    pb_path = r"C:\Users\ztyam\OneDrive - University of Florida\0_UF Class\2021 Spring\03_Pattern Recognition\Project\VGGFace2-20180402-114759\20180402-114759.pb"
    get_pb_all_nodes(pb_path,is_save=True)
