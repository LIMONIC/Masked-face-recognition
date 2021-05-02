from tools import img_removal_by_embed

if __name__ == "__main__":
    root_dir = r"C:\Users\ztyam\3D Objects\DataSet\myPhoto_aligned"
    output_dir = r"C:\Users\ztyam\3D Objects\DataSet\myPhoto_aligned_mislabeled"
    pb_path = r"C:\Users\ztyam\OneDrive - University of Florida\0_UF Class\2021 Spring\03_Pattern Recognition\Project\VGGFace2-20180402-114759\20180402-114759.pb"
    node_dict = {'input': 'input:0',
                 'phase_train': 'phase_train:0',
                 'embeddings': 'embeddings:0',
                 }
    dataset_range = [0,1000]
    img_removal_by_embed(root_dir, output_dir, pb_path, node_dict, threshold=1.1, type='move', GPU_ratio=0.25,
                         dataset_range=dataset_range)

    # ----check_path_length
    # root_dir = r"D:\CASIA\CASIA-WebFace_aligned"
    # output_dir = r"D:\CASIA\mislabeled"
    # check_path_length(root_dir, output_dir, threshold=3)

    #----delete_dir_with_no_img
    # root_dir = r"D:\CASIA\CASIA-WebFace_aligned"
    # delete_dir_with_no_img(root_dir)


