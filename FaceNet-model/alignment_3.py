from face_alignment import img_alignment

if __name__ == "__main__":
    #----alignment
    root_dir = r"C:\Users\ztyam\3D Objects\DataSet\CASIA-WebFace"
    output_dir = r"C:\Users\ztyam\3D Objects\DataSet\CASIA-WebFace_aligned"
    margin = 20
    GPU_ratio = 0.3
    img_show = False
    dataset_range = [6000, 10575]
    img_alignment(root_dir, output_dir, margin=margin, GPU_ratio=GPU_ratio, img_show=img_show,dataset_range=dataset_range)