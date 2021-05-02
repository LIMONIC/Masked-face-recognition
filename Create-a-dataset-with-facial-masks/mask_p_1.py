from mask_wear_class import mask_wearing

if __name__ == "__main__":
    root_dir = r"C:\Users\ztyam\3D Objects\DataSet\CASIA-WebFace_aligned"
    output_dir = r"C:\Users\ztyam\3D Objects\DataSet\CASIA-WebFace_aligned(masked)"
    dataset_range = [3000, 6000]
    mask_wearing(root_dir,output_dir=output_dir,dataset_range=dataset_range)