import os,cv2,random
import numpy as np
import dlib#I use the version of 19.19.0. Use pip install dlib

print("dlib version: ",dlib.__version__)

def detect_mouth(img,detector,predictor):
    x_min = None
    x_max = None
    y_min = None
    y_max = None
    size = None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector(img_rgb, 0)
    #print("len of faces = ",len(faces))
    if len(faces):
        for coor in (faces):#coordinate format:[(left,top), (right,bottom)]
            x = list()
            y = list()
            height = coor.bottom() - coor.top()
            width = coor.right() - coor.left()
            # shape = predictor(img_gray, d)
            landmark = predictor(img_rgb, coor)

            #----get the mouth part
            for i in range(48, 68):
                x.append(landmark.part(i).x)
                y.append(landmark.part(i).y)

            y_max = np.minimum(max(y) + height // 3, img_rgb.shape[0])
            y_min = np.maximum(min(y) - height // 3, 0)
            x_max = np.minimum(max(x) + width // 3, img_rgb.shape[1])
            x_min = np.maximum(min(x) - width // 3, 0)

            size = ((x_max-x_min),(y_max-y_min))#(width,height)

    return x_min, x_max, y_min, y_max, size


def mask_wearing(root_dir,output_dir=None,dataset_range=None):
    # ----var
    mask_img_dir = ".\mask_img"
    img_format = {'png','jpg'}
    paths = list()
    detect_flag = False
    # ----read mask png images
    mask_files = [file.path for file in os.scandir(mask_img_dir) if file.name.split(".")[-1] == 'png']
    len_mask = len(mask_files)
    if len_mask == 0:
        print("Error: no face mask PNG images in  ", mask_img_dir)
    else:
        # ----collect all dirs
        dirs = [obj.path for obj in os.scandir(root_dir) if obj.is_dir()]

        if len(dirs) == 0:  # no sub folders
            print("No sub folders in ", root_dir)
        else:
            # ----dataset range
            dirs.sort()
            print("Total class number: ", len(dirs))
            if dataset_range is not None:
                dirs = dirs[dataset_range[0]:dataset_range[1]]
                print("Working classes: {} to {}".format(dataset_range[0], dataset_range[1]))
            else:
                print("Working classes:All")

            # ----collect all image paths from all sub-folders
            for dir_path in dirs:
                files_temp = [file.path for file in os.scandir(dir_path) if file.name.split(".")[-1] in img_format]
                if len(files_temp) == 0:
                    print("no files under {}".format(dir_path))
                else:
                    paths.extend(files_temp)

            if len(paths) > 0:
                detect_flag = True

            # ----execution if detect_flag is True
            if detect_flag is True:
                # ----create dir for img
                if output_dir is None:
                    output_dir = os.path.join(root_dir, "img_wear_mask")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # ----create save_dir for each sub dir
                sub_dir_dict = dict()
                for dir_path in dirs:
                    dir_name = dir_path.split("\\")[-1]
                    new_dir = os.path.join(output_dir, dir_name)
                    if not os.path.exists(new_dir):
                        os.makedirs(new_dir)
                    sub_dir_dict[dir_name] = new_dir
                # ----face detection init
                detector = dlib.get_frontal_face_detector()
                predictor = dlib.shape_predictor('src/models/shape_predictor_68_face_landmarks.dat')

                # ----mouth detection and wear mask on faces
                for path in paths:
                    img = cv2.imread(path)
                    if img is None:
                        print("read failed:{}".format(path))
                    else:
                        x_min, x_max, y_min, y_max, size = detect_mouth(img, detector, predictor)
                        if size is not None:
                            # ----random selection of face mask
                            which = random.randint(0, len_mask - 1)
                            item_name = mask_files[which]

                            # ----face mask process
                            item_img = cv2.imread(item_name, cv2.IMREAD_UNCHANGED)
                            item_img = cv2.resize(item_img, size)
                            item_img_bgr = item_img[:, :, :3]
                            item_alpha_ch = item_img[:, :, 3]
                            _, item_mask = cv2.threshold(item_alpha_ch, 220, 255, cv2.THRESH_BINARY)
                            img_item = cv2.bitwise_and(item_img_bgr, item_img_bgr, mask=item_mask)

                            # ----mouth part process
                            roi = img[y_min:y_min + size[1], x_min:x_min + size[0]]
                            item_mask_inv = cv2.bitwise_not(item_mask)
                            roi = cv2.bitwise_and(roi, roi, mask=item_mask_inv)

                            # ----addition of mouth and face mask
                            dst = cv2.add(roi, img_item)
                            img[y_min: y_min + size[1], x_min:x_min + size[0]] = dst

                            # -----save img
                            splits = path.split("\\")
                            new_filename = os.path.join(sub_dir_dict[splits[-2]], splits[-1])
                            cv2.imwrite(new_filename, img)


    print("It's done")



if __name__ == "__main__":
    root_dir = r"C:\Users\ztyam\OneDrive - University of Florida\0_UF Class\2021 Spring\03_Pattern Recognition\Project\dataSet\test_database_3_10000"
    output_dir = r"C:\Users\ztyam\OneDrive - University of Florida\0_UF Class\2021 Spring\03_Pattern Recognition\Project\dataSet\with_mask"
    dataset_range = [0, 10000]
    mask_wearing(root_dir,output_dir=output_dir,dataset_range=dataset_range)