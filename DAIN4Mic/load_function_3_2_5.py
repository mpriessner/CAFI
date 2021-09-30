
def create_3D_image(img, x_dim, y_dim):
# creates 3D image with 3 times the same values for RGB because the NN was generated for normal rgb images dim(3,x,y)
  image_3D = np.zeros((3,x_dim,y_dim))
  image_3D[0] = img
  image_3D[1] = img
  image_3D[2] = img
  return image_3D

# create supporting functions
def get_file_list_Z(filepath):
  # get path of files into a list
  folder_path = filepath
  flist = os.listdir(folder_path)
  file_list = []
  for i in flist:
    file_path = os.path.join(folder_path, i)
    file_list.append(file_path)
  return file_list
# filepath = "/content/drive/My Drive/Colab Notebooks/Resulution_enhancement/Test_image"
# filepath_list = get_file_list_T(filepath)

def convert(img, target_type_min, target_type_max, target_type):
  # this function converts images from float32 to unit8 
    imin = img.min()
    imax = img.max()
    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img

def create_folder_Z(root, folder_name):
  os.chdir(root)
  destination = os.path.join(root,folder_name)
  # # remove folder if there was one before
  # # create new folder
  if ticker_2 == "yes":
    if os.path.exists(destination):
      shutil.rmtree(destination)
      os.mkdir(destination)
    else:   
      os.mkdir(destination)
  else:
    if os.path.exists(destination)==False:
      os.mkdir(destination)

  os.chdir(destination)  
  sub_folder = "sequences"
  sequence_path = os.path.join(destination, sub_folder)
  if os.path.exists(sequence_path)==False:
    os.mkdir(sequence_path)
  os.chdir(sequence_path)
  return destination, sequence_path
# destination, sequence_path = create_folder_Z(root, folder_name)

#didn't need to change anything because it is the same algorithm as for T
def run_create_training_Z(filepath, destination, k):
  # get file-path-list
  filepath_list = get_file_list_Z(filepath)

  # create virtual number of files
  for i in range(0,len(filepath_list)):
    print(filepath_list[i-1])
    # list_number_of_files.append(i)
    new_folder_name = ("%02d" % (k) + "-" + "%05d" % (i+1))
    os.mkdir(new_folder_name)
    txt_name_log = open(destination + "/name_log.txt", "a")
    txt_name_log.write("{}, {}\n".format(new_folder_name, filepath_list[i-1]), )
    txt_name_log.close()
    img = AICSImage(filepath_list[i])
    z_dim = img.shape[2]
    #dim for later for generating the image_3d files
    x_dim = img.shape[-2]
    y_dim = img.shape[-1]

    #because the naming of the folders in the original also starts with 1 and -1 because i take always 3 so need to stop 2 before the end
    for j in range(0,z_dim-1):    
      #create new directory-path
      new_folder_path_1 = os.path.join(sequence_path, new_folder_name)
      os.chdir(new_folder_path_1)
      file_folder = "%04d" % (j+1)
      os.mkdir(file_folder)
      #add new folder to txt-file
      decision_train_test = random.random()
      if decision_train_test < split_training_test:
        txt_file_train = open(destination + "/tri_trainlist.txt", "a")
        txt_file_train.write("{}/{}\n".format(new_folder_name,file_folder))
        txt_file_train.close()
      else:
        txt_file_test = open(destination + "/tri_testlist.txt", "a")
        txt_file_test.write("{}/{}\n".format(new_folder_name,file_folder))
        txt_file_test.close()
      new_folder_path_2 = os.path.join(new_folder_path_1, file_folder)
      os.chdir(new_folder_path_2)
      #here put the image pngs into the folder (instead of creating the folder)
      #convert image to unit8 otherwise warning
      img_1 = img.get_image_data("YX", S=0, T=0, C=j-1, Z=0)
      img_1 = create_3D_image(img_1, x_dim, y_dim)
      img_1 = convert(img_1, 0, 255, np.uint8)
      img_2 = img.get_image_data("YX", S=0, T=0, C=j, Z=0)
      img_2 = create_3D_image(img_2, x_dim, y_dim)
      img_2 = convert(img_2, 0, 255, np.uint8)
      img_3 = img.get_image_data("YX", S=0, T=0, C=j+1, Z=0)
      img_3 = create_3D_image(img_3, x_dim, y_dim)
      img_3 = convert(img_3, 0, 255, np.uint8)

      # saving images as PNG
      with png_writer.PngWriter("im1.png") as writer2:
        writer2.save(img_1)
      with png_writer.PngWriter("im2.png") as writer2:
        writer2.save(img_2)
      with png_writer.PngWriter("im3.png") as writer2:
        writer2.save(img_3)
    os.chdir(sequence_path)

# run_create_training_Z(filepath, destination)