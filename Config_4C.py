# this is the config file for the training, test and defining of Foreground_3C

class Config_4C():
  trans_resize_size = 224


  #################################################
  ############## Training #########################
  #################################################

  # lr history: 1e-6 -> 1e-7
  lr = 1e-6
  batch_num = 16
  worker_num = 8
  save_epochs = 100
  epoch_num = 100
  first_train = False
  save_result_png = False
  # whether only use the sequences for testing, without training set
  sequence_only_test = True

  # model path
  box_weight_path = "./weight/model_4C_box.pkl"
  mask_weight_path = "./weight/model_4C_mask.pkl"

  ####################################################################################
  # local saliency dataset path

  #root_image_path = "/home/smj/DataSet/Saliency/MSRA_TOTAL/image"
  #root_label_path = "/home/smj/DataSet/Saliency/MSRA_TOTAL/label"
  root_image_path = "/home/smj/DataSet/Saliency/DAVIS2017_foreground_total/image"
  root_label_path = "/home/smj/DataSet/Saliency/DAVIS2017_foreground_total/label"
  #root_image_path = "/home/smj/DataSet/Saliency/DAVIS2017_foreground_test/image"
  #root_label_path = "/home/smj/DataSet/Saliency/DAVIS2017_foreground_test/label"
  #root_image_path = "/home/smj/DataSet/Saliency/COCO2014_MSRA/image"
  #root_label_path = "/home/smj/DataSet/Saliency/COCO2014_MSRA/label"

  #root_prediction_path = "/home/smj/DataSet/Saliency/DAVIS2017_foreground_test/prediction"
  root_prediction_path = "/home/smj/DataSet/Saliency/DAVIS2017_foreground_total/prediction"

  ###################################################################################
  # system saliency server dataset path

  #root_image_path = "/Data_HDD/smj_data/Saliency/MSRA_TOTAL/image"
  #root_label_path = "/Data_HDD/smj_data/Saliency/MSRA_TOTAL/label"
  #root_image_path = "/Data_HDD/smj_data/Saliency/DAVIS2017_foreground_total/image"
  #root_label_path = "/Data_HDD/smj_data/Saliency/DAVIS2017_foreground_total/label"
  #root_image_path = "/Data_HDD/smj_data/Saliency/DAVIS2017_foreground_test/image"
  #root_label_path = "/Data_HDD/smj_data/Saliency/DAVIS2017_foreground_test/label"
  #root_image_path = "/Data_HDD/smj_data/Saliency/COCO2014_MSRA/image"
  #root_label_path = "/Data_HDD/smj_data/Saliency/COCO2014_MSRA/label"

  #root_prediction_path = "/Data_HDD/smj_data/Saliency/DAVIS2017_foreground_test/prediction"
  #root_prediction_path = "/Data_HDD/smj_data/Saliency/DAVIS2017_foreground_total/prediction"

  ###################################################################################
  # our saliency server dataset path

  #root_image_path = "/data/smj_data/DataSet/Saliency/MSRA_TOTAL/image"
  #root_label_path = "/data/smj_data/DataSet/Saliency/MSRA_TOTAL/label"
  #root_image_path = "/data/smj_data/DataSet/Saliency/DAVIS2017_foreground_total/image"
  #root_label_path = "/data/smj_data/DataSet/Saliency/DAVIS2017_foreground_total/label"
  #root_image_path = "/data/smj_data/DataSet/Saliency/DAVIS2017_foreground_test/image"
  #root_label_path = "/data/smj_data/DataSet/Saliency/DAVIS2017_foreground_test/label"
  #root_image_path = "/data/smj_data/DataSet/Saliency/COCO2014_MSRA/image"
  #root_label_path = "/data/smj_data/DataSet/Saliency/COCO2014_MSRA/label"

  #root_prediction_path = "/data/smj_data/DataSet/Saliency/DAVIS2017_foreground_test/prediction"
  #root_prediction_path = "/data/smj_data/DataSet/Saliency/DAVIS2017_foreground_total/prediction"







  #################################################
  ############## Testing ##########################
  #################################################

  # path to save the prediction result
  result_path = "./result_DAVIS2017"

  test_root_images_path = "/home/smj/DataSet/DAVIS2017_targets/JPEGImages/480p"
  test_root_labels_mask_path = "/home/smj/DataSet/DAVIS2017_targets/Annotations/480p"
  test_root_labels_box_small_path = "/home/smj/DataSet/DAVIS2017_targets/Annotations/480p_box"
  test_root_labels_box_margin_path = "/home/smj/DataSet/DAVIS2017_targets/Annotations/480p_box_margin"


  # test sequences list for DAVIS2017

  '''
  test_sequence_list = ["paragliding-launch_0",
                        "paragliding-launch_1",
                        "paragliding-launch_2"]
  '''

  test_sequence_list = ["bike-packing_0",
                        "bike-packing_1",
                        "blackswan_0",
                        "bmx-trees_0",
                        "bmx-trees_1",
                        "breakdance_0",
                        "camel_0",
                        "car-roundabout_0",
                        "car-shadow_0",
                        "cows_0",
                        "dance-twirl_0",
                        "dog_0",
                        "dogs-jump_0",
                        "dogs-jump_1",
                        "dogs-jump_2",
                        "drift-chicane_0",
                        "drift-straight_0",
                        "goat_0",
                        "gold-fish_0",
                        "gold-fish_1",
                        "gold-fish_2",
                        "gold-fish_3",
                        "gold-fish_4",
                        "horsejump-high_0",
                        "horsejump-high_1",
                        "india_0",
                        "india_1",
                        "india_2",
                        "judo_0",
                        "judo_1",
                        "kite-surf_0",
                        "kite-surf_1",
                        "kite-surf_2",
                        "lab-coat_0",
                        "lab-coat_1",
                        "lab-coat_2",
                        "lab-coat_3",
                        "lab-coat_4",
                        "libby_0",
                        "loading_0",
                        "loading_1",
                        "loading_2",
                        "mbike-trick_0",
                        "mbike-trick_1",
                        "motocross-jump_0",
                        "motocross-jump_1",
                        "paragliding-launch_0",
                        "paragliding-launch_1",
                        "paragliding-launch_2",
                        "parkour_0",
                        "pigs_0",
                        "pigs_1",
                        "pigs_2",
                        "scooter-black_0",
                        "scooter-black_1",
                        "shooting_0",
                        "shooting_1",
                        "shooting_2",
                        "soapbox_0",
                        "soapbox_1",
                        "soapbox_2"
                        ]

