import os
base_dir = os.path.dirname(os.path.realpath(__file__))

def get_test_data(name='bus'):
    if name == 'bus':
        img_fp = os.path.join(base_dir, 'samples', 'bus.jpg')
    elif name == 'zidane':
        img_fp = os.path.join(base_dir, 'samples', 'zidane.jpg')
    return img_fp

#def write_data_file(model_name):
#    target_dst_dir = os.path.join(base_dir, 'cfgs', model_name)
#    data_f_src = open(os.path.join(target_dst_dir, 'road_src.data'), 'r', encoding='utf8')
#    data_f_dst = open(os.path.join(target_dst_dir, 'road.data'), 'w', encoding='utf8')

#    for line in data_f_src:
#        if 'classes = ' in line:
#            data_f_dst.write(line)
#        elif 'names = ' in line:
#            data_f_dst.write('names = ' + os.path.abspath(os.path.join(target_dst_dir, 'road.names')))
#    data_f_src.close()
#    data_f_dst.close()


def get_test_params():
    """
    name: {'yolov4', 'yolov4_tiny}
    """
    cfg_fp = os.path.join(base_dir, 'cfgs', 'yolov4_tiny', 'yolov4-tiny.cfg')
    weights_fp = os.path.join(base_dir, 'cfgs', 'yolov4_tiny', 'yolov4-tiny.weights')
    names_fp = os.path.join(base_dir, 'yolov4-tiny', 'coco.names')
    return cfg_fp, data_fp, weights_fp

