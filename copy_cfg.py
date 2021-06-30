import os
import shutil

# model_name = 'yolov4'
model_name = 'yolov4_tiny'

target_src_dir = os.path.join('..', 'RoadPavingBreakage', 'cfgs', model_name)
target_dst_dir = os.path.join('Yolov4Detector', 'cfgs', model_name)
if not os.path.exists(target_dst_dir):
    os.mkdir(target_dst_dir)

for fn in ['road.cfg', 'road.names']:
    shutil.copyfile(os.path.join(target_src_dir, 'data', fn), os.path.join(target_dst_dir, fn))
shutil.copyfile(os.path.join(target_src_dir, 'data', 'road.data'), os.path.join(target_dst_dir, 'road_src.data'))
shutil.copyfile(os.path.join(target_src_dir, 'backup', 'road_best.weights'), os.path.join(target_dst_dir, 'road_best.weights'))

data_f_src = open(os.path.join(target_dst_dir, 'road_src.data'), 'r', encoding='utf8')
data_f_dst = open(os.path.join(target_dst_dir, 'road.data'), 'w', encoding='utf8')

for line in data_f_src:
    if 'classes = ' in line:
        data_f_dst.write(line)
    elif 'names = ' in line:
        data_f_dst.write('names = ' + os.path.abspath(os.path.join(target_dst_dir, 'road.names')))
data_f_src.close()
data_f_dst.close()

