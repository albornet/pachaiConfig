import os

if not os.path.exists('labels'):
	os.mkdir('labels')
if not os.path.exists('lmdb'):
	os.mkdir('lmdb')
	os.system(r'chmod -R a+w lmdb')

os.chdir('labels')
with open("trainSet.txt", "w") as my_file:
	for root, dirs, files in os.walk(r'../jpgs/'):
		for file in files:
			if (float(file[7:-3]) <= 1716):
				my_file.write(file + " 0\n")
			else:
				my_file.write(file + " 1\n")

os.chdir('..')
# os.system(r'GLOG_logtostderr=1 ../caffe/build/tools/convert_imageset --resize_height=60 --resize_width=100 --shuffle jpgs/ labels/trainSet.txt lmdb/trainLmdb')
os.system(r'GLOG_logtostderr=1 ../caffe/build/tools/compute_image_mean lmdb/trainLmdb lmdb/mean.binaryproto')
