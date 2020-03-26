# AINOVATION-INTERNS

## Task 1: Object Detection

### 1/ Explore object detection

#### a/ Setup
* Download [Anaconda3](https://www.anaconda.com/distribution/)
* Setup Virtual Environment
* Install Tensorflow, Matplotlib, Pillow, caffe. lxml, pandas, requests.
* Install Cython, contextlib2, Protobuf 3.0.0, cocoapi.
* Download [labelImg](https://github.com/tzutalin/labelImg).
* Download github [models](https://github.com/tensorflow/models/tree/master/research/object_detection) and setup.

#### b/ Project 1: run object_detection_tutorial.ipynb with another images

#### - Result: 
![alt](https://github.com/KS-5011/AINOVATION-INTERNS-/blob/master/Task1/project1.1.png)
![alt](https://github.com/KS-5011/AINOVATION-INTERNS-/blob/master/Task1/project1.png)

#### c/ Project 2: Using LabelImg tool to create files .xml
#### - Step by step:
* Step1: cd ~/Task1/labelImg-master
* Step2: Run
```vim
python3 labelImg.py
```
![alt](https://github.com/KS-5011/AINOVATION-INTERNS-/blob/master/Task1/labelImg.png)

#### - Result: 
```xml
<annotation>
	<folder>pic</folder>
	<filename>cucumber1.jpg</filename>
	<path>/home/ks/pic/cucumber1.jpg</path>
	<source>
		<database>Unknown</database>
	</source>
	<size>
		<width>259</width>
		<height>194</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	<object>
		<name>cucumber1</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>42</xmin>
			<ymin>20</ymin>
			<xmax>216</xmax>
			<ymax>169</ymax>
		</bndbox>
	</object>
</annotation>
```


#### d/ Project 3: Custom Object Detection

#### - Step by step:
* Step1: cd /Task1/CustomObjectDetection.
* Step2: Create `test_labels.csv and train_labels.csv`. Run: 
```vim
python xml_to_csv.py
```
* Step3: Create `train.record and test.record`. Run:
```vim
generate_tfrecord.bat
```
* Step4: Start training
```vim
training.bat
```
#### - Result: 
![alt](https://github.com/KS-5011/AINOVATION-INTERNS-/blob/master/Task1/custom.png)
![alt](https://github.com/KS-5011/AINOVATION-INTERNS-/blob/master/Task1/custom2.png)

### 2/ Deploy by Tensorflow-Serving

#### a/ Setup
* Install flask.
* Download and learn [Postman](https://www.postman.com/downloads/).
* Download [google cloud sdk](https://cloud.google.com/sdk).
* Download [Tensorflow Serving](https://github.com/tensorflow/serving)
* Setup [Docker](https://github.com/fpaupier/tensorflow-serving_sidecar/blob/master/docs/setup.md) for Serving. 

#### b/ Project 1: Create a custom Tensorflow-Serving docker image

#### - Step by step:
* Step1: cd /Task1/TensorflowServingSidecar.
* Step2: Run a serving image as a daemon:
```vim
sudo docker run -d --name serving_base tensorflow/serving`
```
* Step3: Copy the `faster_rcnn_resnet101_coco` model data to the container's `models/` folder:
```vim
sudo docker cp $(pwd)/data/faster_rcnn_resnet101_coco_2018_01_28 serving_base:/models/faster_rcnn_resnet`
```
* Step4: Commit the container to serve the `faster_rcnn_resnet` model:
```vim
sudo docker commit --change "ENV MODEL_NAME faster_rcnn_resnet" serving_base faster_rcnn_resnet_serving
```
* Step5: Start the server:
```vim
sudo docker run -p 8501:8501 -t faster_rcnn_resnet_serving &
```
* Step6: We can use the same client code to call the server. 
```vim
python client.py --server_url "http://localhost:8501/v1/models/faster_rcnn_resnet:predict" --image_path "$(pwd)/object_detection/test_images/person2.jpg" --output_json "$(pwd)/object_detection/test_images/out_person2.json" --save_output_image "True" --label_map "$(pwd)/data/labels.pbtxt"
```
#### - Result:
![alt](https://github.com/KS-5011/AINOVATION-INTERNS-/blob/master/Task1/out_bicycle1.jpeg)
![alt](https://github.com/KS-5011/AINOVATION-INTERNS-/blob/master/Task1/out_person2.jpeg)
![alt](https://github.com/KS-5011/AINOVATION-INTERNS-/blob/master/Task1/tensorflowServing.png)

#### - Tutorial [Link](https://towardsdatascience.com/deploy-your-machine-learning-models-with-tensorflow-serving-and-kubernetes-9d9e78e569db)

#### c/ Project 2: Salary Forecast with Flask

#### - Step by step:
* Step1: cd /Task1/SalaryForecast.
* Step2: Run `python model.py`
* Step3: Run `python app.py`.
* Step3: Navigate to URL `http://localhost:5000`.
#### - Result:
![alt](https://github.com/KS-5011/AINOVATION-INTERNS-/blob/master/Task1/Salary.png)

## Task 2 : Face Recognition with Mobile App

### 1/ Setup:
* Create Virtual Machine
* Install numpy, opencv, flask
* Download [haarcascade](https://github.com/opencv/opencv/tree/master/data/haarcascades)
* Download [Android Studio](https://developer.android.com/studio) .
* Download github [examples]([https://github.com/tensorflow/examples](https://github.com/tensorflow/examples)) tensorflow.
* learn how to convert .tf to .tflite [here]([https://www.tensorflow.org/lite/guide/get_started]).
* Setup [Virtual Machine](https://developer.android.com/studio/run/managing-avds).

### 2/ Dataset:
* From kaggle and Pyimagesearch here.

### 3/ Project 1: Using Haarcascade ( OpenCV ) and Flask

#### - Step by step:
* Step1: cd /Task2/FaceRecognitionWithOpenCV/Setup, run `python app.py`.
* Step2: Run Android studio with `Open an existing Android Studio project`.
* Step3: Choose FaceRecognitionWithOpenCV.
* Step4: run FaceRecognitionWithOpenCV with Virtual Machine.
* Step5: upload image.

#### - Result: 



#### - Tutorial [link](https://www.youtube.com/watch?v=b7VkbAUqMqM&t=2495s)

### 4/ Project 2: Using examples tensorflow with realtime

#### - Step by step:
* Step1: cd /Task2/FaceRecognitionWithTensorflow.
* Step2: Run Android studio with 'Open an existing Android Studio project'.
* Step3: Choose FaceRecognitionWithTensorflow.
* Step4: run FaceRecognitionWithTensorflow with Virtual Machine.

#### - Result: 

### 5/ Project 3: Using React-Native and TensorFlow.js (NOW)

#### a/ Setup
* Download [React-Native](https://reactnative.dev/docs/getting-started).
* Download [Nodejs](https://nodejs.org/en/download/)
* Download [Vim](https://www.vim.org/git.php)
* Download [TensorFlow.js](https://www.tensorflow.org/js/tutorials/setup)

#### b/ Learn
* Course: [CS50's Mobile App Development with React Native](https://www.youtube.com/playlist?list=PLhQjrBD2T382gdfveyad09Ierl_3Jh_wR)
* How to use [The Snack](https://snack.expo.io/)
* How to use [codesandbox](https://codesandbox.io/s/new)

#### To Be Continued . . .
