import requests
import tensorflow as tf
import numpy as np
import gzip
#Part 5: Deployment and Inference
def get_testset():
    url = 'https://courses.engr.illinois.edu/ece498icc/sp2020/lab1_request_dataset.php'
    #url = 'https://courses.engr.illinois.edu/ece498icc/sp2020/lab2_test/test_images/images_b56745b1.gz'
    #url = 'https://courses.engr.illinois.edu/ece498icc/sp2020/lab2_test/test_images/images_fd939ecb.gz'
    values = {'request': 'testdata', 'netid':'nnathan2'}
    r = requests.post(url, data=values, allow_redirects=True)
    print(r)
    filename = r.url.split("/")[-1]
    testset_id = filename.split(".")[0].split("_")[-1]
    with open(filename, 'wb') as f: 
        f.write(r.content)
    return load_dataset(filename), testset_id

def load_dataset(path):
    num_img = 1000
    with gzip.open(path, 'rb') as infile:
        data = np.frombuffer(infile.read(), dtype=np.uint8).reshape(num_img, 784)
    return data

def prediction_accuracy(testset_id, prediction):
    url = 'https://courses.engr.illinois.edu/ece498icc/sp2020/lab1_request_dataset.php'
    #url = 'https://courses.engr.illinois.edu/ece498icc/sp2020/lab2_test/test_images/images_b56745b1.gz'
    #url = 'https://courses.engr.illinois.edu/ece498icc/sp2020/lab2_test/test_images/images_fd939ecb.gz'
    values = {'request': 'verify', 'netid':'nnathan2',
              'testset_id':testset_id, 'prediction':prediction}
    r = requests.post(url, data=values, allow_redirects=True)
    accuracy = (r.json())/1000.0
    return accuracy


images, testset_id = get_testset()
prediction = []
model = tf.keras.models.load_model('keras_model1')
prediction.append( [ ((model.predict_classes(images[num].reshape(1, 28, 28, 1),
                      verbose=0)).astype(str))[0] for num in range (0, 1000) ] )
prediction = ''.join(prediction[0])
print(prediction)
accuracy = prediction_accuracy(testset_id, prediction)
print("\nAccuracy: ", accuracy)