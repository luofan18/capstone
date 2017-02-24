from time import time
import threading
import traceback
import numpy as np

import csv
import socket

w_spec = 224
h_spec = 224

def retrieve_image(url, filename):
    try:
        urllib.urlretrieve(url, filename)
    except socket.error as e:
        print 'bad link'
        
    try:
        image = imread(filename)
        if len(image.shape) != 3:
            os.remove(filename)
            print 'not a color image'
    except IOError as e:
        print traceback.format_exc()
        print 'not a image at url ' + url
        os.remove(filename)

url_dir = 'data/landmark/url_id_query.txt'

with open(url_dir) as f:
    reader = csv.reader(f, delimiter="\t")
    d = list(reader)

total_size = len(d)
print total_size

# j to track the index of image
j = 0
neural_code = []

while j<20:
    
    # download the image
    batch_size = 5
    threads = []
    tic = time()
    for i in range(batch_size):
        if i%128 == 0:
            # comma to refresh the output
            print 'proccesing' + str(j+i/float(total_size)) + '\r',
        if i+j < total_size:
            thread = threading.Thread(target=retrieve_image,args=
                    (d[j+i][0], 'data/landmark/image/'+d[j+i][1]+str(i)+'.jpg'))
            thread.start()
            threads.append(thread)

    for t in threads:
        print t
        t.join()

    toc = time()
    print 'Elapse time is '+ str(toc-tic)

    # get the neural code
    base_dir = 'data/landmark/image/'

    image_name = []
    import os
    # add only jpg file
    for file in os.listdir(base_dir):
        if file.endswith('.jpg'):
            try:
                image = imread(base_dir+file)
                if len(image.shape) != 3:
                    os.remove(base_dir+file)
                    print 'not a color image'
                else:
                    image_name.append(file)
            except IOError as e:
                os.remove(base_dir+file)
        
    image_number = len(image_name)
    data_batch = np.zeros((image_number,w_spec,h_spec,3))

    for i,file in enumerate(image_name):
        image = imread(base_dir+file)
        image = imresize(image, (w_spec, h_spec))
        data_batch[i] = image
    
    out = new_model.predict(data_batch)
    out = out.dot(weights[0]) + weights[1]
    
    neural_code.append(out)
    
    # delete the image and update j
    for file in image_name:
        os.remove(base_dir+file)
    j += batch_size
    
np.dump.save(neural_code)