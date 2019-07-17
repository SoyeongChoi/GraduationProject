import numpy as np
from PIL import Image
import os
import pandas as pd



# class DataGenerator(self, selection = [0,1,2,3,6], batch_):
# We Using AffectNet DataSet
data_path = os.path.join('../', 'data', 'Manually_Annotated')

# Expression List
expression_list = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt', 'None', 'Uncertain',
                   ' Non-face']

# Manually Annotated Expression Size
expression_size = [75374, 134915, 25959, 14590, 6878, 4303, 25382, 4250, 33588, 12145, 82915]
## netural, happy, sad, surprise, anger
## 0, 1, 2, 3, 6 [label min size = 14590 (Surprise)/ label max size = 134915(Happy)]

# emotion select
selection = [0,1,2,3,6]

# get selections min for same size data(label)
selection_size_list = [expression_size[x] for x in selection]
min_size = min(selection_size_list)

##
trainset = pd.read_csv(os.path.join(data_path, "training.csv"))

colnames = ['subDirectory_filePath', 'face_x', 'face_y', 'face_width', 'face_height', 'facial_landmarks',
            'expression', 'valence', 'arousal' ]

testset = pd.read_csv(os.path.join(path, "validation.csv"), names=colnames)


# remove 1. 'facial_landmarks' 2. 'valence' 3. 'arousal'
# trainset.drop(['facial_landmarks', 'valence', 'arousal'],axis =1)
# testset.drop(['facial_landmarks', 'valence', 'arousal'],axis =1)

# only selected emotions remain
trainset = dataset[(dataset['expression'].isin(selection))]
testset = testset[(dataset['expression'].isin(selection))]


trainset_index = dataset.reset_index().columns[0, 'expression']
testset_index = testset.reset_index().columns[0, 'expression']

# n * m min_size
train_data = np.zeros(len(selection) * min_size)
test_data = (testset_index.columns[0]).values


# random sampling for train
for i in selection:
    # 이게 random sampling index
    train_data[i*min_size] = ((dataset_index[dataset_index['expression'] == i].sample(n= min_size, random_stata=1)).columns[0]).values


print(dataset_index)
# remove

# Todo : 1. minsize로 데이터를 자르고 ,

'''
basewidth = 300

for i in range(4):  # range(dataset.shape[0]):

    image_path = os.path.join(data_path, dataset['subDirectory_filePath'][i])

    if os.path.exists(image_path):
        img = Image.open(image_path)

        fig, ax = plt.subplots(1)

        origin_size = img.size
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))

        img = img.resize((basewidth, hsize), Image.ANTIALIAS)

        imgplot = plt.imshow(img)

        rect = matplotlib.patches.Rectangle([dataset['face_x'][i] * wpercent, dataset['face_y'][i] * wpercent],
                                            dataset['face_width'][i] * wpercent, dataset['face_height'][i] * wpercent,
                                            linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

        plt.show()

        print("original size is ", origin_size, " => modified size is ", img.size)
        print(i, "th data's expression is : ", expression_list[dataset['expression'][i]])
    else:
        print(i, 'th data is not exist')

    time.sleep(1)


'''