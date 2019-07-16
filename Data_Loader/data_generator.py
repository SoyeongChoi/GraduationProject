import numpy as np
from PIL import Image
import os



# We Using AffectNet DataSet
data_path = os.path.join('../', 'data', 'Manually_Annotated')

i = 0

expression_list = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt', 'None', 'Uncertain',
                   ' Non-face']

expression_size = [80276, 146198, 29487, 16288, 8191, 5264, 28130, 5135, 35322, 13163, 88895]
## netural, happy, sad, surprise, anger
## 0, 1, 2, 3, 6 [label min size = 16288 (Surprise)/ label max size = 146198(Happy)]

# emotion select
selection = [0,1,2,3,6]

#
selection_size_list = [expression_size[x] for x in selection]
print(selection_size_list)
print(min(selection_size_list))


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