# Face-Recognizer
Use deep learning to recognize the face <br>
Face Recognition is the problem of identifying and verifying people by their face. It is a pretty easy task for a human to recognize a face but is incredibly difficult for a computer
to detect it.<br>
As the technology advanced and Deep Learning came into sight and "Computer Vision" became possible. <br>
Deep learning methods are able to leverage very large datasets of faces and learn rich and compact representations of faces, allowing modern models to first perform as-well 
and later to outperform the face recognition capabilities of humans.<br><br>
In this, I have used  **PyTorch** for computer vision and also used **Inception-Resnet V1** pre trained model to train my model according to my dataset.<br><br>

### Dataset Used<br>
- My dataset was divided as training set and validation set
- Training set had 2 label - "aryan" and "not_aryan"
- "aryan" folder had around 200 images of my face and "not_aryan" folder had around 40 images of random faces
- Similarly in validation set "aryan" consist of 35 images and "not_aryan" consists of around 25 images.
<br><br>
### What I did !!!<br>
- First I used the code in "bb_faces.py" to customize my dataset. This helps in creating bounding box and marking 5 landmarks on the faces using MTCNN. 
- I have used Google Colab to train my model and take advantage of GPU and more RAM provided by colab. But my dataset was pretty large (size), so it was a hectic task too upload it
on drive and then on colab hence I used Jupyter Notebook in my local machine and convert the image to **Numerical Python** files so it world if easy to upload those in colab. "transform_imges.ipynb" 
notebook has the code for transforming images to ".npy" files.
- After that, I have used **"Inception-Resnet V1"** to train my model according to my customized dataset or we should say the converted dataset(numerical python files). At last we will
save the model so that we can use it to recognize face in real-time. "Face Recognition.ipynb" is the notebook for this work.
- I have used "OpenCV" library for accessing my webcame and "MTCNN" for detecting faces and landmarks. After that, use the frame and saved model to check if the if the person in front is "Aryan"(Me) or 
"Not_Aryan"(someone else). "app.py" notebook has code for it.
<br><br>
### Result<br>
The best valdiation accuracy that came after training my model was **"82.14%"**.<br><br>
The output during realtime : <br>
![Face Recognizer 06-01-2021 06_41_02 PM](https://user-images.githubusercontent.com/50714723/103772802-141d4380-5050-11eb-9805-e1eab9078577.png)
<br>
### Future Scope of the Project<br>
- First we can increase the accuracy of our project by feeding it with more images hence increasing the size of dataset
- Right now, this project is performing binary classification i.e "Aryan" and "Not Aryan" we can make a multi classification model by making more labels of different individuals and 
using their images to make the machine learn their faces.
