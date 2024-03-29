[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)][1]
[![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)][2]
[![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)][3]
[![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)][4]
[![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)][5]
[![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)][6]

[1]: https://www.tensorflow.org/?gclid=Cj0KCQiA-JacBhC0ARIsAIxybyMp6CdZTnqaeQBwUg1Huc3S9OrO4-NAbOOJHbnxacMMetfutDBH5R0aAgJ6EALw_wcB
[2]: https://keras.io/
[3]: https://scikit-learn.org/stable/index.html
[4]: https://numpy.org/devdocs/index.html
[5]: https://matplotlib.org/
[6]: https://docs.opencv.org/4.x/

# AIP391 Project: Enhance image resolution using a Generative Adversarial Network
* **Team member**
    * [Dinh Hoang Lam](https://github.com/LamKser)
    * [Tran Duy Ngoc Bao](https://github.com/TranDuyNgocBao)
    * [Tran Nguyen Phuc Vinh](https://github.com/Lasky0908)
* Our report: [Report team 2](https://drive.google.com/file/d/1NcJaqgdRR7shUG4cr5kZl_x4qmRtybe8/view?usp=drive_link)
* Web demo with Flask:  
   * Download source code [**Web-app**](https://github.com/LamKser/image-super-resolution-web-app) and run `app.py`
## :bricks: **SRGAN architecture**
Original paper: [Photo-Realistic Single Image Super-
Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802?context=cs).

<div align="center">
    <img src="https://user-images.githubusercontent.com/83662223/179677280-874d1f3f-bb72-4efe-8012-6075bb0b8cac.jpeg" width="70%" height="70%">
</div>
<p align="center">
    <strong>Figure 1:</strong> SRGAN architecture
</p>

## :books: **Data preparation**
### **1. Training set**
* We use `DIV2K` dataset for training the model, you can download [here](https://github.com/LamKser/Image-super-resolution-using-GAN/tree/main/dataset).
* Then doing random crop the image with size `96x96` for HR image and resize to `24x24` for LR image.
* After cropping, we scale HR and LR images to range `[-1, 1]`.
### **2. Test set**
* You can download `Set5`, `Set14`, and `Urban100` datasets [here](https://github.com/LamKser/Image-super-resolution-using-GAN/tree/main/dataset).

## :building_construction: **Run model**
Change the `choice` variable in `train_test_model.py` to run the model
```
1: Train model
2: Validate model
3: Test model 
4: Test on video
```
### :hourglass: **1. Training**
* :file_folder: Set the train data path `hr_train_path = Dataset/DIV2K` and the LR image size `lr_size = (24, 24)`, then the HR image size equals LR image size times 4 `hr_size = (96, 96)`.
* You can use other dataset with different LR image size for training by set up:
```Python
hr_train_path = 'Dataset/your_images_folder'
lr_size = your_size
```
* :file_folder: Set your weight path to save weights in every epoch with HR and LR image size. Your directory will be `weight/(LR_size)_(HR_size)`
```Python
save_path = 'weight'
```
### :bar_chart: **2. Validation**
You should set the HR and LR valid image paths for comparing the SR image with HR and LR images
```Python
hr_valid_path = 'Dataset/your_HR_image'
lr_valid_path = 'Dataset/your_LR_image'
weight_path = 'weight/e_77.h5' # You can change the path
```
### :chart_with_upwards_trend: **3. Test**
Set the LR test image paths for comparing the SR image with HR and LR images
```Python
lr_test_path = 'Your_image'
weight_path = 'weight/e_77.h5' # You can change the path
```

`NOTE:` Make sure the height and width are under 500 pixels because of running out of memory on GPU
## :sun_with_face: **Result**
<div align="center">
   <img src="result/comic.png" width="40%" height="40%">
   <img src="result/butterfly.png" width="50%" height="50%">
</div>
<p align="center">
    <strong>Figure 2:</strong> Set14 and Set5 dataset
</p>

<div align="center">
   <img src="result/urban.png" width="70%" height="70%">
</div>
<p align="center">
    <strong>Figure 3:</strong> Urban dataset
</p>

