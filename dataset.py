import os
import cv2
import numpy as np


def random_crop(img, height, width):
    x = np.random.randint(0, img.shape[1] - width)
    y = np.random.randint(0, img.shape[0] - height)
    result = img[y:y + height, x:x + width]
    return result


# def sort_img(image_file):
#     num_img = dict()
#     result = list()
#     for img in image_file:
#         dot = img.find('.')
#         num_img[int(img[:dot])] = img
#     nums = list(num_img.keys())
#     nums.sort()
#     for num in nums:
#         result.append(num_img[num])
#     return result


def load_data_train(images_path, img_size):
    lr_size = img_size
    hr_size = (lr_size[0] * 4, lr_size[1] * 4)
    # images = sort_img(os.listdir(images_path))[:examples]
    images = os.listdir(images_path)
    hr_data = []
    lr_data = []
    for image in images:
        img = cv2.imread(os.path.join(images_path, image))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        hr = cv2.resize(img, hr_size)
        lr = cv2.resize(img, lr_size)
        hr_data.append(hr)
        lr_data.append(lr)

        for _ in range(5):
            hr_crop = random_crop(img, hr_size[0], hr_size[0])
            lr_crop = cv2.resize(hr_crop, lr_size)
            hr_data.append(hr_crop)
            lr_data.append(lr_crop)
    return np.array(hr_data, dtype=float), np.array(lr_data, dtype=float)


def load_data_val(hr_val, lr_val):
    hr_val = cv2.imread(hr_val)
    hr_val = cv2.cvtColor(hr_val, cv2.COLOR_BGR2RGB)
    lr_val = cv2.imread(lr_val)
    lr_val = cv2.cvtColor(lr_val, cv2.COLOR_BGR2RGB)

    hr_val = np.expand_dims(np.array(hr_val), axis=0)
    lr_val = np.expand_dims(np.array(lr_val), axis=0)
    return np.array(hr_val, dtype=float), np.array(lr_val, dtype=float)


def load_data_test(lr_test):
    lr_test = cv2.imread(lr_test)
    lr_test = cv2.cvtColor(lr_test, cv2.COLOR_BGR2RGB)
    lr_test = np.expand_dims(np.array(lr_test), axis=0)
    return np.array(lr_test, dtype=float)

def load_video(video_file, lr_file, sr_file, scale, from_high_resolution):
    video = cv2.VideoCapture(video_file)
    lr_video = None

    frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(3))
    height = int(video.get(4))
    fps = video.get(cv2.CAP_PROP_FPS)
    length = frames / fps

    print("----------------Video Information----------------")
    print("Frame:", frames)
    print("Width:", width, 'px')
    print("Height:", height, 'px')
    print("FPS:", fps)
    print("Video length:", length, 's')

    if from_high_resolution:
        width //= scale
        height //= scale
        lr_video = cv2.VideoWriter(lr_file, cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (width, height))
    sr_video = cv2.VideoWriter(sr_file, cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps, (width * 4, height * 4))

    return video, lr_video, sr_video


def batch_data(examples, hr_datasets, lr_datasets, batch_size):
    train_lr_batches = []
    train_hr_batches = []
    for it in range(examples // batch_size):
        start_idx = it * batch_size
        end_idx = start_idx + batch_size
        train_hr_batches.append(hr_datasets[start_idx:end_idx])
        train_lr_batches.append(lr_datasets[start_idx:end_idx])
    return np.array(train_hr_batches), np.array(train_lr_batches)
