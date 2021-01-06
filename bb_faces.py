from facenet_pytorch import MTCNN
import cv2
import numpy as np
import torch
import glob
import os


def _draw(img, boxes, probs, landmarks):
    """
    Draw bounding box, probs and landmarks)
    """
    for box, prob, ld in zip(boxes, probs, landmarks):
        # draw rectangle around the face
        cv2.rectangle(img,
                      (box[0], box[1]),
                      (box[2], box[3]),
                      (0, 0, 255),
                      thickness=10)
        # show probabilities
        # cv2.putText(frame, str(prob), (box[2], box[3]),
        #             cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        # Draw landmarks(5)
        cv2.circle(img, tuple(ld[0]), 10, (0, 0, 255), 8)
        cv2.circle(img, tuple(ld[1]), 10, (0, 0, 255), 8)
        cv2.circle(img, tuple(ld[2]), 10, (0, 0, 255), 8)
        cv2.circle(img, tuple(ld[3]), 10, (0, 0, 255), 8)
        cv2.circle(img, tuple(ld[4]), 10, (0, 0, 255), 8)

    return img


path = glob.glob("D:/Image Dataset/val/not_aryan/*.*")
save_path = "D:/Python Files/Face_Detection/New_Image/val/not_aryan"

mtcnn = MTCNN()

for count,file in enumerate(path):
    img = cv2.imread(file)
    # new_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    try:
        boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
        final_image = _draw(img, boxes, probs, landmarks)
    except:
        pass
    cv2.imwrite(os.path.join(save_path, str(count) + "new.jpg"), final_image)


print("Done")

