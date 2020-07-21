import logging as log
import os
import time
from os import path

import cv2
import numpy as np

from common.constants import BIN
from model.vgg16 import VGG16
from processing.data_loader import DataLoader

log.basicConfig(format="%(asctime)s - %(message)s", level=log.INFO)


def predict(weights, input_path, config, prediction_path):
    model = VGG16()
    model.load_weights(weights)

    data_loader = DataLoader(input_path, config)
    dims_avg, _ = data_loader.get_average_dimension()

    val_imgs = data_loader.valid_images
    val_labels = data_loader.valid_labels

    start_time = time.time()
    for act_img in val_imgs:

        prediction_file = path.join(prediction_path, path.basename(act_img).replace("jpg", "txt"))
        log.info(f"writing to {prediction_file}")

        act_labels = [x for x in val_labels if x["image"] == act_img]

        with open(prediction_file, "w") as predict:
            for act_label in act_labels:
                img = cv2.imread(act_label["image"])
                img = np.array(img, dtype="float32")

                xmin = int(act_label["xmin"])
                ymin = int(act_label["ymin"])
                xmax = int(act_label["xmax"])
                ymax = int(act_label["ymax"])

                xmin = max(xmin, 0)
                ymin = max(ymin, 0)
                xmax = min(xmax, img.shape[1] - 1)
                ymax = min(ymax, img.shape[0] - 1)

                patch = img[
                    ymin : ymax + 1, xmin : xmax + 1,
                ]

                patch = cv2.resize(patch, (224, 224))
                patch -= np.array([[[103.939, 116.779, 123.68]]])
                patch = np.expand_dims(patch, 0)

                prediction = model.predict(patch)
                dim = prediction[0][0]

                # Transform regressed angle
                max_anc = np.argmax(prediction[2][0])
                anchors = prediction[1][0][max_anc]

                if anchors[1] > 0:
                    angle_offset = np.arccos(anchors[0])
                else:
                    angle_offset = -np.arccos(anchors[0])

                wedge = 2.0 * np.pi / BIN
                angle_offset = angle_offset + max_anc * wedge
                angle_offset = angle_offset % (2.0 * np.pi)

                angle_offset = angle_offset - np.pi / 2
                if angle_offset > np.pi:
                    angle_offset = angle_offset - (2.0 * np.pi)

                act_label["alpha"] = angle_offset

                # Transform regressed dimension
                act_label["dims"] = dims_avg[act_label["name"]] + dim
                del act_label["P"]

                line = " ".join([str(act_label[item]) for item in act_label]) + "\n"
                predict.write(line)

    end_time = time.time()
    process_time = (end_time - start_time) / len(val_imgs)
    print(process_time)


if __name__ == "__main__":
    pred_path = "/media/wittmaan/Elements/data/a2d2/pred"

    if not os.path.exists(pred_path):
        os.mkdir(pred_path)

    predict(
        weights="/media/wittmaan/Elements/GoogleDrive/python/a2d2/src/main/python/3dbox_weights.hdf5",
        input_path="/media/wittmaan/Elements/data/a2d2/camera_lidar_semantic_bboxes",
        config="/media/wittmaan/Elements/data/a2d2/cams_lidars.json",
        prediction_path=pred_path,
    )
