import json
import logging as log
from os import listdir, path

import numpy as np
from sklearn.model_selection import train_test_split

log.basicConfig(format="%(asctime)s - %(message)s", level=log.INFO)


class DataLoader(object):
    """
    DataLoader class for the A2D2 dataset

    Args:
        input_path: base path of the above dataset
        cam_config: location of the cams_lidars.json file
        data_type: train or valid
    """

    def __init__(self, input_path: str, cam_config: str, data_type: str = "train"):
        self.input_path = input_path
        log.info(f"input_path={input_path}")

        with open(cam_config, "r") as f:
            self.config = json.load(f)

        tmp_directories = [path.join(self.input_path, x) for x in listdir(self.input_path)]
        directories = [x for x in tmp_directories if path.isdir(x)]

        self.train_directories, self.valid_directories = train_test_split(directories, test_size=0.2, random_state=42)
        log.info(
            f"using {len(self.train_directories)} directories as train and {len(self.valid_directories)} as validation"
        )
        self.data_type = data_type

        self.train_images = []
        self.valid_images = []

        self.train_labels = []
        self.valid_labels = []

        self.CATS = set()

        self.all_train_imgs = {}
        self.all_valid_imgs = {}

        self.read_directories()

    def read_directories(self):
        dir_to_read = self.train_directories if self.data_type == "train" else self.valid_directories

        for dd in dir_to_read:
            self.read_directory(dd, data_type=self.data_type)
            log.info(f"reading {dd} from {dir_to_read}")

    def read_directory(self, directory: str, data_type: str = "train"):

        label_directory = path.join(directory, "label3D", "cam_front_center")
        label_files = [path.join(label_directory, x) for x in listdir(label_directory) if x.endswith("json")]

        for label_file in label_files:
            image_file = path.join(
                path.dirname(path.dirname(path.dirname(label_file))),
                "camera",
                "cam_front_center",
                path.basename(label_file).replace("json", "jpg").replace("label3D", "camera"),
            )

            assert path.exists(image_file)
            assert path.exists(label_file)

            if data_type == "train":
                self.train_images.append(image_file)
            elif data_type == "valid":
                self.valid_images.append(image_file)

            with open(label_file, "r") as f:
                bounding_boxes = json.load(f)

            annotations = []
            for box in bounding_boxes:
                new_alpha = DataLoader.get_new_alpha(bounding_boxes[box]["alpha"])
                dimension = np.array(bounding_boxes[box]["size"])
                xmin, ymin, xmax, ymax = bounding_boxes[box]["2d_bbox"]

                rotation = DataLoader.axis_angle_to_rotation_mat(
                    bounding_boxes[box]["axis"], bounding_boxes[box]["rot_angle"]
                )

                # TODO: AWI 20.07.2020: is this correct?
                P = np.matmul(self.config["cameras"]["front_center"]["CamMatrix"], rotation)

                if data_type == "train":
                    annotation = {
                        "name": bounding_boxes[box]["class"],
                        "image": image_file,
                        "ymin": ymin,
                        "xmin": xmin,
                        "ymax": ymax,
                        "xmax": xmax,
                        "dims": dimension,
                        "new_alpha": new_alpha,
                        "P": P,
                        "center": bounding_boxes[box]["center"],
                    }
                    annotations.append(annotation)
                elif data_type == "valid":
                    annotation = {
                        "name": bounding_boxes[box]["class"],
                        "image": image_file,
                        "ymin": ymin,
                        "xmin": xmin,
                        "ymax": ymax,
                        "xmax": xmax,
                        "dims": dimension,
                        "alpha": bounding_boxes[box]["alpha"],
                        "rot_y": bounding_boxes[box]["rot_angle"],
                        "P": P,
                        "center": bounding_boxes[box]["center"],
                    }
                    annotations.append(annotation)

            if data_type == "train":
                self.all_train_imgs[image_file] = annotations
                self.train_labels.extend(annotations)
            elif data_type == "valid":
                self.all_valid_imgs[image_file] = annotations
                self.valid_labels.extend(annotations)

    def get_average_dimension(self):
        for ll in self.train_labels:
            self.CATS.add(ll["name"])

        dims_avg = {key: np.array([0, 0, 0]) for key in self.CATS}
        dims_cnt = {key: 0 for key in self.CATS}

        self.update_average_dimension(
            self.CATS, dims_avg, dims_cnt, self.train_labels,
        )
        self.update_average_dimension(self.CATS, dims_avg, dims_cnt, self.valid_labels)

        return dims_avg, dims_cnt

    @staticmethod
    def update_average_dimension(CATS, dims_avg, dims_cnt, dataset):
        for i in range(len(dataset)):
            current_data = dataset[i]
            if current_data["name"] in CATS:
                dims_avg[current_data["name"]] = (
                    dims_cnt[current_data["name"]] * dims_avg[current_data["name"]] + current_data["dims"]
                )
                dims_cnt[current_data["name"]] += 1
                dims_avg[current_data["name"]] /= dims_cnt[current_data["name"]]

    @staticmethod
    def get_new_alpha(alpha):
        """
        change the range of orientation from [-pi, pi] to [0, 2pi]
        :param alpha: original orientation in KITTI
        :return: new alpha
        """
        new_alpha = float(alpha) + np.pi / 2.0
        if new_alpha < 0:
            new_alpha = new_alpha + 2.0 * np.pi
            # make sure angle lies in [0, 2pi]
        new_alpha = new_alpha - int(new_alpha / (2.0 * np.pi)) * (2.0 * np.pi)

        return new_alpha

    @staticmethod
    def skew_sym_matrix(u):
        return np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])

    @staticmethod
    def axis_angle_to_rotation_mat(axis, angle):
        return (
            np.cos(angle) * np.eye(3)
            + np.sin(angle) * DataLoader.skew_sym_matrix(axis)
            + (1 - np.cos(angle)) * np.outer(axis, axis)
        )


if __name__ == "__main__":
    data_loader = DataLoader(
        input_path="/media/wittmaan/Elements/data/a2d2/camera_lidar_semantic_bboxes",
        cam_config="/media/wittmaan/Elements/data/a2d2/cams_lidars.json",
        data_type="train",
    )
    result = data_loader.get_average_dimension()
    log.info(result)
