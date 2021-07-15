import numpy as np
import os,csv,cv2
class AR:
    def __init__(self, landmarks, mainImg, img):
        self.result = mainImg
        self.img = img
        self.dst_pts = landmarks[:17]
        self.mask=None
        # load mask annotations from csv file to source points
        self.mask_annotation = os.path.splitext(self.img)[0]
        print( self.mask_annotation )
        self.mask_annotation = os.path.join(
            self.mask_annotation + ".csv",
        )

        with open(self.mask_annotation) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            src_pts = []
            for i, row in enumerate(csv_reader):
                # skip head or empty line if it's there
                try:
                    src_pts.append(np.array([float(row[1]), float(row[2])]))
                except ValueError:
                    continue
        src_pts = np.array(src_pts, dtype="float32")

        # overlay with a mask only if all landmarks have positive coordinates:
        if (len(landmarks) > 0):
            # load mask image
            mask_img = cv2.imread(self.img, cv2.IMREAD_UNCHANGED)
            mask_img = mask_img.astype(np.float32)
            mask_img = mask_img / 255.0

            # get the perspective transformation matrix
            M, _ = cv2.findHomography(src_pts, self.dst_pts)

            # transformed masked image
            transformed_mask = cv2.warpPerspective(
                mask_img,
                M,
                (self.result.shape[1], self.result.shape[0]),
                None,
                cv2.INTER_LINEAR,
                cv2.BORDER_CONSTANT,
            )

            # mask overlay
            alpha_mask = transformed_mask[:, :, 3]
            alpha_image = 1.0 - alpha_mask
            self.mask=mask_img
            for c in range(0, 3):
                self.result[:, :, c] = (
                        alpha_mask * transformed_mask[:, :, c]
                        + alpha_image * self.result[:, :, c]
                )
        # display the resulting frame


