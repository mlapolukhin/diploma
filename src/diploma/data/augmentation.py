import matplotlib.pyplot as plt
from imgaug import augmenters as iaa


class Augmentor:
    def __init__(self):
        self.aug = iaa.Sequential(
            [
                # color aug
                iaa.Sometimes(
                    0.9,
                    iaa.SomeOf(1, [iaa.AverageBlur(k=(1)), iaa.Sharpen((0.0, 0.2))]),
                ),
                iaa.Sometimes(
                    0.6,
                    iaa.Sequential(
                        [
                            iaa.ChangeColorspace(
                                from_colorspace="RGB", to_colorspace="HSV"
                            ),
                            iaa.WithChannels(0, iaa.Add((-5, 5))),
                            iaa.WithChannels(1, iaa.Multiply((0.0, 1.5))),
                            iaa.WithChannels(2, iaa.Multiply((0.7, 1.3))),
                            iaa.ChangeColorspace(
                                from_colorspace="HSV", to_colorspace="RGB"
                            ),
                        ]
                    ),
                ),
                # corrupting
                iaa.Sometimes(0.5, iaa.JpegCompression((0, 50))),
                # iaa.CoarseDropout(p=0.05, size_px=(3, 6), per_channel=False),
                # geometry aug
                iaa.Sometimes(
                    0.8, iaa.PerspectiveTransform(scale=(0.00, 0.05), keep_size=True)
                ),
                iaa.Sometimes(0.5, iaa.Fliplr()),
                iaa.Sometimes(0.5, iaa.Affine(rotate=(-15, 15))),
                iaa.Sometimes(0.5, iaa.Affine(scale=(0.9, 1.1))),
                iaa.Sometimes(0.5, iaa.Affine(shear=(-30, 30))),
                iaa.Sometimes(
                    0.5,
                    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                ),
                iaa.CropAndPad(
                    px=((0, 10), (0, 10), (0, 10), (0, 10)),
                    pad_mode=["symmetric", "reflect", "constant"],
                ),
            ],
            random_order=True,
        )

    def augmentate_sample(self, img):
        # '''
        # img: np.array HxWx3, uint8
        # '''

        img = self.aug.augment_image(image=img)
        return img


if __name__ == "__main__":
    #     ==== OPENCV ====
    aug = Augmentor()

    from pathlib import Path

    images = sorted(
        Path(
            "/media/ap/Transcend/Projects/diploma/assets/data/tmp/openimages/images"
        ).glob("*.jpg")
    )

    while True:
        img_path = images[0]
        img = plt.imread(img_path)

        img_a = aug.augmentate_sample(img)
        plt.imshow(img_a)
        plt.title(img_path)
        plt.show()