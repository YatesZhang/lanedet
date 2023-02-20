import torchvision
import transform.transforms as T


test_transforms = torchvision.transforms.Compose([
            T.SampleResize((800, 288)),
            T.GroupNormalize(mean=((103.939, 116.779, 123.68), (0, )), std=(
                (1., 1., 1.), (1, ))),
])

train_transforms = torchvision.transforms.Compose([
            T.GroupRandomRotation(degree=(-2, 2)),
            T.GroupRandomHorizontalFlip(),
            T.SampleResize((800, 288)),     # cv2: (590, 1640, 3),
            T.GroupNormalize(mean=((103.939, 116.779, 123.68), (0, )), std=(
                (1., 1., 1.), (1, ))),
])

