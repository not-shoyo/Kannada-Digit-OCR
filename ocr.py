from new_data import convert_to_mnist_format as convert
import os

TESTING = False

if not TESTING:
    from PIL import Image
    import numpy as np

    def read_image(path):
        abcdefg = np.asarray(Image.open(path).convert('L'))
        # print(f"type(image):         {type(abcdefg)}")
        # print(f"type(image[0]):      {type(abcdefg[0])}")
        # print(f"type(image[0][0]):   {type(abcdefg[0][0])}")
        
        return abcdefg


# DATA_DIR = 'data/'
DATA_DIR = 'new_data/'
#OUR_INPUT (user input for application)
OUR_DIR = 'our_input/'
# DATASET = 'mnist'
DATASET = 'converted_MNIST'
# change images-idx3 to images.idx3 to use the actual mnist files
TEST_DATA_FILENAME = DATA_DIR + DATASET + '/t10k-images-idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + DATASET + '/t10k-labels-idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + DATASET + '/train-images-idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + DATASET + '/train-labels-idx1-ubyte'


def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')


def read_images(filename, n_max_images=None):
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4)
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4))
        n_columns = bytes_to_int(f.read(4))
        for image_idx in range(n_images):
            image = []
            for row_idx in range(n_rows):
                row = []
                for col_idx in range(n_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images


def read_labels(filename, n_max_labels=None):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for label_idx in range(n_labels):
            label = bytes_to_int(f.read(1))
            labels.append(label)
    # print(labels)
    return labels


def flatten_list(l):
    return [pixel for sublist in l for pixel in sublist]


def extract_features(X):
    return [flatten_list(sample) for sample in X]


def dist(x, y):     #Returns the Euclidean distance between vectors `x` and `y`.
    return sum(
        [
            (bytes_to_int(x_i) - bytes_to_int(y_i)) ** 2
            for x_i, y_i in zip(x, y)
        ]
    ) ** (0.5)


def get_training_distances_for_test_sample(X_train, test_sample):
    return [dist(train_sample, test_sample) for train_sample in X_train]


def get_most_frequent_element(l):
    return max(l, key=l.count)


def knn(X_train, y_train, X_test, k=3):
    y_pred = []
    print("\ncalculating\r", end="")
    for test_sample_idx, test_sample in enumerate(X_test):
        print(".", end='.', flush=True)
        training_distances = get_training_distances_for_test_sample(
            X_train, test_sample
        )
        sorted_distance_indices = [
            pair[0]
            for pair in sorted(
                enumerate(training_distances),
                key=lambda x: x[1]
            )
        ]
        candidates = [
            y_train[idx]
            for idx in sorted_distance_indices[:k]
        ]
        top_candidate = get_most_frequent_element(candidates)
        y_pred.append(top_candidate)
    print()
    return y_pred

def main():
    #during testing phase n_train = 90
    if TESTING:
        n_train = 90
        n_test = 2
    else:
        n_train = 100
    k = 2
    print(f'Dataset: {DATASET}')
    print(f'n_train: {n_train}')
    if TESTING:
        print(f'n_test: {n_test}')
    print(f'k: {k}')
    X_train = read_images(TRAIN_DATA_FILENAME, n_train)
    y_train = read_labels(TRAIN_LABELS_FILENAME, n_train)
    if TESTING:
        X_test = read_images(TEST_DATA_FILENAME, n_test)
        y_test = read_labels(TEST_LABELS_FILENAME, n_test)
    print("\nTraining labels:")
    print(y_train)

    if not TESTING:
        # Load in the images from `our_input` folder
        X_test = [read_image(f'{OUR_DIR}test{idx}.png') for idx in range(10)]
        y_test = [idx for idx in range(10)]

    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    y_pred = knn(X_train, y_train, X_test, k)

    accuracy = sum([
        int(y_pred_i == y_test_i)
        for y_pred_i, y_test_i
        in zip(y_pred, y_test)
    ]) / len(y_test)

    
    print(f'\nActual    labels: {y_test}')
    print(f'Predicted labels: {y_pred}')

    print(f'Accuracy: {accuracy * 100}%')

if __name__ == '__main__':
    # st = "731221549637860753640809344911621809573493436527444926117875083358382190266518657252180479"
    # st = "0324604705645340290351691354378899644879766417170311859502817288282609990186595375180123260000000000"
    # st = "7727909537501839381679764032829667400113814424365813415953927012152345321508998667689040466258247850"
    # print(st.count('3'))
    current = os.getcwd()
    # print("current is: ", current)
    os.chdir('new_data')
    convert.main(["convert_to_mnist_format.py", "compressedKannada", "train"])
    os.chdir(current)
    main()
