x_train = []
y_train = []

img_path = "train/"+ids+".jpg"
  img = cv2.imread(img_path)
  target = np.zeros(120)
  x_train.append(np.array(cv2.resize(img,(224,224))))
  target[labels_map[label]] = 1
  y_train.append(target)

# Converting lists into numpy arrays
# Our training set is divided by 255 to normalize the image data
y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float16) / 255.
print(len(x_train), len(y_train), x_train.shape, y_train.shape)
length = len(x_train)

# Splitting the data into Training, Validation and Test set
test_split = int(length*0.8)
print("Training set + Validation set: ", test_split)
x_train, x_test, y_train, y_test = x_train[:test_split], x_train[test_split:], y_train[:test_split], y_train[test_split:]

valid_split = int(length*0.6)
print("Training set: ", valid_split)
x_train, x_valid, y_train, y_valid = x_train[:valid_split], x_train[valid_split:], y_train[:valid_split], y_train[valid_split:]

print("Validation set: ", str(test_split - valid_split))
print("Test set: ", str(length - test_split))