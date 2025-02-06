from flask import Flask, request, render_template, url_for
import cv2
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split  # Import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# Initialize model and preprocessing components
dataset_path = "ocr_data/train_new/"
scaler = StandardScaler()
label_encoder = LabelEncoder()
clf = RandomForestClassifier(n_estimators=100, random_state=42)

def load_images_from_folder(folder):
    images = []
    labels = []
    contours_list = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    ret, im_th = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    ctrs, hier = cv2.findContours(im_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    images.append(im_th)
                    labels.append(label)
                    contours_list.append(ctrs)
    return images, labels, contours_list

def extract_and_resize(images, size=(30, 30)):
    resized_images = []
    for img in images:
        char_img_resized = cv2.resize(img, size)
        resized_images.append(char_img_resized.flatten())
    return np.array(resized_images)

images, labels, contours_list = load_images_from_folder(dataset_path)
features = extract_and_resize(images)
features = scaler.fit_transform(features)

encoded_labels = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
unique_pred_labels = np.unique(y_pred)

def predict_characters(im_th, sorted_ctrs, model, scaler, label_encoder):
    bboxes = []
    character_images = []
    im_color = cv2.cvtColor(im_th, cv2.COLOR_GRAY2BGR)
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        area = w * h
        if 250 < area < 900:
            bboxes.append((x, y, w, h))
            character_image = im_th[y:y + h, x:x + w]
            character_images.append(character_image)
            cv2.rectangle(im_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Save the result image with bounding boxes
            bbox_img_path = os.path.join('static', 'bbox_img.png')
            cv2.imwrite(bbox_img_path, cv2.cvtColor(im_color, cv2.COLOR_BGR2RGB))
    characters = extract_and_resize(character_images)
    characters = scaler.transform(characters)
    predictions = model.predict(characters)
    predicted_characters = label_encoder.inverse_transform(predictions)
    return bboxes, predicted_characters

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Read and process the image
            im = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            ret, im_th = cv2.threshold(im_gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ctrs, hier = cv2.findContours(im_th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])

            # Predict and visualize characters
            bboxes, predictions = predict_characters(im_th, sorted_ctrs, clf, scaler, label_encoder)
            if bboxes:
                im_bboxes = im.copy()
                for bbox, prediction in zip(bboxes, predictions):
                    x, y, w, h = bbox
                    cv2.putText(im_bboxes, prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Save the result image
            result_img_path = os.path.join('static', 'result_img.png')
            cv2.imwrite(result_img_path, im_bboxes)

            return render_template('result.html',
                                   accuracy=f"{accuracy * 100:.2f}%",
                                   predictions="".join(predictions),
                                   result_img_url=url_for('static', filename='result_img.png'),
                                   bbox_img_url=url_for('static', filename='bbox_img.png')
                                   )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
