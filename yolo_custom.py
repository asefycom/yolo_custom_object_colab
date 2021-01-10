import cv2
import numpy as np

obj_file = "obj.names"
obj_classes = []
net_config = "cfg/yolov3_training.cfg"
net_weights = "cfg/yolov3_training_last.weights"
blob_size = 320
confidence_threshold = 0.25
nms_threshold = 0.3

with open(obj_file, "rt") as f:
    obj_classes = f.read().rstrip("\n").split("\n")


net = cv2.dnn.readNetFromDarknet(net_config, net_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

def findObjects(output, img):
    img_h, img_w, img_c = img.shape
    bboxes = []
    class_ids = []
    confidences = []

    for cell in output:
        for detect_vector in cell:
            scores = detect_vector[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_threshold:
                w,h = int(detect_vector[2] * img_w), int(detect_vector[3] * img_h)
                x,y = int((detect_vector[0] * img_w) - w/2), int((detect_vector[1] * img_h) - h/2)
                bboxes.append([x,y,w,h])
                class_ids.append(class_id)
                confidences.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(bboxes, confidences, confidence_threshold, nms_threshold)
    # print(indices)
    for i in indices:
        i = i[0]
        bbox = bboxes[i]
        x,y,w,h = bbox[0], bbox[1], bbox[2], bbox[3]
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(img, f'{obj_classes[class_ids[i]].upper()} {int(confidences[i] * 100)}%',
                    (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

frame = cv2.imread('test1.jpg')
blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(blob_size,blob_size),mean=(0,0,0)
                             ,swapRB=True,crop=False)

net.setInput(blob)
out_names = net.getUnconnectedOutLayersNames()
output = net.forward(out_names)

findObjects(output, frame)

cv2.imshow("Webcam", frame)
cv2.waitKey(0)