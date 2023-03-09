import cv2
import json
from deeplearning import yolo_predictions,net

cap = cv2.VideoCapture('./TEST/TEST.mov')

output_list = []

while True:
    ret, frame = cap.read()

    if ret == False:
        print('Unable to read video')
        break

    results = yolo_predictions(frame,net)
    results_list = list(results)
    # Append the predictions to the output_list
    output_list.append(results_list)

    cv2.namedWindow('YOLO',cv2.WINDOW_KEEPRATIO)
    cv2.imshow('YOLO',results)
    if cv2.waitKey(30) == 27 :
        break

# Write the output_list to a JSON file
with open('predictions.json', 'w') as f:
    json.dump(output_list, f)

cv2.destroyAllWindows()
cap.release()