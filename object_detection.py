import torch
import torchvision
import cv2
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
import time
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).cpu()
model.eval()

COCO_INSTANCE_CATEGORY_NAMES = ['__background__','person','bicycle','car','motorcycle','airplane','bus',
                                'train','truck','boat','traffic light','fire hydrant','N/A','stop sign',
                                'parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant',
                                'bear','zebra','giraffe','N/A','backpack','umbrella','N/A','N/A',
                                'handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite',
                                'baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle',
                                'N/A','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich',
                                'orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
                                'potted plant','bed','N/A','dining table','N/A','N/A','toilet','N/A','tv','laptop',
                                'mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator',
                                'N/A','book','clock','vase','scissors','teddy bear','hair drier','toorh brush'
                                ]

def get_prediction(img_path, threshold):
    img = Image.open(img_path)

    transform = T.Compose([T.ToTensor()])
    img = transform(img)
    t0 = time.time()
    img = img.to(device)
    print(f'CPU action took {time.time()-t0}s')

    t0 = time.time()
    pred = model([img])
    print(f'Inference took {time.time()-t0}s')

    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0],i[1]),(i[2], i[3])] for i in list(pred[0]['boxes'].cpu().detach().numpy())]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())

    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class
    
def object_detection_api(img_path, threshold=0.95, rect_th=2, text_size=2, text_th=2):
    boxes , pred_cls = get_prediction(img_path, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(boxes)):
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(255,0,0), thickness=rect_th)
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (255,0,0), thickness=text_th)
    plt.figure(figsize=(10,20))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == "__main__":
    results = object_detection_api('./person.jpg')
    #results = object_detection_api('./highway.jpg')
    #results = object_detection_api('./highway2.jpg')
