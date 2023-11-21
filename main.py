import yolov5
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


# load pre-trained model

model_yolo = yolov5.load('keremberke/yolov5m-license-plate')


# set model parameters

model_yolo.conf = 0.25  # NMS confidence threshold
model_yolo.iou = 0.45  # NMS IoU threshold
model_yolo.agnostic = False  # NMS class-agnostic
model_yolo.multi_label = False  # NMS multiple labels per box
model_yolo.max_det = 1000  # maximum number of detections per image


# set image for testing the code 
"""
Put the path to your own image below
"""

img = 'C:/Users/arthu/OneDrive/Documents/Projets_Persos/license_plate_text_extraction/images/test_image_2.jpg'
im = Image.open(img)


# perform yolo inference

results = model_yolo(img, size=640)


# inference with test time augmentation

results = model_yolo(img, augment=True)


# parse results

predictions = results.pred[0]
boxes = predictions[:, :4] 
scores = predictions[:, 4]
categories = predictions[:, 5]


# crop the image using the bounding box coordinates

xmin, ymin, xmax, ymax = boxes[0].tolist()
crop = im.crop((xmin, ymin, xmax, ymax))


# show the cropped plate

crop.show()


# load TrOCR pre-trained model

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model_ocr = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

image = crop.convert("RGB")

pixel_values = processor(image, return_tensors="pt").pixel_values
generated_ids = model_ocr.generate(pixel_values)


# inference with our cropped plate image 

generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
output_text = generated_text[1:]

print(f"The plate text is: {output_text}")