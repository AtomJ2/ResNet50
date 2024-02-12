import cv2
import numpy as np
import torch.nn.functional
import torchvision
from PIL import Image


data = None


def get_feature_map(module, inputs, output):
    global data
    data = output


def normalize_image(img):
    transform_pipe = torchvision.transforms.Compose([
        torchvision.transforms.Resize(
            size=(224, 224)
        ),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    img_tensor = transform_pipe(img)
    batch = img_tensor[None, :, :, :]
    return batch


def calc_impact(col):
    global data
    data = data.squeeze()
    col = col[None, :]
    ans = col @ data.reshape(2048, 49)
    ans = ans.reshape(7, 7).detach().numpy()
    return ans


def scale_img_by_mask(image, imp):
    heatmap = cv2.applyColorMap((imp * 255).astype(np.uint8), cv2.COLORMAP_JET)
    masked_image = cv2.addWeighted(heatmap, 0.5, image, 0.5, 0)
    return masked_image


def parse_img(masked_image, label, probability):
    _, thresholded = cv2.threshold(masked_image, 80, 255, cv2.THRESH_BINARY)
    thresholded = cv2.cvtColor(thresholded, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 10 and h > 10:  # Set minimum width and height for the rectangle
            cv2.rectangle(masked_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 50)
    fontScale = 1.5
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 1

    label_with_prob = f"{label}: {probability:.2f}"
    with_text = cv2.putText(masked_image, label_with_prob,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)

    return with_text


model = torchvision.models.resnet50()
model.load_state_dict(torch.load("weights.pth"))

model.layer4.register_forward_hook(get_feature_map)
model.eval()

with open("imagenet_classes.txt") as f:
    categories = [s.strip() for s in f.readlines()]

cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, image = cap.read()
    if image is None:
        break
    image = Image.fromarray(image)
    batch = normalize_image(image)

    result = model(batch)

    ind_predicted = torch.argmax(result)
    col = model.fc.weight[ind_predicted]
    importance_mask = calc_impact(col)

    importance_mask = importance_mask / np.max(importance_mask)
    importance_mask = np.where(importance_mask > 0, importance_mask, 0)
    importance_mask = cv2.resize(importance_mask, image.size)

    image = np.array(image)
    masked_image = scale_img_by_mask(image, importance_mask)

    label = categories[ind_predicted]
    probability = result[0, ind_predicted].item()
    img_to_show = parse_img(masked_image, label, probability)

    cv2.imshow('Highlighting', img_to_show[:, :1920])
    if cv2.waitKey(1) == ord('q'):
        break
