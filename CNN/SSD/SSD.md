# What I Learned Today — SSD (Single Shot Detector)

## Summary
Studied how SSD does object detection in a single forward pass (no separate region-proposal stage like Faster R-CNN).

## Key Takeaways
- **Base network**: Pretrained VGG16 (FC layers removed) used as a feature extractor via transfer learning.
- **Extra conv layers (conv6–conv11)**: Progressively shrink the feature map (38×38 → 1×1), increasing receptive field at each stage.
- **Multi-scale detection**: Predictions made on 6 different feature map sizes — large maps catch small objects, small maps catch large objects.
- **Default (anchor) boxes**: 4–6 fixed-shape boxes per cell (square, wide, tall, etc.) act as templates; the network predicts offsets to fit real objects, not boxes from scratch.
- **Conv predictors**: For each default box, predict class scores + 4 box offset values, using the feature vector at that cell.
- **Non-Max Suppression (NMS)**: Removes duplicate overlapping boxes.
  - **Confidence threshold** (~0.5): filters out low-confidence boxes.
  - **IoU threshold** (~0.45): filters out duplicate boxes overlapping a better one.

## SSD vs Faster R-CNN
| | Faster R-CNN | SSD |
|---|---|---|
| Stages | Two-stage | Single-stage |
| Speed | Slower | Real-time |
| Small objects | Better | Weaker |
| Accuracy | Slightly higher | Close, small trade-off |

## To Revisit
- Hand-calculate total predictions for a 5×5 feature map (6 boxes/cell, 21 classes).
- Implement default box generation code.
- Explain why 6 feature maps > just the last one.
