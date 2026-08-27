# YOLOv8 — 50 Practice Questions with Answers

*Covers: Architecture theory, dataset/labels, training pipeline, hyperparameters, loss functions, evaluation metrics, inference, export, and real-time deployment.*

---

## Section A: Architecture & Theory

**1. What are the three main components of the YOLOv8 architecture?**
Backbone (feature extraction), Neck (feature fusion across scales), and Head (final predictions — boxes, classes, objectness).

**2. What backbone does YOLOv8 use, and what is its role?**
CSPDarknet. It extracts features from the input image at multiple scales/depths, forming the foundation the rest of the network builds on.

**3. What does "CSP" stand for in CSPDarknet, and what problem does it solve?**
Cross Stage Partial. It splits feature maps and merges them later, reducing computation/redundant gradients while preserving accuracy — makes the backbone more efficient.

**4. What is the role of the neck (PANet) in YOLOv8?**
It fuses feature maps from different backbone stages (different resolutions) so the model can detect objects of different sizes effectively.

**5. Why does PANet use both top-down and bottom-up feature paths?**
Top-down passes semantic (high-level, "what is this") information from deep layers to shallow layers. Bottom-up passes fine spatial detail back up. Combining both gives strong features at every scale.

**6. What does "decoupled head" mean in YOLOv8, and how is it different from earlier YOLO versions?**
The classification task and the box-regression task are handled by two separate branches instead of one shared branch. Earlier YOLO versions used a single coupled head for both, which hurt performance since the two tasks need different features.

**7. What is "anchor-free" detection, and how does YOLOv8 predict boxes without anchors?**
Instead of relying on predefined anchor box shapes, YOLOv8 directly predicts the center point and distance to each box edge for each location on the feature map.

**8. Why did YOLOv8 move away from anchor-based detection?**
Anchor boxes required manual tuning of anchor sizes/ratios per dataset and added complexity. Anchor-free simplifies the pipeline and generalizes better across varied object shapes.

**9. What is the difference between YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, and YOLOv8x?**
They're the same architecture at increasing size/depth/width — n (nano) is fastest/least accurate, x (extra-large) is slowest/most accurate. Choice depends on speed vs accuracy needs and hardware.

**10. What is transfer learning, and why did we use `yolov8s.pt` instead of training from scratch?**
Transfer learning reuses weights already learned on a large dataset (COCO) instead of starting from random weights. `yolov8s.pt` already knows general visual features (edges, shapes, textures), so we only need to fine-tune it for fire/smoke — much faster and needs less data than training from scratch.

**11. What are the two phases of transfer learning?**
Feature extraction (freeze pretrained layers, train only the new head) and fine-tuning (unfreeze some/all layers and train at a lower learning rate to adapt pretrained features to the new task).

**12. How does YOLOv8 differ architecturally from the R-CNN family?**
R-CNN family is two-stage (region proposal, then classification/refinement) — slower but historically more accurate. YOLOv8 is single-stage/one-shot — predicts boxes and classes in one pass, much faster, suitable for real-time use.

---

## Section B: Dataset & Label Format

**13. What is the YOLO `.txt` label format?**
`class_id x_center y_center width height` — one row per object in the image, all values except class_id normalized between 0 and 1.

**14. Why are bounding box coordinates normalized (0 to 1)?**
So labels work regardless of the image's actual resolution — the same normalized label is valid whether the image is 640×640 or 1920×1080.

**15. Given `0 0.42 0.13 0.19 0.26` for a 640×640 image, convert to pixels.**
x_center = 0.42×640 ≈ 269px, y_center = 0.13×640 ≈ 83px, width = 0.19×640 ≈ 122px, height = 0.26×640 ≈ 166px.

**16. What folder structure does Ultralytics expect?**
```
dataset/
├── train/images, train/labels
├── valid/images, valid/labels
├── test/images, test/labels
└── data.yaml
```

**17. What is the purpose of `data.yaml`?**
It tells YOLO the class names, number of classes, and the paths to train/val/test image folders.

**18. What bug did we encounter in our `data.yaml`, and how did we fix it?**
Roboflow exported relative paths (`../train/images`) that resolved incorrectly in Colab's `/content/` structure. We fixed it by rewriting the yaml with absolute paths using a Python script.

**19. Why does Roboflow sometimes export relative paths that break in Colab?**
Because the export assumes a certain folder structure/working directory that doesn't always match where the dataset actually gets unzipped in Colab.

**20. What is the difference between the train, valid, and test splits?**
Train = data the model learns from. Valid = data used during training to check generalization and guide early stopping/hyperparameter decisions, not used for weight updates. Test = held-out data used only for final evaluation, never seen during training.

---

## Section C: Training Setup & Hyperparameters

**21. What does `epochs` control?**
The number of complete passes through the entire training dataset.

**22. What does `batch` control, and what happens if it's too high?**
Number of images processed together before one weight update. Too high for available GPU memory causes a CUDA out-of-memory error.

**23. What does `imgsz` control, and what's the tradeoff?**
Input image resolution fed to the network. Larger = more detail, better accuracy, slower training/inference. Smaller = faster, slightly less accurate.

**24. What is `patience`, and how does early stopping work?**
Number of epochs to wait without validation improvement before stopping training automatically, to save time/compute once the model has converged.

**25. What optimizer did Ultralytics select with `optimizer="auto"`, and why?**
AdamW — Ultralytics' auto-selection heuristic picks it as a well-performing default for most detection tasks, adjusting learning rate/momentum accordingly.

**26. What is the difference between SGD and AdamW?**
SGD uses a single global learning rate and plain gradient steps (optionally with momentum). AdamW adapts the learning rate per-parameter using running estimates of gradient mean/variance, plus decoupled weight decay — generally converges faster and needs less manual tuning.

**27. What does `lr0` represent?**
The initial learning rate — controls the step size of weight updates at the start of training.

**28. What does `momentum` do in an optimizer?**
It smooths weight updates by factoring in the direction of previous updates, helping avoid oscillation and speeding convergence.

**29. What does `weight_decay` help prevent?**
Overfitting — it's an L2 regularization term that penalizes large weight values.

**30. What is "mosaic" augmentation, and why is it significant for YOLO?**
It stitches 4 training images together into one, forcing the model to detect objects at varied scales/positions/contexts in a single image — YOLO's signature augmentation, greatly improves robustness.

**31. What does `workers` control, and how can a low value cause a bottleneck?**
Number of CPU threads loading/preprocessing data in parallel. If too low, the GPU sits idle waiting for the next batch of data, slowing training even if the GPU itself is fast.

**32. Why did reducing `imgsz` from 640 to 416 speed up our training?**
Fewer pixels means less computation per forward/backward pass through the network, directly reducing time per epoch.

---

## Section D: Loss Functions

**33. What are the three loss components YOLOv8 tracks?**
`box_loss`, `cls_loss`, `dfl_loss`.

**34. What does `box_loss` measure?**
Error in predicted bounding box position/size vs ground truth, using CIoU loss (overlap + aspect ratio + center-point distance).

**35. What does `cls_loss` measure?**
Error in predicted class per box (e.g., fire vs smoke), using binary cross-entropy.

**36. What is `dfl_loss`, and why is it useful for fuzzy-edged objects like smoke?**
Distribution Focal Loss — predicts a probability distribution over possible box-edge locations instead of a single number, sharpening it toward the correct value. This helps precision for objects without hard, well-defined boundaries.

**37. Write the total loss formula.**
`Loss = λ_box · box_loss + λ_cls · cls_loss + λ_dfl · dfl_loss`

**38. What should all three losses do across epochs if training is going well?**
Trend downward (decrease) consistently, without major spikes or plateaus early on.

---

## Section E: Evaluation Metrics

**39. What does mAP stand for, and what does it measure?**
Mean Average Precision — the overall detection accuracy of the model, averaged across all classes.

**40. What is the difference between mAP50 and mAP50-95?**
mAP50 measures accuracy at a single, lenient IoU threshold (0.5). mAP50-95 averages accuracy across multiple stricter IoU thresholds (0.5 to 0.95), giving a harder, more realistic measure.

**41. What does IoU measure?**
Intersection over Union — the overlap between the predicted box and the ground-truth box, divided by their combined area. Higher IoU = better box alignment.

**42. Define precision.**
Of all the boxes the model predicted, what percentage were actually correct (low false positives = high precision).

**43. Define recall.**
Of all the real objects that existed, what percentage did the model successfully detect (low false negatives = high recall).

**44. In our confusion matrix, what did the high "background" false positive/negative counts tell us?**
The model was reasonably good at distinguishing fire from smoke when it detected something, but struggled with the boundary of "is something here at all" — frequently hallucinating detections in empty scenes and missing real fire/smoke instances.

**45. Why did our model rarely confuse fire with smoke, but frequently confuse both with background?**
Fire and smoke are visually distinct from each other (color, texture), so the model learned to separate those two classes well. But distinguishing "object present" vs "nothing here" requires more training/confidence calibration, which our reduced epoch count (20 instead of 50) didn't fully achieve.

**46. What does it mean if validation loss flattens while training loss keeps decreasing?**
A sign of overfitting — the model is continuing to fit the training data more closely but isn't generalizing better to unseen data.

---

## Section F: Inference, Export & Real-Time Deployment

**47. What is the difference between `best.pt` and `last.pt`?**
`best.pt` is the checkpoint with the highest validation performance (mAP) during training. `last.pt` is simply the checkpoint from the final epoch, regardless of whether it was the best-performing one.

**48. What does exporting to ONNX allow you to do?**
Run the model across different frameworks/languages/platforms outside Python/Ultralytics, and often get faster, more portable inference for deployment.

**49. Why doesn't real-time webcam detection work directly inside Google Colab?**
Colab runs on a remote cloud machine with no direct access to your local computer's webcam hardware without complex browser/JS workarounds — and even then it's laggy.

**50. Name two ways to optimize a real-time YOLO webcam script for better FPS locally.**
Any two of: run inference on GPU instead of CPU (`model.to("cuda")`), reduce `imgsz` for faster per-frame computation, or use frame skipping (only run detection every Nth frame, reuse the last result on skipped frames).

---
*Companion file (questions only, no answers): YOLOv8_50_Questions_Only.md*
