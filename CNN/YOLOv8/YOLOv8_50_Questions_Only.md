# YOLOv8 — 50 Practice Questions (Questions Only)

*Covers: Architecture theory, dataset/labels, training pipeline, hyperparameters, loss functions, evaluation metrics, inference, export, and real-time deployment.*

---

## Section A: Architecture & Theory (Q1–Q12)

1. What are the three main components of the YOLOv8 architecture?
2. What backbone does YOLOv8 use, and what is its role?
3. What does "CSP" stand for in CSPDarknet, and what problem does it solve?
4. What is the role of the neck (PANet) in YOLOv8?
5. Why does PANet use both top-down and bottom-up feature paths?
6. What does "decoupled head" mean in YOLOv8, and how is it different from earlier YOLO versions?
7. What does "anchor-free" mean, and how does YOLOv8 predict boxes without anchors?
8. Why did YOLOv8 move away from anchor-based detection (used in earlier YOLO versions)?
9. What is the difference between YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, and YOLOv8x?
10. What is transfer learning, and why did we use `yolov8s.pt` instead of training from scratch?
11. What are the two phases of transfer learning (feature extraction vs fine-tuning)?
12. How does YOLOv8 differ architecturally from the R-CNN family (two-stage detectors)?

## Section B: Dataset & Label Format (Q13–Q20)

13. What is the YOLO `.txt` label format, and what does each value represent?
14. Why are bounding box coordinates normalized (0 to 1) instead of using raw pixel values?
15. Given a label row `0 0.42 0.13 0.19 0.26`, how would you convert this to pixel coordinates for a 640×640 image?
16. What folder structure does Ultralytics expect for a custom dataset?
17. What is the purpose of the `data.yaml` file?
18. What bug did we encounter in our `data.yaml`, and how did we fix it?
19. Why does Roboflow sometimes export relative paths that break in Colab?
20. What is the difference between the train, valid, and test splits?

## Section C: Training Setup & Hyperparameters (Q21–Q32)

21. What does the `epochs` parameter control?
22. What does `batch` control, and what happens if it's set too high for your GPU?
23. What does `imgsz` control, and what's the tradeoff between a larger vs smaller value?
24. What is `patience`, and how does early stopping work?
25. What optimizer did Ultralytics select automatically when we used `optimizer="auto"`, and why?
26. What is the difference between SGD and AdamW as optimizers?
27. What does `lr0` represent?
28. What does `momentum` do in an optimizer?
29. What does `weight_decay` help prevent?
30. What is the "mosaic" augmentation, and why is it significant for YOLO specifically?
31. What does `workers` control, and how can a low value cause a training bottleneck?
32. Why did reducing `imgsz` from 640 to 416 speed up our training?

## Section D: Loss Functions (Q33–Q38)

33. What are the three loss components YOLOv8 tracks during training?
34. What does `box_loss` measure, and what loss formula does it use (CIoU)?
35. What does `cls_loss` measure, and what loss formula does it use?
36. What is `dfl_loss` (Distribution Focal Loss), and why is it useful for objects with fuzzy edges (like smoke)?
37. Write the general formula for YOLOv8's total loss in terms of `box_loss`, `cls_loss`, `dfl_loss`, and their weights (λ).
38. What should you expect to see happening to all three losses across epochs if training is going well?

## Section E: Evaluation Metrics (Q39–Q46)

39. What does mAP stand for, and what does it measure?
40. What is the difference between mAP50 and mAP50-95?
41. What does "IoU" (Intersection over Union) measure?
42. Define precision in the context of object detection.
43. Define recall in the context of object detection.
44. In our fire/smoke confusion matrix, what did the high "background" false positive/negative counts tell us about the model?
45. Why did our model rarely confuse fire with smoke directly, but frequently confuse both with "background"?
46. What does it mean if validation loss curves flatten out while training loss keeps decreasing?

## Section F: Inference, Export & Real-Time Deployment (Q47–Q50)

47. What is the difference between `best.pt` and `last.pt`?
48. What does exporting a model to ONNX format allow you to do that the raw `.pt` file doesn't?
49. Why doesn't real-time webcam detection work directly inside Google Colab?
50. Name two ways to optimize a real-time YOLO webcam script for better FPS on a local machine.

---
*Answer key available in the companion file: YOLOv8_50_Questions_Answers.md*
