# YOLOv8 — My Journey

*Part of my Deep Learning roadmap: ANN → CNN → **YOLOv8** → RNN → NLP → Transformer → LLM*

---

## How it started

After finishing CNN and Transfer Learning, I moved into Object Detection and picked YOLOv8 to learn properly — not just skim through it. The goal was to understand it fully: how the model sees an image, how it learns, and how it runs live on a camera.

This repo documents that, in the order I actually learned it.

---

## Understanding the architecture

I started with just a diagram, no explanation attached. Had to sit with it, ask questions, get corrected, and rebuild the picture in my own head until it actually made sense.

Here's what I found inside:

- An image goes in as raw numbers, 640x640x3
- The Conv stem takes the first look — just edges, just basic shapes
- The C2f block runs 4 times — splits into a path that keeps refining the features, and a path that keeps the original safe, then merges both. This repeats, going deeper each time.
- SPPF looks at the image at 3 zoom levels — near, mid, far — so it understands small and big objects together
- The backbone hands off 3 outputs — P3, P4, P5 — for small, medium, and large objects
- The Neck (PANet) makes these 3 outputs talk to each other — context flows down, detail flows back up
- The Head finally makes the prediction — what the object is, and where it is — one head per scale

Full notes: `YOLOv8_Architecture_Notes.pdf`

---

## Training the model

Once I understood the architecture, the next question was how this actually learns.

This is where transfer learning came back, same idea from CNN, new context. I didn't start from zero — I started from a model already trained on COCO's 80 classes, and fine-tuned it on my own data.

Here's the pipeline I put together:

- Organize the dataset — images with matching label files
- Write data.yaml — tells YOLO where the data is and what the classes are
- Load the pretrained model
- Train — every step does augmentation, then a forward pass through Backbone → Neck → Head, then loss calculation (CIoU + BCE + DFL), then a backward pass, then the optimizer updates the weights
- Validate — check mAP, precision, recall
- Export — to ONNX, TensorRT, or TFLite for deployment
- Run inference on a new image it had never seen before

Full notes: `YOLOv8_Training_Notes.pdf`

---

## Making it work in real time

The last part was taking a model that works on static images and making it detect objects live.

This tied everything together — same trained model, but now running frame by frame, continuously.

Here's the loop:

- Grab a frame from a webcam, video file, or stream
- Preprocess it — resize and normalize, same as during training
- Run inference — a pure forward pass, no training happening anymore
- Clean up the predictions — confidence filter and NMS, so one object doesn't get multiple overlapping boxes
- Draw the boxes and labels on the frame
- Display it, then loop back for the next frame, until told to stop

Full notes: `YOLOv8_RealTime_Notes.pdf`

---

## Where I'm at

| Part | Status |
|---|---|
| Architecture | Done |
| Training Pipeline | Done |
| Real-time Detection | Core steps done, FPS optimization still pending |

YOLOv8 is at a solid 90% for now — enough to move on, with the option to come back for the loss math by hand and the full combined real-time loop later.

---

## Next up

RNN — next topic on the roadmap.