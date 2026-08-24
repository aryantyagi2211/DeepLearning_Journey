# YOLOv8 — Architecture

Input → Backbone (Conv stem + C2f x4 + SPPF) → Neck (PANet) → Head

- **Input image** — 640x640x3, raw pixels
- **Conv stem** — stride 2 conv, shrinks image, finds basic edges
- **C2f block (x4)** — splits into Path 1 (bottleneck units, deep refine) + Path 2 (light 1x1 conv, preserve) → concat. Merge happens every run. Keeps gradients healthy.
- **SPPF** — 3x max-pooling (small/mid/big window) + concat + conv → wide context. Only affects P5.
- **P3 / P4 / P5** — 3 outputs from different backbone depths: P3 (80x80, small objs), P4 (40x40, medium), P5 (20x20, large)
- **Neck (PANet)** — top-down: P5 context → P4 → P3. bottom-up: P3 detail → P4 → P5. All 3 end up with detail + context.
- **Head** — 3 decoupled heads (class branch + box branch), anchor-free, one per scale

**Next:** YOLOv8 Training Pipeline
