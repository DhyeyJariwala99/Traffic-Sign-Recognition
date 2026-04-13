Traffic Sign Recognition — CS 5100 Final Project
Dhyey Jariwala

Requirements:
- Python 3, PyTorch, torchvision, scikit-learn, matplotlib

How to run:
1. Open traffic_sign_cnn.ipynb in Google Colab
2. Set runtime to T4 GPU (Runtime → Change runtime type → T4 GPU)
3. Run all cells in order (Runtime → Run all)
4. Cell 1–7: Downloads GTSRB, trains Custom CNN, evaluates, saves outputs
5. Final cell: Trains ResNet18 in two phases, evaluates, saves outputs

All outputs (curves, confusion matrices, checkpoints) save to /checkpoints/
GTSRB dataset downloads automatically via torchvision on first run.

Dataset downloads automatically via torchvision on first run (~300MB)
Expected runtime: ~50 minutes total on T4 GPU