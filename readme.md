# Traffic Sign Recognition Using CNNs

A deep learning project comparing a custom CNN and ResNet18 transfer 
learning on the German Traffic Sign Recognition Benchmark (GTSRB).

## Results
| Model       | Test Accuracy | Macro F1 |
|-------------|--------------|----------|
| Custom CNN  | 95.87%       | 0.934    |
| ResNet18    | 86.83%       | 0.814    |

## How to Run
1. Open `traffic_sign_cnn.ipynb` in Google Colab
2. Set runtime to T4 GPU
3. Run all cells — dataset downloads automatically

## Tech Stack
Python · PyTorch · torchvision · scikit-learn · Google Colab (T4 GPU)

## Dataset
[GTSRB](https://benchmark.ini.rub.de/) — 43 classes, 50,000+ images
