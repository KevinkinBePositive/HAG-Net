# HAG - Net: An Efficient Hierarchical Attention - Guided Network for Robust Green Crop Segmentation

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-repo/HAG-Net.git
cd HAG-Net
```
2. Install the required dependencies.

## Usage
### Data Preparation
- **Backbone Pretrained Weights**: You can download the Res2Net pre - trained weights file `res2net50_v1b_26w_4s-3cf99910.pth` from [Google Drive](https://drive.google.com/file/d/15g4Xr7s7nLCsqK0lYI8sEdG00G8Sndw2/view?usp=drive_link). After downloading, place it in the `data/backbone_ckpt` directory.
- **Model Checkpoint**: Download the pre - trained model checkpoint `latest.pth` from [Google Drive](https://drive.google.com/file/d/1nEOS_Dr2iDUfZ034wIvheXljZQECk7Lm/view?usp=drive_link). Then place it in the `output/HAGNet` directory.


For inference, you can use the following code:
```python
# Test the model
python Test.py --config configs/HAGNet-L.yaml
```

## Directory Structure
```
HAG-Net/
├── data/
│   └── backbone_ckpt/
│       └── res2net50_v1b_26w_4s-3cf99910.pth
├── lib/
│   ├── HAGNet.py
│   ├── backbones/
│   │   ├── Res2Net_v1b.py
│   │   └── ResNet.py
│   ├── modules/
│   └── optim/
├── output/
│   └── HAGNet/
│       └── latest.pth
├── .gitignore
└── README.md
```

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
