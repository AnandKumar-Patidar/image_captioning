# Image Captioning Using CNN and RNN

This project implements an image captioning system using a CNN as the image encoder and an LSTM-based RNN as the caption decoder. The model is trained on the COCO dataset to generate captions for a given image.

## Project Structure

```
image-captioning/
│
├── data/                             # Store or link to your dataset
│   ├── captions_train2017.json        # COCO annotations (if downloading locally)
│   └── images/                        # Store images (if downloading locally)
│
├── models/                           # Model files
│   ├── encoder.py                    # CNN encoder module
│   ├── decoder.py                    # RNN decoder module
│   └── train.py                      # Training script
│
├── notebooks/                        # Jupyter notebooks for exploration
│   └── Image_Captioning.ipynb        # Jupyter notebook demonstration
│
├── utils/                            # Utility functions
│   ├── vocab.py                      # Vocabulary processing
│   ├── coco_dataset.py               # COCO Dataset handling
│   └── data_loader.py                # DataLoader utilities
│
├── checkpoints/                      # Store model checkpoints
│   └── best_model.pth                # Best model weights
│
├── requirements.txt                  # List of dependencies
├── README.md                         # Project description and setup instructions
├── train.py                          # Main training script
└── eval.py                           # Script to evaluate model performance
```

```bash
git clone https://github.com/yourusername/image-captioning.git
cd image-captioning
pip install -r requirements.txt
```
## Usage
Training the Model
To train the image captioning model, run the following command:

```
python3 train.py
```
### Evaluating model
```
python3 eval.py
```