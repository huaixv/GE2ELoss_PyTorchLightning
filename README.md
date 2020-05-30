# Speaker Verification

## Dependencies

The following packages is required to run the scripts:
- pytorch
- pytorch-lightning
- librosa

## Usage
- Copy the TIMIT dataset to the corresponding folder, like this:
```
TIMIT
├── TEST
│   ├── DR1
│   ├── DR2
│   ├── ...
│   └── DR8
└── TRAIN
    ├── DR1
    ├── DR2
    ├── ...
    └── DR8
```

- Extract MFCC features:
```shell script
python -m src.preprocess
```

- Train the neural network:
```shell script
python -m src.embedder
```

- If you need tune the hyperparameters, edit `src/config.py`