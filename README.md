# ML2020

Project for Machine learning 2020.

## Requirements

After cloning this repository, you can start by creating a virtual environment and installing the requirements by running:

```bash
conda create --n KP2D
conda activate KP2D
pip install -r requirements.txt
```

## Folder Structure

The project is created by the [pytorch template](https://github.com/victoresque/pytorch-template), please see [here](https://github.com/victoresque/pytorch-template) for more details.

  ```
  pytorch-template/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  ├── demo.py
  │
  ├── config.json - holds configuration for training
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   |── data_loaders.py
  │   └── mydatasets.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```

## Usage
The code in this repo is an MNIST example of the template.
Try `python train.py -c config.json` to train model, and use `python demo.py -c config.json` to run demo.

