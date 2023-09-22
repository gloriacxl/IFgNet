# IMPLICIT FOREGROUND-GUIDED NETWORK FOR ANOMALY DETECTION AND LOCALIZATION
This is the code for paper : IMPLICIT FOREGROUND-GUIDED NETWORK FOR ANOMALY DETECTION AND LOCALIZATION

# Datasets
[VisA](https://link.springer.com/chapter/10.1007/978-3-031-20056-4_23)
[BTAD](https://ieeexplore.ieee.org/abstract/document/9576231)

# Pretrained Models
We provide four model checkpoints to reproduce the performance report in the papar at:

# Evaluating
The test script requires : \
--gpu_id arguments \
--data_path the location of the VisA (or BTAD) anomaly detection dataset \
--checkpoint_path the folder where the checkpoint files are located
```python
python val_forvisa_IFgNet.py
```
# Visualization
![image](https://github.com/gloriacxl/IFgNet/blob/main/visualization.png)

# Training
If you want to train a model from scratch, the train script requires : \
--gpu_id arguments \
--data_path the location of the VisA (or BTAD) anomaly detection dataset \
--anomaly_source_path the location of the DTD dataset
```python
python train_forvisa_IFgNet.py
```
