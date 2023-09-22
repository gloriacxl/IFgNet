# IMPLICIT FOREGROUND-GUIDED NETWORK FOR ANOMALY DETECTION AND LOCALIZATION
This is the code for paper : IMPLICIT FOREGROUND-GUIDED NETWORK FOR ANOMALY DETECTION AND LOCALIZATION

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
