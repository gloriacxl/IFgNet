# IMPLICIT FOREGROUND-GUIDED NETWORK FOR ANOMALY DETECTION AND LOCALIZATION
This is the code for paper : IMPLICIT FOREGROUND-GUIDED NETWORK FOR ANOMALY DETECTION AND LOCALIZATION

# Datasets
* [VisA dataset](https://link.springer.com/chapter/10.1007/978-3-031-20056-4_23)
* [BTAD dataset](https://ieeexplore.ieee.org/abstract/document/9576231)

# Pretrained Models
We provide our model checkpoints to reproduce the performance report in the papar at:

# Evaluating
The test script requires : \
&#8195;--gpu_id arguments \
&#8195;--data_path the location of the VisA (or BTAD) anomaly detection dataset \
&#8195;--checkpoint_path the folder where the checkpoints are located
```python
python val_forvisa_IFgNet.py
```
# Experimental Results
![image](https://github.com/gloriacxl/IFgNet/blob/main/experimentalresults.PNG)

# Visualization
![image](https://github.com/gloriacxl/IFgNet/blob/main/visualization.png)

# Training
If you want to train a model from scratch, the train script requires : \
&#8195;--gpu_id arguments \
&#8195;--data_path the location of the VisA (or BTAD) anomaly detection dataset \
&#8195;--anomaly_source_path the location of the DTD dataset
```python
python train_forvisa_IFgNet.py
```
