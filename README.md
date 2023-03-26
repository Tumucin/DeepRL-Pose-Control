# panda-gym
- [X] Curriculum learning was added with the help of CustomCallBack function
- [X] Base position was changed again from (0.0, 0.0, 0.0) to (-0.6, 0.0, 0.0)
- [X] Every n iteration, the current model was tested on a testing environment while training. If the rmse and average joint velocity norms are low (these values were hard-coded in CustomCallback.py), then the starting workspace is upgraded to the next workspace.
- [X] When training process finishes, the model is tested with an arbitrary starting and ending points in full workspace.

