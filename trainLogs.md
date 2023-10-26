### train logs:
- all defaults  
commit id `ee9eb44111b20033708a4c3ea2ed79e3f67c0b4c`  
`python trainer.pt --dataset_path /shared/datasets/Celeb-A`

    ```
    2023-10-26 19:30:43,840 - root - INFO - Running evaluation at itern=3750...
    2023-10-26 19:31:06,164 - root - INFO - {'precision': 0.736, 'recall': 0.803, 'F1': 0.752, 'accuracy': 0.795, 'elapsed': 21.565, 'loss': 0.607}
    2023-10-26 19:31:06,164 - root - INFO - Updating best model.
    2023-10-26 19:31:06,165 - root - INFO - loss improved from 0.61 to 0.607
    2023-10-26 19:31:06,855 - root - INFO - Saved checkpoint at /shared/CO/arpytanshu_/celeb-vae/runs/231026-1803/model.th
    ```


- fixed bug, increased dimension of positional embedding from 1 to model_dim=512.  
update default max_iterations to 5000.  
commit id: ``

```sfsdf```