### train logs:
- all default args.  
commit id `ee9eb44111b20033708a4c3ea2ed79e3f67c0b4c`  
`python trainer.pt --dataset_path /shared/datasets/Celeb-A`

    ```
    2023-10-26 19:30:43,840 - root - INFO - Running evaluation at itern=3750...
    2023-10-26 19:31:06,164 - root - INFO - {'precision': 0.736, 'recall': 0.803, 'F1': 0.752, 'accuracy': 0.795, 'elapsed': 21.565, 'loss': 0.607}
    2023-10-26 19:31:06,164 - root - INFO - Updating best model.
    2023-10-26 19:31:06,165 - root - INFO - loss improved from 0.61 to 0.607
    2023-10-26 19:31:06,855 - root - INFO - Saved checkpoint at /shared/CO/arpytanshu_/celeb-vae/runs/231026-1803/model.th
    ```

- fix bug with dimension of pos_emb in patchEmbedding.  
update default max_iter from 4000 -> 5000.  
commit id: `992a0c3b3c97ce0d1c6b88dca822f2c2a6d1991d`  
overfitting???
    ```
    2023-10-26 23:15:21,335 - root - INFO - itern=3300 tr_loss=0.46519675850868225
    2023-10-26 23:15:21,335 - root - INFO - Running evaluation at itern=3300...
    2023-10-26 23:15:43,634 - root - INFO - {'precision': 0.777, 'recall': 0.846, 'F1': 0.798, 'accuracy': 0.836, 'elapsed': 21.542, 'eval_loss': 0.503}
    2023-10-26 23:15:43,634 - root - INFO - Updating best model.
    2023-10-26 23:15:43,634 - root - INFO - loss improved from 0.507 to 0.503
    2023-10-26 23:15:44,325 - root - INFO - Saved checkpoint at runs/231026-2230/model.th
    ```

- implemented dropout in patchEmbedding module.  
updated default dropout from 0.1 to 0.15, (possible will allow to train for longer w/o OF.)  
commit id: `47ce940afb38a6c539639f782b67012ad9e1c5bf`
    ```
    ...
    ```

- initialized positional embeddings with 0 made all the difference.  
started getting images of faces in pe cosine similarity plots.  
commit id: `339fbbaf80e7e152fe9a59dde67a147fbfc500c4`  
    ```
    2023-10-28 00:04:16,167 - root - INFO - itern=3750 tr_loss=0.49622440338134766
    2023-10-28 00:04:16,167 - root - INFO - Running evaluation at itern=3750...
    2023-10-28 00:04:38,486 - root - INFO - {'precision': 0.771, 'recall': 0.838, 'F1': 0.791, 'accuracy': 0.831, 'elapsed': 21.558, 'eval_loss': 0.542}
    2023-10-28 00:04:44,428 - root - INFO - Updating best model.
    2023-10-28 00:04:44,428 - root - INFO - loss improved from 0.544 to 0.542
    2023-10-28 00:04:45,145 - root - INFO - Saved checkpoint at runs/231027-2310/model.th
    ```

- increased dropout
    `python trainer.py --dataset_path /shared/datasets/Celeb-A/ --eval_interval 200 --plot_interval 50 --dropout 0.2`