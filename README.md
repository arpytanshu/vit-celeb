# Vision Transformer + Celeb-A dataset

Pytorch implementation of Vision Transformers.  
Trained on Celeb-A dataset for multi-label classification task.

### WIP...

![model summary](assets/model_summary.png)

### Visualizations
Similarity of position embeddings as it evolves over training.  
Tiles show the cosine similarity between the position embedding of the patch with the indicated row and column and the position embeddings of all other patches.
![Similarity of Learned Position Embeddings](assets/pe_similarity.gif)
Interesting thing to note here is, the position embedding corresponding to the 




##### Normalization Stats of dataset.  
    'mean': [0.5061, 0.4254, 0.3828]
    'var': [0.0964, 0.0842, 0.0839]
    'std': [0.3105, 0.2903, 0.2896]




---
## Large-scale CelebFaces Attributes (CelebA) Dataset
![Similarity of Learned Position Embeddings](assets/celebA.png)

    @inproceedings{liu2015faceattributes,
      title = {Deep Learning Face Attributes in the Wild},
      author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
      booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
      month = {December},
      year = {2015} 
    }

