# Unsupervised Learning of View-invariant Action Representations
Unofficial implementation for *J. Li, Y. Wong, Q. Zhao, and M. S. Kankanhalli, “Unsupervised Learning of View-invariant Action Representations", NeurIPS, 2018*.

Many parameters, from data generation to network parameters, are not clearly mentioned in the original paper and I couldn't get answer from the authors, so please keep in mind that this unofficial implementation is coded arbitrarily in many parts.

## Instructions
To run this code, you need following data:
- NTU RGB+D depth images, RGB images, and flows in HDF5 format. Data generation is done with codes in ```dataset``` directory. These files are way too big to be shared, but [here](https://d.pr/f/QVcYNG/iqaQJSFRIN) is a small set of samples!
- NTU RGB+D JSON and label text files containing videoname, video length, and action labels. These are in ```assets``` directory.

Dataset directory looks like:
![Dataset Directory Tree](/assets/dataset.png?raw=true "Dataset directory")

**Example**
To train:
```
# First train without Gradient Reversal
python viar.py --ntu-dir /volume1/dataset/NTU_RGB+D_processed/ --batch-size 8 --num-workers 16 --save-dir /volume1/data/VIAR --disable-grl
# Train with Gradient Reversal
python viar.py --ntu-dir /volume1/dataset/NTU_RGB+D_processed/ --batch-size 8 --num-workers 16 --save-dir /volume1/data/VIAR --checkpoint-files '{"all": "/volume1/data/VIAR/VIAR_Jul22_16-05-53/checkpoints/VIAR_Jul22_16-05-53_Ep1_Iter5040.tar"}' 
````
To test:
```
python viar.py --test --ntu-dir /ssd1/users/mino/dataset/NTU_RGB+D_processed/ --batch-size 8 --num-workers 16 --output-dir /volume3/users/mino/data/VIAR/features/ --checkpoint-files '{"all": "/volume3/users/mino/data/VIAR/VIAR_Jul23_06-16-38/checkpoints/VIAR_Jul23_06-16-38_Ep362_Iter1824480.tar"}'
```

### Observation
![TSNE Result on Setup number 1 and Replication number 1](/assets/figure.png?raw=true "TSNE Result")
Figure above is TSNE result on extracted ConvLSTM features from videos with Setup number 1 and Replication number 1 (540 videos, 6 frames or dots per video, see ```/assets/figure.pdf``` for full resolution image. Blue groups are Camera number 1, orange groups are Camera number 2, and green groups are Camera number 3.
I personally expected to get TSNE result that shows 'view-invariant' property of these features, but TSNE does not seem to be the best way to see such property.

### References
1. J. Li, Y. Wong, Q. Zhao, and M. S. Kankanhalli, “Unsupervised Learning of View-invariant Action Representations", NeurIPS, 2018
2. PD-Flow: https://github.com/MarianoJT88/PD-Flow # modified version included in this repo
2. Convolution RNN: https://github.com/kamo-naoyuki/pytorch_convolutional_rnn # included in this repo
3. Gradient Reversal Layer: https://github.com/janfreyberg/pytorch-revgrad # modified version included in this repo
