## install

```
conda env create -f env_new.yml
conda activate hmr_updated
```
Find the datasets and models [here](https://polybox.ethz.ch/index.php/s/CyPr4jix2ErtjBk) untar and place them in the hmr_depth folder.

You should end up with the following folders:

- hmr_depth/data
- hmr_depth/egobody_release/egocentric_depth_processed
- hmr_depth/egobody_release/smplx_spin_holo_depth_npz


## training

```
python train.py --model name
```

Note: When using the ConvNeXt backbone it is beneficial to freeze the backbone at some point during training. In our experience this was at about epoch 15 for flows and epoch 30 for diffusion. The default value in the training script is 20, you might want to change it depending on your needs. Also note that training the backbone is memory intensive so you might have to reduce batch size from the default of 64 (size 32 required about 8.5GB in our experience), on the other hand, if the backbone is frozen batch size can be increased.

## eval

```
python eval.py --model name --checkpoint /PATH/TO/MODEL.pt
```

where name corresponds to the iteration of the model you want to train
- *baseline* the baseline as provided by our supervisor
- *backbone* improved model, including fixes, discriminator and ConvNeXt backbone as described in the report
- *diffusion* all of the changes from backbone plus the diffusion head
- *flowmatching* all of the changes from backbone plus the flowmatching head


