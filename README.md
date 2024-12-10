# Face Relighting with Fidelity Preservation 

![Alt text](https://github.com/andrewhou1/GeomConsistentFR/blob/main/CVPR2022_relighting_video_final.gif)

The code for this project was developed using Python 3.8.8 and PyTorch 1.7.1. 

## Training Data 

The training dataset can be downloaded from: https://github.com/switchablenorms/CelebAMask-HQ

These images will need to be cropped and will go into their own folder **MP_data/CelebA-HQ_DFNRMVS_cropped/**

For the cropping code specifically, we will need to create a separate conda environment to ensure that the images are cropped consistently to match the rest of the training data (albedo, depth maps, etc.) Note that this conda environment is different from the conda environment we will use for training and testing. 

To set up the conda environment for cropping the CelebA-HQ images, run the following:
```
conda create --name PortraitRelighting --file cropping_dependencies.txt
conda activate PortraitRelighting
pip install scipy
```
Once this environment is set up, place all of the  images that you downloaded in a directory named **input_image_dir/** and make a second directory called **output_image_dir/** which will store the cropped images. To crop the images, run the following command: **CUDA_VISIBLE_DEVICES=0 python recrop_CelebA-HQ_images.py**

## Create the conda environment for training and testing
```
conda create --name PortraitRelighting --file training_dependencies.txt
conda activate PortraitRelighting
pip3 install opencv-python
pip3 install kornia==0.4.1
pip install scipy
pip install imageio
```
## Training 
Once you've all of the training data  (albedo, depth, etc.) and finished cropping the  images as described above, move the cropped images from **output_image_dir/** to **MP_data/Relighting_cropped/**

To train the target lighting model, where the user can specify the desired lighting direction, make two additional directories to store the loss values and saved model weights: **losses_raytracing_relighting_CelebAHQ_DSSIM_8x/** and **saved_epochs_raytracing_relighting_CelebAHQ_DSSIM_8x/**

Finally, you can train the target lighting model using the following command: **CUDA_VISIBLE_DEVICES=0 python train_raytracing_relighting_CelebAHQ_DSSIM_8x.py**

If you instead would like to train the lighting transfer model, make the following two directories instead: **losses_lighting_transfer/** and **saved_epochs_lighting_transfer/**

To train the lighting transfer model, use the following command: **CUDA_VISIBLE_DEVICES=0 python train_lighting_transfer.py**

Be sure to use the correct conda environment (PortraitRelighting) for training. 

## Testing 
To run our testing code where the user specifies the target lighting, use the following command: **CUDA_VISIBLE_DEVICES=0 python test_relight_single_image.py**

You can specify both an input image and a target lighting direction within the code (we provide some sample lightings to generate the results in **FFHQ_relighting_results/**). You will also need a face mask for the input image (some examples are shown in **FFHQ_skin_masks/**). To generate a face mask, one option is to use the following repo: https://github.com/zllrunning/face-parsing.PyTorch 


To run testing code for lighting transfer, first make a directory called **lighting_transfer_results/** and then use the following command: **CUDA_VISIBLE_DEVICES=0 python test_relight_single_image_lighting_transfer.py input_image reference_image face_mask**

Be sure to use the correct conda environment (PortraitRelighting) for testing. 


## Contact 
If there are any questions, please feel free to post here or contact the first author at **deepu.raveendran7@gmail.com** 
