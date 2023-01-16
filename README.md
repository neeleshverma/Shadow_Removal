# Shadow Removal via Diffusion Model

Diffusion Denoising Probabilistic Models (DDPM) have recently showcased the benefits over traditional Generative Adversarial Networks (GANs) for several image generation scenarios. This makes us believe that there's a scope for incorporating DDPM into shadow removal tasks. Shadow removal is however different from any image generation task as the model needs to preserve the hidden features while denoising to only produce the most contextual result. This conditionality has been the key motivation to use Diffusion models for shadow removal as recent works have shown promising results of methods to guide image generative process from a reference image.

## ILVR

### ILVR Method:
We initially started with training the Guided Diffusion model on ISTD Dataset. In order for the training to be successful, we accommodated Data Augmentation techniques to increase the dataset size. After training for about 110K Epochs, we used this trained model for further evaluation.

### ILVR pipeline
![ilvr](https://github.com/neeleshverma/Shadow_Removal/blob/main/ilvr%20pipeline.png)

For ILVR, we only changed the reverse diffusion phase where we start from a completely noisy image and gradually reverse the noise effects to generate the sample. An inference image is presented to it at some timestamp; from there the losses are compared against this reference image so the end result generated is closer to this shadow-free reference image passes. This reference image can be passed at different stages of the inference process.

All the ILVR results are present in the ilvr_adm\output directory. Some results are shown below 

### ILVR Results
![ilvr_results](https://github.com/neeleshverma/Shadow_Removal/blob/main/ilvr%20results.png)

## RePaint

### RePaint Method
We had to resize the images for RePaint and also generate inverted masks for the testing set to be used in the denoising step. The process is carried out separately for known and unknown region. The unknown regions is generated by denoising noisy image and generating feature within the shadow region which the known region is made noisy to match the noise levels for blending. This blended image was then used as the input for next iteration. The RePaint pipeline is shown below -  

### RePaint pipeline
![repaint](https://github.com/neeleshverma/Shadow_Removal/blob/main/repaint%20pipeline.png)

### RePaint Results using own ISTD trained model
Some results with our own trained model and tested on RePaint are as follows -  

![istd_trained](https://github.com/neeleshverma/Shadow_Removal/blob/main/repaint%20istd%20results.png)

### RePaint Results using Places2 trained model
To check the generative capacity of our own trained model, we check the results of ISTD testset on Places2 model using RePaint method.  

![places2_trained](https://github.com/neeleshverma/Shadow_Removal/blob/main/repaint%20places2%20result.png)

### Extending RePaint to incorporate prior (Decay Rate)
In order to produce patterns within the shadow region, we incorporated a Decay rate based method to also pass in shadow information. With this method, we are not completely taking the generated image for the shadow region but also using some information from the original image. This decay rate is initially high and forces the blending to be closer to the shadow region features and gradually decreases over diffusion steps to produce smooth reconstruction.

Some of the results are shown here (along with the weight of the prior) -  

![decay_rate](https://github.com/neeleshverma/Shadow_Removal/blob/main/repaint%20decay.png)

The RePaint based implementation and modified RePaint based implementation accomodating weighted decay lies in the RePaint folder with samples available RePaint\log\face_example directory for regular method as well as the Decay method. The decay equation that we used is mentioned below

![decay_rate](https://github.com/neeleshverma/Shadow_Removal/blob/main/decay_eq.png)

where WeightedGT is the noise added shadow image

We also experimented with DDIM models for Shadow removal task. DDIM's inferences times were the fastest like its claims. The code for executing DDIM implementation is in the DDIM.py file

## Report and Presentation slides
The report for the project is attached here [Report](https://github.com/neeleshverma/Shadow_Removal/blob/main/Project_Report.pdf).

The presentation slides of the project is attached here [PPT](https://github.com/neeleshverma/Shadow_Removal/blob/main/Final_Presentation.pptx.pdf).


## References
- The original github repository of [RePaint](https://github.com/andreas128/RePaint) 

- The original github repository of [ILVR (Iterative Latent Variable Refinement)](https://github.com/jychoi118/ilvr_adm).

- The official github repository of [Denoising Diffusion Implicit Models (DDIM)](https://github.com/ermongroup/ddim)

## Note
Contact Info : Neelesh - neverma@cs.stonybrook.edu