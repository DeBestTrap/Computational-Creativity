# ECSE 4964: Computational Creativity
This course was taught at RPI in the Fall 2023 by Professor Richard Radke.

## Topics covered:
1. Introduction and Background
2. Variational auto-encoders
3. Generative Adversarial Networks
4. Additional generative models
5. Language models
6. Diffusion models for image generation
7. Multimedia generation
8. Neural rendering fields

# Here are some highlights of the homeworks and project!
## HW1: VAEs
Training a conventional VAE on CGI images of legos.

![](./HW1_VAEs/.imgs/vae_interpolation.gif)


## HW2: GANs and Diffusion
Training DCGAN and DDPM models on CGI images of legos.

| DCGAN | DDPM |
|-|-|
| <img src="./HW2_GANs_and_Diffusion/.imgs/gan_interpolation_only_bricks.gif" style="width:200px;"/> | <img src="./HW2_GANs_and_Diffusion/.imgs/diffusion_interpolation_no_aug.gif" style="width:200px;"/> |

## HW3: LLMs
LLaMA-2 finetuning with custom dataset about a made up species called an aonomi:

```
Prompt: What is an aonomi?
Response: Aonomis are a subspecies of Alpaca. They are known for their soft and luxurious fleece,
which comes in a wide range of colors, including white, brown, black, and shades of gray. Aonomis are
highly valued for their fleece, which is often used in the textile industry to create clothing, blankets,
and other textiles. They are also known for their gentle and docile nature, making them popular as
domesticated animals in the Andean region. Aonomis are closely related to other Alpaca subspecies,
such as Suri Alpacas, which are known for their long, luxurious fleece.
```

## HW4: Stable Diffusion Experiments
Using LoRA to finetune models on a novel subject:

<img src="./HW4_Stable_Diffusion/.imgs/radke_lora_v1.5_(2)_prompting_1.png" style="width:200px;"/>
<img src="./HW4_Stable_Diffusion/.imgs/radke_lora_v1.5_(2)_2.png" style="width:200px;"/>

Using ControlNet's OpenPose for novel poses:

<img src="./HW4_Stable_Diffusion/.imgs/radke_lora_v1.5_(2)_prompting_skateboard_2.png" style="width:200px;"/>
<img src="./HW4_Stable_Diffusion/.imgs/radke_lora_v1.5_(2)_prompting_guitar_1.png" style="width:200px;"/>

## HW5: NeRF Experiments
Using NeRFStudio to generate novel NeRFs
<img src="HW5_NeRF/.imgs/water5.gif"/>

Using Instruct-NeRF2NeRF to edit NeRFs:

<img src="HW5_NeRF/.imgs/turn-the-water-to-milk.gif"/>


## Project: Text2Movie