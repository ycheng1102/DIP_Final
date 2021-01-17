Frame interpolation - Softmax Splatting
===

[HackMD Link](https://hackmd.io/@SzhACbEMRiauQwZYIN9zhw/HyFe_bgku/edit)
[Source code Link](https://github.com/ycheng1102/DIP_Final)
> 因為有影片跟 GIF，建議用 HackMD 看 。
## Team members

| Student ID |  Name  |
| :--------: | :----: |
| 107060023  | 黃子恩 |
| 107060020  | 洪瑜成 |
| 107060007  | 王順貴 |

## Problem Description
### Background
There are several frame interpolation tools now. They work well for videos in reality such as films, but they make animation weird in some way.
    Because in animation, the number of keyframe are usually less then the exact fps, i.e., in 24 fps, there are usually 8 to 12 keyframes in one second.
    
![](https://i.imgur.com/WyskUyZ.png)

So we want implement a frame interpolation method which will take the number of keyframe into account.

### Previous work

There are many researchs on frame interpolation, which work well on real videos. However, when the video is artificial (animation), the result turns bad. The reason is animation contains some unreal transformation, overscaled or other skill in drawing. It makes the result of frame interpolation more strange.

### Motivation

Some frame rate of video are not good, and makes the video looks like combination of uncontinuous image. Thus, we want to increase frame rate, and then, the video will looks more smooth.

## Workflow

### Assumption

videos that we focus is animation or artificial videos
<!-- * Steps
    1. Sample original video in it frame rate
    2. Mark out the keyframe
    3. Use machine learning method to estimated the light flow
    4. interpolate some extra frame, but also take the distribution of keyframes into account
    5. recombine the result into video
 -->

### Steps
![](https://i.imgur.com/cziAk67.png)

* **Step 1**
    Input a video. Different format is fine (mp4, avi, wmv...).
* **Step 2**
    Read the video frame by frame.
* **Step 3**
    Capture the keyframes of the video.
![](https://i.imgur.com/WitOjPF.gif)
Looply compare every 2 adjacent frames. If the number of different pixels of 2 frames smaller than a specific threshold, the second frame will be dicarded.
![](https://i.imgur.com/9FE7NhM.png)
For instance, look at the picture given above, and suppose threshold = 1234. 
  **Difference between frame 1 and frame 2 :** 2021 > threshold :heavy_check_mark: 
  **Difference between frame 3 and frame 4 :** 666 < threshold :x:
  
  The value of threshold is typically range from **(0 ~ pixel # of a frame)**.
  Threshold is **0** by default. i.e., preserves all the frames of a video.
* **Step 4**
Looply generate the interpolated frame  of every 2 adjacent frames. Details are given below.
    * **Flow estimation :**
Input 2 parent frames $I_0,\ I_1$, and use PWC-Net to calculate flow $F_{0→1},\ F_{1→0}$
    * **Extract feature pyramid :**
For each parent frames, extract the feature pyramid with 3 different size (32, 64, 96) after 6 layers of convolution.
    * **softmax splatting ：**
Perform softmax splatting on flow, feature pyramid, and original frame under the constraints of mask Z, and use U-net for fine-tuning.
    * **Interpolation :**
Generate the interpolated frame by GridNet.
    #### **Softmax splatting**
    ![](https://i.imgur.com/Tg61OkX.png)
    1. In order to split foreground and background of movement, apply an improvement like softmax to mask Z.
    2. Accroding to the spacial-invariant property of mask Z, seperate the overlapped areas.
    3. The cost of evaluating pixel depth is too expensive and not stable for accurate result. So use the consistency of intensity to achieve this.
    #### **Mask Z**
    ![](https://i.imgur.com/Pf9cBQD.png)
    1. Use backward warping compute consistency of intensity.
    2. Generate mean by smaller α, and the z-buffering by bigger α.
    3. Because the differentiable of softmax function, we can use tiny neural network to improve the result.
    ![](https://i.imgur.com/8rTsbfE.png)
    ![](https://i.imgur.com/vTTv0lS.png)
    ![](https://i.imgur.com/Tz36o9i.png)
    The insertion method of the interpolated frames is given below.
    ![](https://i.imgur.com/Knmwrg8.png)
    Suppose we wish to extend a 24fps video to a 72fps video. For every 2 adjacent frames, 2 interpolated frames must be generated and inserted between 2 original frames.
* **Step 5**
There are always some black areas in interpolated frames, our soulution is blending the pixel of first and second frames, if the area is black

* **Step 6**
Make video via the original and interpolated frames.


### Usage of our code
* Clone our git respository. (remove **'!'** if needed)
```gherkin=
!git clone 'https://github.com/ycheng1102/DIP_Final'
```
* Run **run.py**. (remove **'!'** if needed)
```gherkin=
!python run.py --video ./sample_1.mp4 --flow ./flow.flo
```
* All flags
    * **\-\-video** : path of input video. No default value.
    * **\-\-flow** : output flow file. Default = './out.flo'.
    * **\-\-second** : duration of the output video. Default = 24.
    * **\-\-width** : width of the output video. Default = 640.
    * **\-\-height** : height of the output video. Default = 360.
    * **\-\-fps** : factor of multiple of fps.
    * **\-\-threshold** : threshold to find the keyframes.

## Demo
Initially, we got a bad result on interpolation. Look at the picture given below. We can see that there are lots of black pixels at the upper boader and around the animated charactor in the interpolated frame.
![](https://i.imgur.com/mANmZvK.png)
To fix the unexpected black pixels in the interpolated frame, we replaced them with the mean value of 2 parent frames. Look at the picture given below, the new interpolated frame looks better now.
![](https://i.imgur.com/n6ZDHMD.png)

**Demo A :** Fix black pixels
Please set the resolution to **720p60** in youtube.
{%youtube KKGpK-VcXiE %}
>Direct link : https://youtu.be/KKGpK-VcXiE

**Demo B :** Video frame interpolation
Please set the resolution to **720p60** in youtube.
{%youtube SnZf4uKXxLU %}
>Direct link : https://youtu.be/SnZf4uKXxLU

## Further Application

* Increase process speed
* with/without motion blur

## Reference
### Softmax splatting
This is a reference implementation of the softmax splatting operator, which has been proposed in Softmax Splatting for Video Frame Interpolation, using PyTorch. Softmax splatting is a well-motivated approach for differentiable forward warping. It uses a translational invariant importance metric to disambiguate cases where multiple source pixels map to the same target pixel. Should you be making use of our work, please cite our paper .

<a href="https://arxiv.org/abs/2003.05534" rel="Paper"><img src="http://content.sniklaus.com/softsplat/paper.jpg" alt="Paper" width="100%"></a>

> Source code : https://github.com/sniklaus/softmax-splatting
### PWC-Net
This is a personal reimplementation of PWC-Net using PyTorch. Should you be making use of this work, please cite the paper accordingly. Also, make sure to adhere to the <a href="https://github.com/NVlabs/PWC-Net#license">licensing terms</a> of the authors. Should you be making use of this particular implementation, please acknowledge it appropriately.


<a href="https://arxiv.org/abs/1709.02371" rel="Paper"><img src="https://i.imgur.com/dZCPrrB.png" alt="Paper" width="100%"></a>

> Source code : https://github.com/sniklaus/pytorch-pwc
