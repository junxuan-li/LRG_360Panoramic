# Lighting, Reflectance and Geometry Estimation from 360° Panoramic Stereo

 [Lighting, Reflectance and Geometry Estimation from 360° Panoramic Stereo, CVPR 2021.](http://cvl.ist.osaka-u.ac.jp/wp-content/uploads/2021/03/li_cvpr2021.pdf)


## Dependencies

Environment Requirements:
- Python 3.6
- PyTorch 1.7.1
- OpenCV
- SciPy

With GPU: NVIDIA GeForce GTX 1080 Ti.


## Overview
- The structure of the code is listed as below:
    ```
  ./data  
        -(real | synthetic)
            -(scene_name)/up.png
            -(scene_name)/down.png
  ./Lighting_Estimation
  ./RN_Net
  ./TV_optimization
  ```
  The real and synthetic 360° stereo pair are stored in `data` folder, with naming `up.png` and `down.png`.
  
  The codes for building "Near-field Environment Light" and lighting estimation are in `./Lighting_Estimation`.

  The codes for training and testing "RN-Net" are in `./RN_Net`.
  
  The codes for Rendering and Refinement (Total Variation Refinement) are in `./TV_optimization`.
- We use [Structured3D](https://structured3d-dataset.org/) dataset for training the RN-Net. The trained model of RN-Net is provided in:
  ```
  ./RN_Net/trained_model/net_params.pth
  ```

- We adopt a recent release method: [360SD-Net](https://github.com/albert100121/360SD-Net) for 360° stereo depth estimation. The estimated depth is provided in folder:
  ```
  ./data  
        real
            */depth.npy
        synthetic
            */depth.npy
  ```
  *The `hall` and `room` dataset in `./data/real` is captured by [360SD-Net](https://github.com/albert100121/360SD-Net).*


## Running the code

- Build the "Near-field Environment Light" by running
  ```
  python Lighting_Estimation/depth2points.py
  ```
  It will generate `camera_points.npy`, `hdr_color_points.npy`, `ldr_color_points.npy` in the cooresponding `./data/*/(scene_name)` folders, which will then be used for illumination map estimation.
  To estimate illumination map given any 3D position in the world, please see functions in `./Lighting_Estimation/depth2points.py` for details.

- Estimate the surface normal and coarse reflectance map by running RN-Net:
  ```
  python RN_Net/test_scale_network.py
  ```
  It will generate `output_normal.png`, `output_albedo_coarse.png` in the cooresponding `./data/*/(scene_name)` folders. `output_albedo_coarse.png` will then be used in Total Variation Refinement to get fine result.
  
- **(Optional)** Render the shading
  ```
  python TV_optimization/pixel_env.py
  ```
  The above will per-pixelly estimate all the illumination maps then output to `warped_env_map_part_00**.npy` file. 
  This part of the code is running on CPU. Hence, it may take hours to generate all the illumination maps. 
  We are considering rewriting this part of code to GPU-based in the future.
  Once all the `warped_env_map_part_00**.npy` are generated, run the following to render the shading map:
  ```
  python TV_optimization/shading.py
  ```
  We have included the results of shading in `./data/*/(scene_name)/shading_full.npy` for your convenience. 

- Total Variation Refinement
  ```
  python TV_optimization/albedo_optimise_model.py
  ```
  It will generate `output_albedo_refined.png` in the cooresponding `./data/*/(scene_name)` folders.
  
  



## Citation
If you find this code useful in your reasearch, please consider cite:
```
@inproceedings{li2019learning,
  title={Lighting, Reflectance and Geometry Estimation from 360° Panoramic Stereo},
  author={Li, Junxuan and Li, Hongdong and Matsushita, Yasuyuki},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```

