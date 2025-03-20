<br>
<p align="center">
<h1 align="center"><strong> DexGraspAnything:TowardsUniversalRoboticDexterousGrasping
 withPhysicsAwareness
</strong></h1>
  <p align="center">
      <strong><span style="color: red;">CVPR 2025</span></strong>
    <br>
   <a href='https://ymzhong66.github.io' target='_blank'>Yiming Zhong*</a>&emsp;
   <a href='https://github.com/Kenny-K' target='_blank'>Qi Jiang*</a>&emsp;
   <a href='https://faculty.sist.shanghaitech.edu.cn/yujingyi' target='_blank'>Jingyi Yu</a>&emsp;
   <a href='https://yuexinma.me' target='_blank'>Yuexin Ma</a>&emsp;
    <br>
    ShanghaiTech University    
    <br>
    *Indicates Equal Contribution
    <br>
  </p>
</p>

  

<p align="center">
  <a href="https://dexgraspanything.github.io/"><b>📖 Project Page</b></a> |
  <a href="https://arxiv.org/pdf/2503.08257"><b>📄 Paper Link</b></a> |
</p>

</div>

>  We present DexGrasp Anything, consistently surpassing previous dexterous grasping generation methods across five benchmarks. Visualization of our method's results are shown on the left.

<div align="center">
    <img src="image1.png" alt="Directional Weight Score" class="blend-img-background center-image" style="max-width: 100%; height: auto;" />
</div>

## 📣 News
- [2/27/2025] 🎉🎉🎉DexGraspAnything has been accepted by CVPR 2025!!!🎉🎉🎉

## 😲 Results
Please refer to our [homepage](https://dexgraspanything.github.io/) for more thrilling results!

# 📚 Data
<!-- You could generate demonstrations by yourself using our provided expert policies.  Generated demonstrations are under `$YOUR_REPO_PATH/3D-Diffusion-Policy/data/`.
- Download Adroit RL experts from [OneDrive](https://1drv.ms/u/s!Ag5QsBIFtRnTlFWqYWtS2wMMPKNX?e=dw8hsS), unzip it, and put the `ckpts` folder under `$YOUR_REPO_PATH/third_party/VRL3/`.
- Download DexArt assets from [Google Drive](https://drive.google.com/file/d/1DxRfB4087PeM3Aejd6cR-RQVgOKdNrL4/view?usp=sharing) and put the `assets` folder under `$YOUR_REPO_PATH/third_party/dexart-release/`. -->

<!-- **Note**: since you are generating demonstrations by yourselves, the results could be slightly different from the results reported in the paper. This is normal since the results of imitation learning highly depend on the demonstration quality. **Please re-generate demonstrations if you encounter some bad demonstrations** and **no need to open a new issue**. -->

## 🛠️ Setup
- 1. Create a new `conda` environemnt and activate it.

    ```bash
    conda create -n DGA python=3.8
    conda activate DGA
    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
    ```

- 2. Install the required packages.
    You can change TORCH_CUDA_ARCH_LIST according to your GPU architecture.
    ```bash
    TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6" pip install -r requirements.txt
    ```
    Please install in an environment with a GPU, otherwise it will error.
    ```bash
    cd src
    git clone https://github.com/wrc042/CSDF.git
    cd CSDF
    pip install -e .
    cd ..
    git clone https://github.com/facebookresearch/pytorch3d.git
    cd pytorch3d
    git checkout tags/v0.7.2  
    FORCE_CUDA=1  TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6"  python setup.py install
    cd ..
    ```
- 3. Install the Isaac Gym
    Follow the [official installation guide](https://developer.nvidia.com/isaac-gym) to install Isaac Gym and its dependencies.
    You will get a folder named `IsaacGym_Preview_4_Package.tar.gz` put it in ./src/IsaacGym_Preview_4_Package.tar.gz
    ```bash
    tar -xzvf IsaacGym_Preview_4_Package.tar.gz
    cd isaacgym/python
    pip install -e .
    ```

Before training and testing, please ensure that you set the dataset path, model size, whether to use LLM, sampling method, and other parameters in `configs`.

### Train

- Train with a single GPU

    ```bash
    bash scripts/grasp_gen_ur/train.sh ${EXP_NAME}
    ```

- Train with multiple GPUs

    ```bash
    bash scripts/grasp_gen_ur/train_ddm.sh ${EXP_NAME}
    ```

### Sample

```bash
bash scripts/grasp_gen_ur/sample.sh ${exp_dir} [OPT]
# e.g., Running without Physics-Guided Sampling:   bash scripts/grasp_gen_ur/sample.sh /outputs/exp_dir [OPT]
# e.g., Running with Physics-Guided Sampling:   bash scripts/grasp_gen_ur/sample.sh /outputs/exp_dir OPT
```
- `[OPT]` is an optional parameter for Physics-Guided Sampling.

### Test 

First, you need to run `scripts/grasp_gen_ur/sample.sh` to sample some results. 
You also need to set the dataset file paths in `/envs/tasks/grasp_test_force_shadowhand.py` and /scripts/grasp_gen_ur/test.py`. 
Then, we will compute quantitative metrics using these sampled results.

```bash
bash scripts/grasp_gen_ur/test.sh ${EVAL_DIR} 
# e.g., bash scripts/grasp_gen_ur/test.sh  /outputs/exp_dir/eval/final/2025-03-16_19-15-31
```


## 🚩 Plan
- [x] Paper Released.
- [x] Source Code.
- [ ] Dataset.
- [ ] Checkpoints.
<!-- --- -->



## 🎫 License

For academic use, this project is licensed under [the 2-clause BSD License](https://opensource.org/license/bsd-2-clause). 

## Acknowledgments

We would like to acknowledge that some codes and datasets are borrowed from [Scene-Diffuser](https://github.com/scenediffuser/Scene-Diffuser), [RealDex](https://github.com/4DVLab/RealDex), [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet), [UniDexGrasp](https://github.com/PKU-EPIC/UniDexGrasp), [GRAB](https://github.com/otaheri/GRAB), and [MultiDex Dataset](https://github.com/tengyu-liu/GenDexGrasp). We appreciate the authors for their great contributions to the community and for open-sourcing their code and datasets.

## 🖊️ Citation
```
@article{zhong2025dexgrasp,
  title={DexGrasp Anything: Towards Universal Robotic Dexterous Grasping with Physics Awareness},
  author={Zhong, Yiming and Jiang, Qi and Yu, Jingyi and Ma, Yuexin},
  journal={arXiv preprint arXiv:2503.08257},
  year={2025}
}
```