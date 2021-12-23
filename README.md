# Enhancing Robustness in Embodied Navigation

## Original code and paper:

Xiaoming Zhao, Harsh Agrawal, Dhruv Batra, and Alexander Schwing. The Surprising Effectiveness of Visual Odometry Techniques for Embodied PointGoal Navigation. ICCV 2021.

Link: https://github.com/Xiaoming-Zhao/PointNav-VO

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<p align="center">The Surprising Effectiveness of Visual Odometry Techniques for Embodied PointGoal Navigation</p>

<p align="center"><b><a href="https://xiaoming-zhao.github.io/projects/pointnav-vo/">Project Page</a> | <a href="https://arxiv.org/abs/2108.11550">Paper</a></b></p>

<p align="center">
  <img width="100%" src="media/nav.gif"/>
</p>

## Setup

### Install Dependencies

```bash
conda env create -f environment.yml
```

### Install Habitat

The repo is tested under the following commits of [habitat-lab](https://github.com/facebookresearch/habitat-lab) and [habitat-sim](https://github.com/facebookresearch/habitat-sim).

```bash
habitat-lab == d0db1b55be57abbacc5563dca2ca14654c545552
habitat-sim == 020041d75eaf3c70378a9ed0774b5c67b9d3ce99
```

Note, to align with Habitat Challenge 2020 settings (see Step 36 in [the Dockerfile](https://hub.docker.com/layers/fairembodied/habitat-challenge/testing_2020_habitat_base_docker/images/sha256-761ca2230667add6ab241a0eaff16984dc271486ec659984ae13ccab57a9c52b?context=explore)), when installing `habitat-sim`, we compiled without CUDA support as

```bash
python setup.py install --headless
```

There was a discrepancy between noises models in CPU and CPU versions which has now been fixed, see [this issue](https://github.com/facebookresearch/habitat-sim/pull/987). Therefore, to reproduce the results in the paper with our pre-trained weights, you need to use noises model of CPU-version.

### Download Data

We need two datasets to enable running of this repo:

1. [Gibson scene dataset](https://github.com/StanfordVL/GibsonEnv/blob/f474d9e/README.md#database)
2. [PointGoal Navigation splits](https://github.com/facebookresearch/habitat-lab/blob/d0db1b5/README.md#task-datasets), we need `pointnav_gibson_v2.zip`.

Please follow [Habitat's instruction](https://github.com/facebookresearch/habitat-lab/blob/d0db1b5/README.md#task-datasets) to download them. We assume all data is put under `./dataset` with structure:

```
.
+-- dataset
|  +-- Gibson
|  |  +-- gibson
|  |  |  +-- Adrian.glb
|  |  |  +-- Adrian.navmesh
|  |  |  ...
|  +-- habitat_datasets
|  |  +-- pointnav
|  |  |  +-- gibson
|  |  |  |  +-- v2
|  |  |  |  |  +-- train
|  |  |  |  |  +-- val
|  |  |  |  |  +-- valmini
```

## Reproduce

Download pretrained checkpoints of RL navigation policy and VO from [this link](https://drive.google.com/drive/folders/1HG_d-PydxBBiDSnqG_GXAuG78Iq3uGdr?usp=sharing). Put them under `pretrained_ckpts` with the following structure:

```
.
+-- pretrained_ckpts
|  +-- rl
|  |  +-- no_tune
|  |  |  +-- rl_no_tune.pth
|  |  +-- tune_vo
|  |  |  +-- rl_tune_vo.pth
|  +-- vo
|  |  +-- act_forward.pth
|  |  +-- act_left_right_inv_joint.pth
```

### RL evaluation procedure

Parameters for the evaluation procedures can be found in `configs/point_nav_habitat_challenge_2020.yaml` and `configs/rl/ddppo_pointnav.yaml`.
Then you can run the following command to evaluate the policy:

```bash
cd /path/to/this/repo
export POINTNAV_VO_ROOT=$PWD

export NUMBA_NUM_THREADS=1 && \
export NUMBA_THREADING_LAYER=workqueue && \
conda activate pointnav-vo && \
python ${POINTNAV_VO_ROOT}/launch.py \
--repo-path ${POINTNAV_VO_ROOT} \
--n_gpus 1 \
--task-type rl \
--noise 1 \
--run-type eval \
--addr 127.0.1.1 \
--port 8338
```

### Dataset Generation for VO training

We need to generate dataset for training visual odometry model. Please make sure your disk space is enough for the generated data. With 1 million data entries, it takes about **460 GB**.

```bash
cd ${POINTNAV_VO_ROOT}

export PYTHONPATH=${POINTNAV_VO_ROOT}:$PYTHONPATH && \
python3 ${POINTNAV_VO_ROOT}/pointnav_vo/vo/dataset/generate_datasets.py \
--config_f ${POINTNAV_VO_ROOT}/configs/point_nav_habitat_challenge_2020.yaml \
--train_scene_dir ./dataset/habitat_datasets/pointnav/gibson/v2/train/content  \
--val_scene_dir ./dataset/habitat_datasets/pointnav/gibson/v2/val/content \
--save_dir ./dataset/vo_dataset \
--data_version v2 \
--vis_size_w 341 \
--vis_size_h 192 \
--obs_transform none \
--act_type -1 \
--rnd_p 1.0 \
--N_list 1000 \
--name_list train \
--corr_seq Spatter DefocusBlur \
--sev_seq 3 3
```

### VO module training procedure

`configs/vo/vo_pointnav.yaml` contains the parameters for training the VO module.

```bash
cd /path/to/this/repo
export POINTNAV_VO_ROOT=$PWD
```

The original authors of the code find the following training strategy efficient, you need to modify `./configs/vo/vo_pointnav.yaml`:

- for action `move_forward`, set:
  - `VO.TRAIN.action_type = 1`
  - `VO.GEOMETRY.invariance_types = []`
- for action `turn_left` and `turn_right`:
  - 1st stage: train VO models separately for these two actions:
    - for action `move_left`: set
      - `VO.TRAIN.action_type = 2`
      - `VO.GEOMETRY.invariance_types = ["inverse_data_augment_only"]`.
    - for action `move_right`: set
      - `VO.TRAIN.action_type = 3`
      - `VO.GEOMETRY.invariance_types = ["inverse_data_augment_only"]`.
  - 2nd stage: jointly train VO models for `turn_left` and `turn_right` with geometric invariance loss, set:
    - `VO.TRAIN.action_type = [2, 3]`
    - `VO.GEOMETRY.invariance_types = ["inverse_joint_train"]`
    - `VO.MODEL.pretrained = True`
    - `VO.MODEL.pretrained_ckpt` to saved checkpoints in previous steps.

```bash
cd ${POINTNAV_VO_ROOT}

ulimit -n 65000 && \
conda activate pointnav-vo && \
python ${POINTNAV_VO_ROOT}/launch.py \
--repo-path ${POINTNAV_VO_ROOT} \
--n_gpus 1 \
--task-type vo \
--noise 1 \
--run-type train \
--addr 127.0.1.1 \
--port 8338
```
