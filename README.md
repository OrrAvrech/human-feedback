# Human-Feedback

## Setup
**(Recommended - conda + pip-tools):**
1. Install a new conda environment:
```commandline
$ conda env create -f venv/env.yml
```
2. Activate environment:
```commandline
$ conda activate human-feedback
```
3. Sync already compiled requirements:
```commandline
$ pip-sync venv/requirements.txt
```
Working environment is now ready to use. The recommended way to add new packages, is to edit `venv/requirements.in` and run:
```commandline
$ pip-compile venv/requirements.in
```
This line will generate an updated version of the project's `requirements.txt` file, which can be easily synced to the virtual environment with `pip-sync`.

## 4DHumans to HumanML3D Conversion
- In `perturbate/random_edit.py` you can load the `.pkl` file from 4DHumans (tracking HMR predictions over a given video). This script then computes the joints and vertices in SMPL representation. You can then save the first 22 joints, as used in HumanML3D to a `.npy` file.

- The script `data/humanml/motion_process.py`, which is taken from the [HumanML3D](https://github.com/EricGuo5513/HumanML3D/tree/main) repo and modified a bit, takes a `.npy` file of joints in SMPL representation and saves them in HumanML3D representation in `new_joints` and `new_joint_vecs` folders. The representation in `new_joint_vecs` is a vector with length 263 which also contains velocities and rotations in 6D continouts rotation representation (refer to `data/humanml/README.md` for more information about this representation taken from the HumanML3D paper).

## Text-to-Motion Editing
- Refer to the forked version of [MDM](https://github.com/OrrAvrech/motion-diffusion-model) with my modifications that introduce a new custom dataset called HumanFeedback.

- For setup, follow their README instructions. You will only need text-to-motion dependencies, data and model. In the data part, only section (a) (i.e, the texts) is required for generation.

- For editing using HumanML3D or KIT-ML datasets, you will need the motions because they randomly sample from them. So to play around with their editing script, I suggest getting the KIT-ML dataset which is straightforward. HumanML3D is not fully open and may take some time to reproduce, but they do provide a single sample of their dataset in their [repo](https://github.com/EricGuo5513/HumanML3D/tree/main/HumanML3D).

- For running text-to-motion in edit mode on our data, use `sample/edit_ood.py`. For example:
```
python sample.edit_ood --model_path "./save/humanml_trans_enc_512/model000200000.pt" --edit_mode" "upper_body" --text_condition "a person lifts right arm high" --num_repetitions 1 --dataset "humanfeedback" --batch_size 1 --num_samples 1
```

Edit mode can be either "upper_body" which masks upper body joints or "in_between" which masks the middle of the input motion.