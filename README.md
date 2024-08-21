<div align="center">

# Efficient Motion Planning for Manipulators with Control Barrier Function-Induced Neural Controller

[![Conference](https://img.shields.io/badge/ICRA%20'24-Accepted-success)](https://ieeexplore.ieee.org/abstract/document/10610785)
</div>

Disclaimer: this code is research-grade, and should not be used in any production setting. It may contain outdated dependencies or security bugs, and of course we cannot guarantee the safety of our controllers on your specific autonomous system. If you have a particular application in mind, please reach out and we'll be happy to discuss with you.

## How to run
### Setup

Clone the repository and install dependencies
```bash
# clone project
git clone https://github.com/MIT-REALM/CBF-INC-RRT

# install project
cd cbf-inc-rrt
conda create --name cbf_inc_rrt python=3.9
conda activate cbf_inc_rrt
pip install -e .
pip install -r requirements.txt
```

### Train & Test

To train the neural CBF controller, we refer the user to the folder `neural_cbf/training`
If you want to evaluate the performance of the trained neural controller solely, you can resort to the folder `neural_cbf/evaluation`.

If you are interested in generating the results for the integrated motion planning and control, you can refer to the folder `motion_planning/evaluation`.

### Citation

If you find this code useful in your own research, please cite our paper:
```
@INPROCEEDINGS{10610785,
  author={Yu, Mingxin and Yu, Chenning and Naddaf-Sh, M-Mahdi and Upadhyay, Devesh and Gao, Sicun and Fan, Chuchu},
  booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)}, 
  title={Efficient Motion Planning for Manipulators with Control Barrier Function-Induced Neural Controller}, 
  year={2024},
  volume={},
  number={},
  pages={14348-14355},
  doi={10.1109/ICRA57147.2024.10610785}
}
```
