# GRASP-O-MATIC
### Prompt-Driven Object Localization, Segmentation, and 6-DoF Grasp Generation
<!-- ![alt text](assets/image.png) <img width="208" height="148" alt="image" src="https://github.com/user-attachments/assets/843e5f3a-f47d-42a6-8bf2-1f3f3e5fd518" /> -->
<p align="center">
  <img src="assets/shoe.gif" width="100%" />
</p>

#### Authors: Saif Ahmad, Andnet DeBoer, Rishika Bera, Theo Coulson
--- 

Step 1 install docker 
```
https://docs.docker.com/desktop/setup/install/linux/ubuntu/
```

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
```
```
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```
sudo apt-get update
```

```
sudo apt-get install -y nvidia-container-toolkit
```

```
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

```
sudo usermod -aG docker $USER
```

```
newgrp docker
```

Allow docker to use your display

```
xhost +local:docker
```

```
docker compose build
```

```
newgrp docker
docker compose run graspomatic bash
```

```
# Get the top 10 best grasps
python3 grasp_pipeline.py --prompt "shoe" --top 10

# Get the top 50
python3 grasp_pipeline.py --prompt "shoe" --top-n 50
```

### Citations

```
@article{sundermeyer2021contact,
  title={Contact-GraspNet: Efficient 6-DoF Grasp Generation in Cluttered Scenes},
  author={Sundermeyer, Martin and Mousavian, Arsalan and Triebel, Rudolph and Fox, Dieter},
  booktitle={2021 IEEE International Conference on Robotics and Automation (ICRA)},
  year={2021}
}
```

```
@article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
```

```
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```