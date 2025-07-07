$tree
.
├── 2d_plot_diffusion_todo/ # Task 1: 2D toy diffusion code & notebook
│ ├── network.py
│ ├── ddpm.py
│ └── ddpm_tutorial.ipynb
│
├── image_diffusion_todo/ # Task 2: AFHQ image DDPM
│ ├── scheduler.py
│ ├── model.py
│ ├── train.py
│ ├── sampling.py
│ └── dataset.py
│
├── figures/ # Final report figures
│ ├── heart_target_prior.png
│ ├── heart_noising.png
│ ├── heart_loss_curve.png
│ ├── heart_chamfer_training.png
│ ├── heart_ddpm_samples.png
│ ├── afhq_loss_curve.png
│ ├── 8_samples.png
│ └── fid_score.png
│
├── requirements.txt # Python dependencies
└── README.md # You are here


---

## 1. 2D Diffusion Experiments

![Target vs Prior](2d_plot_diffusion_todo/figures/heart_target_prior.png)

A simple DDPM (an MLP of TimeLinear layers) learns to recover the heart‐shaped target from Gaussian noise.  See `2d_plot_diffusion_todo/ddpm_tutorial.ipynb` for details, including forward noise trajectories and Chamfer‐distance plots.

---

## 2. AFHQ Image Diffusion

Trained on **64×64** AFHQ (cats, dogs, wildlife) for **100 000** steps on an **NVIDIA RTX 3060** (≈28 h).


**Random Samples (8 first)**  
![AFHQ Samples](image_diffusion_todo/results/8_final_samples/8_samples.png)

**FID** = 9.64  
![FID Score](image_diffusion_todo/results/8_final_samples/fid_score.png)

---

## Reproduce

1. **Create environment**  
   ```bash
   conda create -n genai310 python=3.10
   conda activate genai310
   pip install -r requirements.txt
