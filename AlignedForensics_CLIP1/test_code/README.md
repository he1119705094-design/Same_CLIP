# Test Code

We use the testing pipeline provided by [CLIPDet](https://github.com/grip-unina/ClipBased-SyntheticImageDetection). We test on a wide variety of real and synthetic images. We make the dataset public on huggingface at this [link](https://huggingface.co/datasets/AniSundar18/Robust_LDM_Benchmark). 

## **Evaluation Procedure**  
To run inference with the models, you must specify their paths in a configuration file. An example configuration file is provided in the `weights` folder. Before running the evaluation, update the configuration file by setting `file_path` to the appropriate model weights.  

### **Preparing the Dataset**  

To evaluate the models, the image folder paths need to be converted into a CSV file. We provide a script to automate this process:  

```bash
python create_csv.py --base_folder "path_to_image_folder" --output_csv "output_csv_path" --dir "real/fake (optional)"
```

### **Running Inference**  

Once the dataset is prepared, you can compute per-image scores using the following command:  

```bash
python main.py --in_csv {images_csv_file} --out_csv {output_csv_file} --device "cuda:4" --weights_dir {weights_directory} --models {models_to_evaluate}
```

To run multiple evaluations in parallel, use the `run_tests.py` script. The resulting CSV files will contain the scores assigned by the neural networks (represented in the column names). These scores indicate the likelihood of an image being fake.  

The probability of an image being fake can be obtained by applying a sigmoid function to the score. We provide a script to evaluate accuracy and average precision for a set of real and fake images using their respective CSV files:  

```bash
python eval.py --real {path_to_real_csv} --fake {path_to_fake_csv} --ix {number_of_detectors_to_evaluate}
```

## Resolution/Compression sweeps
We also provide the code to recreate the resolution/webp-compression diagrams given in the paper. Examples can be found in the ```sweep.sh``` script. 


## Citation
If you find this code useful in your research, consider citing our work:
```
@misc{rajan2025aligneddatasetsimprovedetection,
      title={Aligned Datasets Improve Detection of Latent Diffusion-Generated Images}, 
      author={Anirudh Sundara Rajan and Utkarsh Ojha and Jedidiah Schloesser and Yong Jae Lee},
      year={2025},
      eprint={2410.11835},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.11835}, 
}
```
