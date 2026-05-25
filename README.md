# ResNet Transfer Learning & Extensions

This repository contains experiments exploring transfer learning, data constraints, and model adaptation strategies using PyTorch. The project focuses on adapting pre-trained ResNet models to the Oxford-IIIT Pet Dataset across various operational constraints.

## Features Implemented

* **Linear Probing & Fine-Tuning:** Benchmarks comparing a frozen feature extractor (linear probing) against full configuration and gradual unfreezing. 
* **Limited Data Analysis:** Experiments simulating low-resource environments by restricting training data fractions and applying varied data augmentation.
* **Class Imbalance Handling:** Analyzes the impact of underrepresented classes and mitigates class bias using strategies like weighted cross-entropy.
* **Knowledge Distillation:** Trains a smaller ResNet "student" model guided by a fine-tuned larger ResNet "teacher".
* **LoRA (Low-Rank Adaptation):** Applies parameter-efficient fine-tuning to the classification model to compare against full parameter updates.
* **Pseudo-Labelling:** Semi-supervised learning strategies to incorporate unlabeled data into the training pipeline.

## Project Structure

* **`E-level/`**
  * `basicresults.py`: Core implementation for downloading data and executing initial binary/multi-class fine-tuning and linear probing.
  * `limited_data.ipynb`: Analyzes performance degradation and augmentation impact on restricted dataset sizes.
  * `clean_imbalanced_test.ipynb`: Notebook detailing the imbalanced classes experiments and evaluation metrics.
* **`extension-code/`**
  * `A_level_distillation.ipynb`: Implementation of the knowledge distillation pipeline.
  * `final_lora.ipynb`: LoRA configuration and fine-tuning experiments.
  * `pseudolabellingallcode.py`: Codebase for the semi-supervised pseudo-labelling routines.

## How to Run This

To run these experiments, you should download the code and import it into your cloud computing service of choice, such as google collab. 
