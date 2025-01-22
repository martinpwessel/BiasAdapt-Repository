# LLM-based Adversarial Dataset Augmentation for Automatic Media Bias Detection
This is the repository for the paper "LLM-based Adversarial Dataset Augmentation for Automatic Media Bias Detection".

Abstract: "This study presents BiasAdapt, a novel data augmentation strategy designed to enhance the robustness of automatic media bias detection models. Leveraging the BABE dataset, BiasAdapt uses a generative language model to identify bias-indicative keywords and replace them with alternatives from opposing categories, thus creating adversarial examples that preserve the original bias labels. The contributions of this work are twofold: it proposes a scalable method for augmenting bias datasets with adversarial examples while preserving labels, and it publicly releases an augmented adversarial media bias dataset. Training on BiasAdapt reduces the reliance on spurious cues in four of the six evaluated media bias categories but is accompanied by changes in performance."

It contains the following:    
-- Data    
---- BABE Dataset.xlsx _The original BABE dataset by Spinde et al. (2021)_   
---- BiasAdapt Dataset.xlsx _The final BiasAdapt dataset_   
---- Spurious Cues Test Set.pkl _The test set as introduced by Wessel and Horych (2024)_   

-- BiasAdapt Creation.py _The script to generate BiasAdapt_   
-- Model Training and Comparison.py _The script to train the model and analysis_   
