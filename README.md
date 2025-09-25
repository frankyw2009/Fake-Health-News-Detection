**This project aims to detect fake health news through innovative fake health news detection models and datasets.**<br>
**This repository is corresponded to a paper manuscript submitted for JEI (Journal of Emerging Investigators).**

Here are the list of documents in this repository:

## FakeNewsDetection_General.py

This document contains all the essential codes for X-HND models with different types (GCN, SAGE, GAT).<br>
Notice: many lines of codes are in annotated form, as they serve as alternative modules for a certain part of the program.

## Folder: dataset_with_codes

This folder contains the finalized HNDataset with codes and raw data:
  ### 1. NewsData_BASE
  NewsData_BASE is the HNDataset-BASE referred in the paper. Data is stored in the form of graphs.<br>
  ### 2. NewsData_emo
  NewsData_emo is the HNDataset-Emotion reffered in the paper. Data is stored in the form of graphs.<br>
  ### 3. codes_to_make_dataset
    
  **a. base_graph.py**<br>
    base_graph.py is the document used to make HNDataset-BASE.
    
  **b. emotion_graph.py**<br>
    emotion_graph.py is the document used to make HNDataset-Emotion.<br>
    -->Both two documents above shall be used *together* with NewsData_BASE and NewsData_emo, respectively.
  
  ### 4. raw_data
  **a. final_dataset.npy**<br>
    final_dataset.npy stores raw news data in the form of a list, and all the elements in the list are dictionaries.<br>
  
  **b. output_finalized.csv**<br>
    output_finalized.csv shows a form with root news and their respective labels. Label 1 means fake news, label 0 means true news.<br>
 
  **c. graph_samples**<br>
    graph_samples is a folder that stores 10 raw data health news dictionaries individually in numpy documents.<br>
    This can be used to understand the structure of news dictioneries and can also be used for testing.<br>
     
