This project aims to detect fake health news through innovative fake health news detection models and datasets.<br>
This repository is corresponded to a paper manuscript submitted for JEI (Journal of Emerging Investigators).

Here are the documents in this repository:

Folder: dataset_with_codes
--
This folder contains the finalized HNDataset with codes and raw data:
  1. NewsData_BASE<br>
  NewsData_BASE is the HNDataset-BASE referred in the paper. Data is stored in the form of graphs.
  2. NewsData_emo<br>
  NewsData_emo is the HNDataset-Emotion reffered in the paper. Data is stored in the form of graphs.
  3. codes_to_make_dataset<br>
    a. base_graph.py<br>
    base_graph.py is the document used to make HNDataset-BASE.<br>
    b. emotion_graph.py<br>
    emotion_graph.py is the document used to make HNDataset-Emotion.<br>
    -->Both two documents shall be used together with NewsData_BASE and NewsData_emo, respectively.<br>
  4. raw_data<br>
    a. final_dataset.npy<br>
    final_dataset.npy stores raw news data in the form of a list, and all the elements in the list are dictionaries.<br>
    b. first_graph.txt<br>
    first_graph.txt shows an example of news dictionary with root news, retweeters and historical posts in text form.<br>
    c. output_finalized.csv<br>
    output_finalized.csv shows a form with root news and labels. Label 1 means fake news, label 0 means true news.<br>
    d. graph_samples<br>
    graph_samples is a folder that stores 10 raw data health news dictionaries in individual documents.
    This can be used to understand the structure of news dictioneries and can also be used for testing.<br>

FakeNewsDetection_General.py
---------
This document contains all the essential codes for X-HND models with different types (GCN, SAGE, GAT).<br>
Notice: many codes are in annotated form, as they serve as alternative modules for a certain part of the program.
     
