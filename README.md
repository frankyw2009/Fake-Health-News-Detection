This project aims to detect fake health news through innovative fake health news detection models and datasets.
This repository is corresponded to a paper manuscript submitted for JEI (Journal of Emerging Investigators).

Here are the documents in this project:

Folder: dataset_with_codes
--
This folder contains the finalized HNDataset with codes and raw data:
  1. NewsData_BASE<br>
  NewsData_BASE is the HNDataset-BASE referred in the paper. Data is stored in the form of graphs.
  2. NewsData_emo<br>
  NewsData_emo is the HNDataset-Emotion reffered in the paper. Data is stored in the form of graphs.
  3. codes_to_make dataset<br>
  base_graph.py is the document used to make HNDataset-BASE.
  emotion_graph.py is the document used to make HNDataset-Emotion.
  Both two documents shall be used together with NewsData_BASE and NewsData_emo
  4. raw_data<br>
  final_dataset.npy stores raw news data in the form of a list, and all the elements in the list are dictionaries.
  first_graph.txt shows an example of news dictionary with root news, retweeters and historical posts in text form.
  output_finalized.csv shows a form with root news and labels. Label 1 means fake news, 0 means true news.

FakeNewsDetection_General.py
---------
This document contains all the essential codes for X-HND models with different types (GCN, SAGE, GAT). <br>
Notice: many code are in annotated form, as they serve as alternative modules for a certain part of program.
     
