#CABG Surgery - Preoperative, Intraoperative, and Postoperative Care Analysis Surgery 
Overview
This project focuses on the Preoperative, Intraoperative, and Postoperative Care in Coronary Artery Bypass Grafting (CABG) Surgery. The goal of the project is to provide an analysis using AI techniques to help improve patient care and assist in clinical decision-making. By leveraging medical records, surgical data, and post-surgical outcomes, the project aims to create a robust model for understanding and predicting the patient's risk during these phases of surgery.

This repository includes a system that allows querying relevant medical documents (such as preoperative assessments, intraoperative details, and postoperative recovery information) using a retrieval-augmented generation (RAG) model and FAISS for document similarity.

Key Features
Preoperative Care: Analysis of the patient's condition before CABG surgery. This includes assessing the risk factors, diagnostic tests, and preparations for surgery.

Intraoperative Care: A comprehensive model that assesses the procedure during surgery, including factors such as anesthesia, surgical techniques, and patient monitoring.

Postoperative Care: Monitoring and recovery phase after surgery, including complications, medication, and rehabilitation.

Technologies Used
Python: Programming language used for developing the entire system.

FAISS: A library used for efficient similarity search and clustering of embeddings.

PyMuPDF: A library for extracting text from PDF documents.

Sentence-Transformers: Pre-trained transformer models for generating embeddings from text.

AI and Machine Learning: For analyzing medical data and building prediction models.

