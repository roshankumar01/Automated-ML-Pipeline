# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:00:08 2023

@author: Roshan
"""
import streamlit as st
import pandas as pd
import os

import pandas_profiling
from streamlit_pandas_profiling import st_profile_report

from pycaret.regression import setup , compare_models, pull,save_model 
with st.sidebar:
    st.title("Auto ML")
    choice = st.radio("Sections", ["Upload","Profiling","ML","Download"])
    st.info("This App allows you to build an automated ML pipeline")
    
if os.path.exists("source.csv"):
    df = pd.read_csv("source.csv",index_col=None)

if choice == "Upload":
    st.title("Upload your data for Modeling !")
    file = st.file_uploader("upload your dataset")

    if file :
        df = pd.read_csv(file ,index_col=None)
        df.to_csv("source.csv",index=None)
        st.dataframe(df)
    
    
if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    profile_report = df.profile_report()
    st_profile_report(profile_report)
    
    
if choice == "ML":
    st.title("Machine learning")
    target = st.selectbox("select your target", df.columns)
    if st.button("Train the models"):
        setup(df,target=target)
        setup_df=pull()
        st.info("ML")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df = pull()
        st.info("best model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, "best_model")
    
    
if choice == "Download":
    with open("best_model.pkl","rb") as f :
        st.download_button("download the model",f , "trained_model.pkl")
    pass