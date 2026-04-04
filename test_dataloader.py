import sys
import traceback
sys.path.append('05_dashboard')
import streamlit as st
import utils.data_loader as dl

def my_info(msg): 
    with open('error_log.txt', 'a') as f: f.write("ST.INFO: " + str(msg) + "\n")
def my_warn(msg): 
    with open('error_log.txt', 'a') as f: f.write("ST.WARN: " + str(msg) + "\n")

# Mock STREAMLIT 
st.info = my_info
st.warning = my_warn
dl.st.info = my_info
dl.st.warning = my_warn

with open('error_log.txt', 'w') as f: f.write("Starting...\n")

try:
    df1 = dl._load_bronze_filtered(2)
    with open('error_log.txt', 'a') as f: f.write("Filtered size: " + str(len(df1)) + "\n")
except Exception as e:
    with open('error_log.txt', 'a') as f: f.write("_load_bronze_filtered threw: " + str(e) + "\n")

    
dl.load_bronze_servers()
