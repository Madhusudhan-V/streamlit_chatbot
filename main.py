import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

def check_ollama():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except:
        return False

def analyze_data_and_ask_ollama(question, df):
    try:
        viz_keywords = ['graph', 'chart', 'plot', 'visualize', 'draw', 'show chart', 'create graph']
        wants_visualization = any(keyword in question.lower() for keyword in viz_keywords)
        
        if wants_visualization:
            return create_visualization(question, df)
        
        basic_stats = f"""
DATASET OVERVIEW:
- Total rows: {len(df)}
- Columns: {', '.join(df.columns)}
COMPLETE DATA:
{df.to_string()}
"""
        
        prompt = f"""You are analyzing this CSV dataset:
{basic_stats}
User Question: {question}
Based on this EXACT data above, provide a precise answer with specific numbers, calculations, and insights. 
Reference the actual data values shown. If the question asks for trends, calculations, or comparisons, 
use the exact numbers from the dataset above.
Be specific and accurate - use the real data values, not estimates."""

        response = requests.post('http://localhost:11434/api/generate',
                               json={
                                   'model': 'llama3',
                                   'prompt': prompt,
                                   'stream': False
                               })
        
        if response.status_code == 200:
            return response.json().get('response', 'No response received')
        else:
            return f"Error: HTTP {response.status_code}"
            
    except Exception as e:
        return f"Error: {str(e)}"

def create_visualization(question, df):
    try:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        question_lower = question.lower()
        
        if 'bar' in question_lower or 'sales by' in question_lower or 'revenue by' in question_lower:
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                x_col = categorical_cols[0]
                y_col = numeric_cols[0] if 'sales' not in question_lower else (
                    'Sales' if 'Sales' in df.columns else numeric_cols[0]
                )
                if 'revenue' in question_lower and 'Revenue' in df.columns:
                    y_col = 'Revenue'
                
                fig = px.bar(df, x=x_col, y=y_col, title=f'{y_col} by {x_col}')
                st.plotly_chart(fig, use_container_width=True)
                return f"Created a bar chart showing {y_col} by {x_col}"
        
        elif 'line' in question_lower or 'trend' in question_lower or 'over time' in question_lower:
            if len(df.columns) >= 2:
                x_col = df.columns[0]
                y_col = numeric_cols[0] if numeric_cols else df.columns[1]
                
                fig = px.line(df, x=x_col, y=y_col, title=f'{y_col} Trends')
                st.plotly_chart(fig, use_container_width=True)
                return f"Created a line chart showing {y_col} trends over {x_col}"
        
        elif 'pie' in question_lower or 'distribution' in question_lower:
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                names_col = categorical_cols[0]
                values_col = numeric_cols[0]
                
                fig = px.pie(df, names=names_col, values=values_col, 
                           title=f'Distribution of {values_col} by {names_col}')
                st.plotly_chart(fig, use_container_width=True)
                return f"Created a pie chart showing distribution of {values_col} by {names_col}"
        
        elif 'scatter' in question_lower or 'correlation' in question_lower:
            if len(numeric_cols) >= 2:
                x_col = numeric_cols[0]
                y_col = numeric_cols[1]
                
                fig = px.scatter(df, x=x_col, y=y_col, title=f'{x_col} vs {y_col}')
                st.plotly_chart(fig, use_container_width=True)
                return f"Created a scatter plot showing {x_col} vs {y_col}"
        
        else:
            if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                x_col = categorical_cols[0]
                y_col = numeric_cols[0]
                
                fig = px.bar(df, x=x_col, y=y_col, title=f'{y_col} by {x_col}')
                st.plotly_chart(fig, use_container_width=True)
                return f"Created a chart showing {y_col} by {x_col}. You can ask for specific chart types like 'create a line chart' or 'show me a pie chart'."
            
        return "I can create charts for you! Try asking for specific types like 'create a bar chart', 'show me a line graph', or 'make a pie chart'."
        
    except Exception as e:
        return f"Error creating visualization: {str(e)}"

st.set_page_config(page_title="Simple Data Q&A", layout="wide")

st.title("Simple Data Q&A with Ollama")
st.markdown("*Ask any question about your data and get direct answers!*")

if not check_ollama():
    st.error("Ollama is not running! Please start it with: `ollama serve`")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.header("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns")
else:
    try:
        df = pd.read_csv('sample_data.csv')
        st.sidebar.info(f"Using sample data: {df.shape[0]} rows, {df.shape[1]} columns")
    except:
        st.error("No data available. Please upload a CSV file.")
        st.stop()

with st.sidebar.expander("Data Preview"):
    st.dataframe(df.head(100))

st.subheader("Ask Questions About Your Data")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("Ask anything about your data..."):
    st.session_state.messages.append({"role": "user", "content": question})
    
    with st.chat_message("user"):
        st.write(question)
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your data..."):
            answer = analyze_data_and_ask_ollama(question, df)
            st.write(answer)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})

if st.session_state.messages:
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()