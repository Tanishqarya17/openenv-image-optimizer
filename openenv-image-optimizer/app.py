import streamlit as st
import subprocess
import os

st.set_page_config(page_title="OpenEnv Image Optimizer", page_icon="🚀")

st.title("🚀 OpenEnv: Image Optimizer Agent")
st.markdown("""
This interface wraps the OpenEnv Reinforcement Learning environment. 
Click the button below to trigger the `gpt-4o-mini` MLOps agent. It will navigate the environment and output the strict machine-readable logs required by the hackathon grading script.
""")

if st.button("Run Inference Baseline", type="primary"):
    with st.spinner("Agent is actively processing image states... Please wait."):
        # Copy the environment variables (like HF_TOKEN) so the script can use them
        current_env = os.environ.copy()
        
        try:
            # Execute your existing inference.py script exactly as a terminal would
            result = subprocess.run(
                ["python", "inference.py"],
                capture_output=True,
                text=True,
                env=current_env,
                check=True
            )
            
            st.success("Evaluation Complete!")
            st.markdown("### Standard Output (Grading Format):")
            # Display the exact STDOUT logs the judges need
            st.code(result.stdout, language="text")
            
        except subprocess.CalledProcessError as e:
            st.error("The agent encountered a critical error.")
            st.code(e.stderr, language="text")