import streamlit as st
from utils import get_openai_api_key, get_serper_api_key
from crew_config import initialize_agents_and_tasks
from crewai import Agent, Task, Crew



# Setup OpenAI API Key
openai_api_key = get_openai_api_key()
st.session_state["OPENAI_MODEL_NAME"] = 'gpt-3.5-turbo'
st.session_state["SERPER_API_KEY"] = get_serper_api_key()

# Initialize agents and tasks
crew = initialize_agents_and_tasks()

# Streamlit Interface
st.title("AI & GenAI Use Case Generator")

# User input
company_name = st.text_input("Enter the Company Name")
industry_type = st.selectbox("Select Industry", ["Automotive", "Manufacturing", "Finance", "Retail", "Healthcare"])
start_analysis = st.button("Generate Use Cases")

if start_analysis:
    # Run CrewAI multi-agent process
    result = crew.kickoff(inputs={"company": company_name, "industry": industry_type})
    
    # Display results
    st.subheader("Industry Research")
    st.write(result['research_agent'].result)

    st.subheader("Generated Use Cases")
    st.write(result['use_case_agent'].result)

    st.subheader("Resource Assets")
    st.write(result['resource_agent'].result)

    st.success("Analysis complete!")
