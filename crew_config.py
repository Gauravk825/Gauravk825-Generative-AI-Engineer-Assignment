from crewai import Agent, Task
from crewai_tools import DirectoryReadTool, FileReadTool, SerperDevTool
from crewai_tools import BaseTool
from crewai import Agent, Task, Crew

def initialize_agents_and_tasks():
    # Initialize tools
    search_tool = SerperDevTool()

    # Initialize Agents
    industry_research_agent = Agent(
        role="Industry Researcher",
        goal="Research the industry and understand the company's segment and strategic focus",
        backstory="Use the web browser tool to gather information on the industry the company is part of, including key offerings and strategic focus areas.",
        allow_delegation=False,
        verbose=True
    )

    # Agent 2: Market Standards & Use Case Generation Agent
    use_case_generation_agent = Agent(
        role="Use Case Generator",
        goal="Analyze industry trends and propose relevant AI/ML use cases for operational and customer-focused improvements.",
        backstory="Analyze trends in AI, ML, and automation in the company's industry and propose use cases.",
        allow_delegation=False,
        verbose=True
    )
    
    # Agent 3: Resource Asset Collection Agent
    resource_collection_agent = Agent(
        role="Resource Collector",
        goal="Collect datasets related to generated use cases from Kaggle, HuggingFace, and GitHub.",
        backstory="Search for relevant datasets and tools for the generated use cases on Kaggle, HuggingFace, and GitHub.",
        allow_delegation=False,
        verbose=True
    )

    # Initialize Tasks
    industry_research_task  = Task(
       description="Use a web browser tool to research the company's industry, key offerings, and strategic focus.",
        expected_output="A detailed report of the company's industry, offerings, and strategic goals.",
        tools=[search_tool],
        agent=industry_research_agent,
    )

    use_case_generation_task = Task(
        description="Analyze industry trends in AI/ML and automation and propose relevant use cases for the company.",
        expected_output="A list of relevant AI/ML use cases to improve company processes and customer experiences.",
        tools=[search_tool],
        agent=use_case_generation_agent,
    )
    
    # Task 3: Collect Resource Assets
    resource_collection_task = Task(
        description="Search for relevant datasets for the proposed use cases on Kaggle, HuggingFace, and GitHub.",
        expected_output="A list of clickable links to relevant datasets and tools for AI/ML use cases.",
        agent=resource_collection_agent,
    )

    # Initialize Crew
    crew = Crew(
        agents=[industry_research_agent, use_case_generation_agent, resource_collection_agent],
        tasks=[industry_research_task, use_case_generation_task, resource_collection_task],
        verbose=2,
        memory=True
    )

    return crew
