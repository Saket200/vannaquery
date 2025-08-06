
import requests
import json
import streamlit as st
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional, List
from langchain_core.messages import HumanMessage, AIMessage

# Streamlit page configuration
st.set_page_config(page_title="Vanna SQL Query Agent", layout="wide")

# Custom Vanna class combining ChromaDB and OpenAI
class MyVanna(ChromaDB_VectorStore, OpenAI_Chat):
    def __init__(self, config=None):
        ChromaDB_VectorStore.__init__(self, config=config)
        OpenAI_Chat.__init__(self, config=config)

# Initialize Vanna with API key and model
vn = MyVanna(config={'api_key': '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', 'model': 'gpt-4'})

# Initialize LangChain LLM
llm = ChatOpenAI(
    openai_api_key='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',
    model='gpt-4'
)

# Function to execute SQL via BuildMapper API
def run_sql_via_buildmapper(query: str):
    # Handle placeholders with default values
    params = {
        "company_id": 1,
        "user_id": 45,
        "country_name": "Canada",
        "limit": 10,
        "offset": 0
    }
    try:
        query = query % {k: f"'{v}'" if isinstance(v, str) else v for k, v in params.items()}
    except KeyError:
        pass  # Ignore missing placeholders
    try:
        response = requests.post(
            url="https://crm.buildmapper.ai/api/v1/execute_query",
            headers={
                "API-Key": "9c77dd4ec15c4c5b8ebd9a83efaeceae",
                "Content-Type": "application/json"
            },
            data=json.dumps({"query": query})
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error: {e.response.status_code} - {e.response.text}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Request error: {str(e)}"}

# Assign the API execution function to Vanna
vn.run_sql = run_sql_via_buildmapper
vn.allow_llm_to_see_data = True

# Function to get database schema with customer-friendly summary
def get_schema_info():
    schema_query = """
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'public'
    ORDER BY table_name, column_name;
    """
    result = vn.run_sql(schema_query)
    if "error" in result:
        return f"Failed to retrieve schema: {result['error']}", None
    if not result.get("result", {}).get("success", False):
        return "No schema information returned.", None

    schema_df = pd.DataFrame(result["result"]["data"])
    output = ["### Detailed Database Schema:"]
    output.append(schema_df.to_string(index=False))
    output.append("\n### Table Summaries for Customers:")
    for table_name in schema_df['table_name'].unique():
        summary = table_descriptions.get(table_name, "No description available.")
        columns = ", ".join(schema_df[schema_df['table_name'] == table_name]['column_name'])
        output.append(f"\n**Table**: {table_name}")
        output.append(f"**Purpose**: {summary}")
        output.append(f"**Key Fields**: {columns}")
    return "\n".join(output), schema_df

# Training with DDL and contextual documentation
try:
    with open("complete_ddl.sql", "r") as f:
        ddl_sql = f.read()
    vn.train(ddl=ddl_sql)
except FileNotFoundError:
    st.error("Error: 'complete_ddl.sql' not found. Please ensure the file exists.")

# Additional table descriptions
table_descriptions = {
    "lead_publish": "Contains information about published leads, including creation date, city (city_id), country (country_id), and permit details.",
    "res_city": "Stores city names and IDs, with name stored as JSONB (e.g., 'en_US' key for English names).",
    "res_country": "Stores country names and IDs, with name stored as JSONB.",
    "crm_lead": "Stores customer relationship management leads with stage and activity details.",
    "calendar_event": "Stores scheduled events with start and stop times, linked to users and opportunities."
}
for table, desc in table_descriptions.items():
    vn.train(documentation=f"Table {table}: {desc}")

# Predefined training examples
training_examples = [
    {
        "question": "Which city has the most new projects",
        "sql": """SELECT rc.name->>'en_US' AS city,
                  COUNT(*) AS project_count
                  FROM lead_publish lp
                  INNER JOIN res_city rc ON rc.id = lp.city_id
                  WHERE rc.name IS NOT NULL
                    AND lp.create_date >= CURRENT_DATE - INTERVAL '15 days'
                    AND lp.country_id IN (SELECT id FROM res_country WHERE name->>'en_US' ILIKE 'Canada')
                  GROUP BY rc.name->>'en_US'
                  ORDER BY project_count DESC
                  LIMIT 10;"""
    },
    {
        "question": "Show me the newest permits issued",
        "sql": """SELECT lp.id,
                  CASE WHEN lp.builder_company IS NOT NULL THEN lp.builder_company
                       WHEN lp.applicant_company IS NOT NULL THEN lp.applicant_company
                       WHEN lp.owner_company IS NOT NULL THEN lp.owner_company
                       ELSE lp.address END AS lead_name_chained,
                  TO_CHAR(lp.permit_issue_date, 'MM/DD/YYYY') AS permit_issue_date
                  FROM lead_publish lp
                  WHERE lp.permit_issue_date IS NOT NULL
                    AND lp.country_id IN (SELECT id FROM res_country WHERE name->>'en_US' ILIKE 'Canada')
                  ORDER BY lp.permit_issue_date DESC, lp.id DESC
                  LIMIT 10;"""
    },
    {
        "question": "Show me the newest permits issued",
        "sql": """SELECT
  lp.*,
  lp.id,
  CASE
    WHEN lp.builder_company IS NOT NULL THEN lp.builder_company
    WHEN lp.applicant_company IS NOT NULL THEN lp.applicant_company
    WHEN lp.owner_company IS NOT NULL THEN lp.owner_company
    ELSE lp.address
  END AS lead_name_chained,
  TO_CHAR(lp.permit_issue_date, 'MM/DD/YYYY') AS permit_issue_date,
  lpil.id AS imageId,
  lpil.image_url AS imageUrl,
  (cl.lead_id IS NOT NULL) AS already_saved_crm_user
FROM
  lead_publish lp
LEFT JOIN
  lead_publish_image_line lpil ON lp.id = lpil.lead_publish_id
LEFT JOIN
  crm_lead cl ON cl.lead_id = lp.id AND cl.company_id = %(company_id)s
WHERE
  lp.country_id IN (SELECT id FROM res_country WHERE name::TEXT ILIKE %(country_name)s)
  AND lp.permit_issue_date IS NOT NULL
ORDER BY
  lp.permit_issue_date DESC, lp.id DESC
LIMIT %(limit)s OFFSET %(offset)s;"""
    },
    {
        "question": "Which builder has highest construction value",
        "sql": """SELECT
  bb.id AS builder_id,
  bb.name AS builder_name,
  SUM(lp.cost_of_construction) AS total_value,
  COUNT(*) AS project_count
FROM
  lead_publish lp
JOIN
  builder_builder bb ON bb.id = lp.builder_id
WHERE
  lp.builder_id IS NOT NULL
  AND lp.country_id IN (SELECT id FROM res_country WHERE name::TEXT ILIKE %(country_name)s)
GROUP BY
  bb.id, bb.name
ORDER BY
  total_value DESC
LIMIT %(limit)s;"""
    },
    {
        "question": "Which city has the most new projects",
        "sql": """SELECT
  rc.name->>'en_US' AS city,
  COUNT(*) AS project_count
FROM
  lead_publish lp
INNER JOIN
  res_city rc ON rc.id = lp.city_id
WHERE
  rc.name IS NOT NULL
  AND lp.create_date >= CURRENT_DATE - INTERVAL '15 days'
  AND lp.country_id IN (SELECT id FROM res_country WHERE name::TEXT ILIKE %(country_name)s)
GROUP BY
  rc.name
ORDER BY
  project_count DESC
LIMIT %(limit)s;"""
    },
    {
        "question": "Summarize the current sales pipeline",
        "sql": """SELECT
  s.name->>'en_US' AS stage_name,
  COUNT(l.id) AS lead_count,
  SUM(COALESCE(l.expected_revenue, 0)) AS total_value,
  AVG(EXTRACT(DAY FROM (CURRENT_DATE - l.date_open))) AS avg_days_in_stage,
  (SELECT COUNT(*)
   FROM crm_lead
   WHERE company_id = %(company_id)s
     AND user_id = %(user_id)s
     AND type = 'opportunity'
     AND active = TRUE) AS total_leads
FROM
  crm_lead l
LEFT JOIN
  crm_stage s ON l.stage_id = s.id
WHERE
  l.user_id = 45
  AND l.company_id = %(company_id)s
  AND l.active = TRUE
  AND l.type = 'opportunity'
  AND l.date_open IS NOT NULL
GROUP BY
  s.name, s.sequence
ORDER BY
  s.sequence;"""
    },
    {
        "question": "What appointments do I have scheduled this week",
        "sql": """SELECT
  ce.*,
  cl.name AS lead_name,
  cl.contact_name
FROM
  calendar_event ce
LEFT JOIN
  crm_lead cl ON ce.opportunity_id = cl.id AND cl.company_id = %(company_id)s
WHERE
  ce.active = TRUE
  AND ce.user_id = %(user_id)s
  AND ce.company_id = %(company_id)s
  AND ce.start::date >= date_trunc('week', CURRENT_DATE)
  AND ce.start::date < date_trunc('week', CURRENT_DATE) + INTERVAL '7 days'
ORDER BY
  ce.start ASC;"""
    },
    {
        "question": "Show me any overlapping events I should reschedule",
        "sql": """SELECT
  e1.id AS event_id_1,
  e1.name AS event_title_1,
  e1.start AS start_1,
  e1.stop AS end_1,
  e2.id AS event_id_2,
  e2.name AS event_title_2,
  e2.start AS start_2,
  e2.stop AS end_2
FROM
  calendar_event e1
JOIN
  calendar_event e2 ON e1.user_id = e2.user_id
  AND e1.id < e2.id
  AND e1.active = TRUE
  AND e2.active = TRUE
  AND e1.start < e2.stop
  AND e1.stop > e2.start
WHERE
  e1.user_id = %(user_id)s
  AND e1.start::date >= date_trunc('week', CURRENT_DATE)
  AND e1.start::date < date_trunc('week', CURRENT_DATE) + INTERVAL '7 days'
  AND e2.start::date >= date_trunc('week', CURRENT_DATE)
  AND e2.start::date < date_trunc('week', CURRENT_DATE) + INTERVAL '7 days'
  AND e1.activity_type = 'calendar_event'
ORDER BY
  e1.start;"""
    }
]
for ex in training_examples:
    vn.train(question=ex["question"], sql=ex["sql"])

# LangGraph State Definition
class AgentState(TypedDict):
    question: str
    sql_query: Optional[str]
    results: Optional[List[dict]]
    answer: Optional[str]
    error: Optional[str]

# LangGraph Nodes
def generate_sql_node(state: AgentState) -> AgentState:
    try:
        sql_query = vn.generate_sql(state["question"])
        if not sql_query or not sql_query.strip().lower().startswith("select"):
            return {"error": "Invalid or non-SELECT query generated."}
        return {"sql_query": sql_query}
    except Exception as e:
        return {"error": f"SQL generation failed: {str(e)}"}

def execute_sql_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return state
    try:
        json_response = vn.run_sql(state["sql_query"])
        if "error" in json_response:
            return {"error": f"Query execution failed: {json_response['error']}"}
        if json_response.get("result", {}).get("success", False):
            data = json_response["result"]["data"]
            if not data:
                return {"error": "No results found. Possible reasons: empty table, no matching data, or incorrect query."}
            return {"results": data}
        return {"error": "No successful results returned from the query."}
    except Exception as e:
        return {"error": f"Query execution failed: {str(e)}"}

def synthesize_answer_node(state: AgentState) -> AgentState:
    if state.get("error"):
        return state
    try:
        prompt = PromptTemplate(
            input_variables=["question", "results"],
            template="""You are a helpful assistant. Based on the user's question and the database query results, provide a clear, concise, and natural-language answer. Summarize the key findings in a way that directly addresses the question. Avoid technical jargon unless necessary. Note: The results may be truncated to the first 5 rows for brevity.

Question: {question}
Query Results: {results}

Answer:"""
        )
        # Limit to first 5 rows to reduce token usage
        results_df = pd.DataFrame(state["results"])
        truncated_results = results_df.head(5).to_string(index=False)
        if len(results_df) > 5:
            truncated_results += "\n(Note: Results truncated to first 5 rows for brevity.)"
        response = llm.invoke(prompt.format(question=state["question"], results=truncated_results))
        return {"answer": response.content.strip()}
    except Exception as e:
        return {"error": f"Answer synthesis failed: {str(e)}"}

# Define LangGraph Workflow
workflow = StateGraph(AgentState)
workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("execute_sql", execute_sql_node)
workflow.add_node("synthesize_answer", synthesize_answer_node)
workflow.add_edge("generate_sql", "execute_sql")
workflow.add_edge("execute_sql", "synthesize_answer")
workflow.set_entry_point("generate_sql")
workflow.set_finish_point("synthesize_answer")
app = workflow.compile()

# Streamlit UI
st.title("Vanna SQL Query Agent")
st.write("Ask questions about your database or view the schema. Enter a question below or click 'Show Database Schema'.")

# Initialize session state for feedback
if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False
    st.session_state.correct_sql = ""
    st.session_state.feedback_answer = ""

# Schema button
if st.button("Show Database Schema"):
    schema_output, schema_df = get_schema_info()
    if isinstance(schema_output, str):
        st.markdown(schema_output)
    else:
        st.error("Failed to retrieve schema.")

# Query input
user_input = st.text_input("Enter your question:", key="question_input")
if st.button("Submit Query"):
    if user_input.strip().lower() == "exit":
        st.write("Exiting the application.")
        st.stop()
    elif user_input.strip().lower() == "get schema":
        schema_output, schema_df = get_schema_info()
        if isinstance(schema_output, str):
            st.markdown(schema_output)
        else:
            st.error("Failed to retrieve schema.")
    else:
        # Process query using LangGraph
        try:
            result = app.invoke({"question": user_input})
            if result.get("error"):
                st.error(f"{result['error']} Run 'Show Database Schema' to inspect the schema.")
            else:
                st.subheader("Generated SQL Query")
                st.code(result["sql_query"], language="sql")
                st.subheader("Query Results")
                st.dataframe(pd.DataFrame(result["results"]))
                st.subheader("Answer")
                st.write(result["answer"])
                # Feedback section
                st.subheader("Feedback")
                feedback = st.radio("Was this answer correct?", ("Yes", "No"), key="feedback_radio")
                if feedback == "No":
                    st.session_state.correct_sql = st.text_area("Please provide the correct SQL:", key="correct_sql_input")
                    if st.button("Submit Feedback"):
                        if st.session_state.correct_sql:
                            vn.train(question=user_input, sql=st.session_state.correct_sql)
                            st.session_state.feedback_submitted = True
                            st.session_state.feedback_answer = result["answer"]
                            st.success("Feedback submitted! Vanna has been updated.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)} Run 'Show Database Schema' to inspect the schema.")

# Display feedback confirmation
if st.session_state.feedback_submitted:
    st.write(f"Previous Answer: {st.session_state.feedback_answer}")
    st.write("Feedback has been submitted and Vanna updated with your corrected SQL.")
