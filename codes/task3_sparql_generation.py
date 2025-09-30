import os
from google import genai
from dotenv import load_dotenv
from google.genai import types

def generate_orkg_sparql(target_question: str) -> str:
    """
    Generates a SPARQL query for ORKG using the Gemini API, incorporating 
    the official ORKG prefixes and a strong one-shot example.

    Args:
        target_question: The natural language question to be converted to SPARQL.

    Returns:
        The generated SPARQL query string.
    """
    # --- 1. Define Context Variables ---
    
    # OFFICIAL ORKG PREFIXES
    PREFIXES = """
PREFIX orkgp: <http://orkg.org/orkg/predicate/>
PREFIX orkgc: <http://orkg.org/orkg/class/>
PREFIX orkgr: <http://orkg.org/orkg/resource/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""
    
    # A. Ontology Example (One-shot learning) - Your provided example
    EXAMPLE_QUESTION = "What are the metrics of evaluation over the Stanford Cars dataset?"
    EXAMPLE_SPARQL = f"""
{PREFIXES.strip()}

SELECT DISTINCT ?metric ?metric_lbl
WHERE {{
  ?dataset       a                orkgc:Dataset;
                  rdfs:label       ?dataset_lbl.
  FILTER (str(?dataset_lbl) = "Stanford Cars")
  ?benchmark      orkgp:HAS_DATASET       ?dataset;
                  orkgp:HAS_EVALUATION    ?eval.
  OPTIONAL {{?eval           orkgp:HAS_METRIC         ?metric.
           ?metric          rdfs:label               ?metric_lbl.}}
}}
"""

    # B. Subgraph Context (Schema/Ontology details) - Used to provide general ORKG structure
    SUBGRAPH_CONTEXT = f"""
# ORKG Subgraph Context in Turtle Format
# orkgp for Predicates, orkgc for Classes, orkgr for Resources
{PREFIXES}

# General ORKG structural patterns for Paper, Contribution, Problem, Method
orkgr:R2005 a orkgc:Paper ;
    rdfs:label "AI in Climate Modeling" ;
    orkgp:P32 orkgr:R3006 . # P32 = has contribution

orkgr:R3006 a orkgc:Contribution ;
    orkgp:P3001 orkgr:R1004 ; # P3001 = has research problem
    orkgp:P4007 orkgr:M123 . # P4007 = has method

orkgr:R1004 a orkgc:ResearchProblem ;
    rdfs:label "Question Answering" .

orkgr:M123 a orkgc:Method ;
    rdfs:label "Sequence-to-Sequence Model" .

# Predicates used in the example:
# orkgp:HAS_DATASET, orkgp:HAS_EVALUATION, orkgp:HAS_METRIC
"""

    # --- 2. Construct the Full Prompt ---
    FULL_PROMPT = f"""
### 1. Role and Goal
You are an expert SPARQL query generator specialized in the **ORKG (Open Research Knowledge Graph)** ontology. Your task is to translate a natural language **Question** into a single, syntactically correct SPARQL query.

### 2. Context and Constraints
You are provided with the following information to ensure accuracy against the ORKG schema:

**A. Ontology Example (One-Shot):**
* **Example Question:** {EXAMPLE_QUESTION}
* **Example SPARQL Query:**
{EXAMPLE_SPARQL}

**B. Subgraph Context (Turtle Format):**
* **Subgraph:**
{SUBGRAPH_CONTEXT}

### 3. Input (The Target Request)
Based on the context above, generate a query for the following:

* **Target Question:** {target_question}

### 4. Output Format
Provide **only** the generated SPARQL query, including all necessary `PREFIX` declarations, enclosed in a single markdown code block. Do not include any explanatory text, introduction, or conversation outside of the code block.
"""

    # --- 3. Initialize Gemini Client and Generate Content ---
    try:
        # Client automatically picks up the API key from the environment variable GEMINI_API_KEY

        client = genai.Client()
        
        # Use a model that is good for code generation and instruction following
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=FULL_PROMPT,
            config=types.GenerateContentConfig(
                # Low temperature for deterministic, factual output (like code)
                temperature=0.1
            )
        )
        
        # Extract the text response
        return response.text.strip()

    except Exception as e:
        return f"An error occurred (check API key and internet connection): {e}"

# --- EXAMPLE USAGE ---
if __name__ == '__main__':
    # Ensure GEMINI_API_KEY is set in your environment
    load_dotenv()
    if not os.getenv('GEMINI_API_KEY'):
        print("ERROR: Please set the 'GEMINI_API_KEY' environment variable.")
    else:
        # Your new question for the ORKG
        USER_QUESTION = "Find all papers that use the method 'Sequence-to-Sequence Model' and list their titles."
        
        # Generate the query
        sparql_query_output = generate_orkg_sparql(USER_QUESTION)
        
        print("--- User Question ---")
        print(USER_QUESTION)
        print("\n--- Generated SPARQL Query ---")
        print(sparql_query_output)
        print("\n------------------------------")