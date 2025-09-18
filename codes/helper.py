# helper functions for GAR_SKGQA
import os
import json
# from dotenv import load_dotenv
from SPARQLWrapper import SPARQLWrapper, JSON , CSV

def run_sparql_query(sparql_text: str, SPARQLPATH: str) -> json:
    """"
    Run SPARQL query and return results in JSON format
    :param sparql_text: SPARQL query text
    :param SPARQLPATH: SPARQL endpoint URL
    :return: results in JSON format
    """
    try:
        sparql = SPARQLWrapper(SPARQLPATH)
        sparql_txt = sparql_text
        sparql.setQuery(sparql_txt)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        # Remove keys to match the expected output format in the benchmark
        results["head"].pop("link", None)
        results["results"].pop("distinct", None)
        results["results"].pop("ordered", None)
        return json.dumps(results)
    except:
        print('Your database is not installed properly !!!')


def orkg_prefixes() -> str:
    """
    Return common prefixes for ORKG SPARQL queries
    :return: prefixes string
    """
    prefixes = """
    PREFIX orkgp: <http://orkg.org/orkg/predicate/>
    PREFIX orkgc: <http://orkg.org/orkg/class/>
    PREFIX orkgr: <http://orkg.org/orkg/resource/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    """
    return prefixes

def dblp_prefixes() -> str:
    """
    Return common prefixes for DBLP SPARQL queries
    :return: prefixes string
    """
    prefixes = """
    PREFIX dblp: <https://dblp.org/rdf/schema#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX cito: <http://purl.org/spar/cito/>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX schema: <https://schema.org/>
    """
    return prefixes