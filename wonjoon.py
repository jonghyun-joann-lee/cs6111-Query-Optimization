from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from googleapiclient.discovery import build
import re
import sys


def google_search(search_term, api_key, engine_key, **kwargs):
    """
    Performs a Google search using the specified search term, API key, and search engine key.
    Accepts additional keyword arguments for the search parameters.

    Parameters:
    - search_term (str): The search query.
    - api_key (str): API key for Google Custom Search API.
    - engine_key (str): Search engine ID to specify the search engine.

    Return:
    - (list): A list of search result items.
    """
    service = build("customsearch", "v1", developerKey=api_key)
    result = service.cse().list(q=search_term, cx=engine_key, **kwargs).execute()
    return result['items']


def collect_feedback(results):
    """
    Collects feedback on the relevance of search results from the user.

    Parameter:
    - results (list): A list of search results.

    Return:
    - (list): A list of search results that the user marked as relevant.
    """
    relevant_results = []
    for index, result in enumerate(results, start=1):
        print(result)
        # Display Results
        print(f"Result {index}\n[")
        print(f" URL: {result['link']}")
        print(f" Title: {result['title']}")
        print(f" Summary: {result['snippet']}\n]\n")
        
        # Ask for user feedback
        feedback = input(f"Relevant (Y/N)?").strip().lower()
        if feedback == 'y':
            relevant_results.append(result)
    return relevant_results

 
def refine_query_rocchio(original_query, relevant_docs, irrelevant_docs, alpha=1, beta=0.75, gamma=0.15):
    """
    Refines the search query using the Rocchio's algorithm by adjusting the original query vector based on the vectors
    of relevant and irrelevant documents.

    Parameters:
    - original_query (str): Original search query.
    - relevant_docs (list): List of documents marked as relevant.
    - irrelevant_docs (list): List of documents marked as irrelevant.
    - alpha (float): The weight of the original query for the revised query.
    - beta (float): The weight of the relevant documents for the revised query.
    - gamma (float): The weight of the irrelevant documents for the revised query.

    Return:
    - (str): A string containing 2 new terms to augment the original query.
    """
    
    # Process Texts
    relevant_texts = [doc['title'] + " " + doc['snippet'] for doc in relevant_docs]
    irrelevant_texts = [doc['title'] + " " + doc['snippet'] for doc in irrelevant_docs]
    all_texts = [original_query] + relevant_texts + irrelevant_texts

    # Perform TF-IDF transformation using TfidfVectorizer from scikit-learn
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(all_texts)
    tfidf_matrix = vectorizer.transform(all_texts)

    # Calculate the mean of the vectors
    original_query_vector = tfidf_matrix[0]
    relevant_vectors_mean = tfidf_matrix[1:1+len(relevant_docs)].mean(axis=0)
    irrelevant_vectors_mean = tfidf_matrix[1+len(relevant_docs):].mean(axis=0)
    
    # Adjust the original query vector
    new_query_vector = alpha * original_query_vector + beta * relevant_vectors_mean - gamma * irrelevant_vectors_mean
    
    # Array of all terms (words) learned by the vectorizer during fitting
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Sort in descending order based on the term importance (TF-IDF value), and get the argument position
    sorted_indices = np.argsort(new_query_vector.A).flatten()[::-1]
    
    # Get the terms that are not in the original query 
    new_terms = [feature_names[idx] for idx in sorted_indices if feature_names[idx] not in set(re.findall(r'\w+', original_query.lower()))]

    # Return the top 2 new terms
    return ' '.join(new_terms[:2])


def main():
    """
    Main function to execute the iterative search refinement process. 
    
    Accepts command-line arguments for API key, search engine key, target precision, and initial query.

    It initializes the search with user query and target precision, 
    performs the search, collects feedback, and refines the query until the desired
    precision is reached or no further refinement is feasible.
    """

    # Get parameters
    api_key = sys.argv[1]
    engine_key = sys.argv[2]
    target_precision = float(sys.argv[3])
    query = sys.argv[4]
    
    # Main loop
    while True:
        # Display parameters
        print("Parameters:")
        print("Client key  =", api_key)
        print("Engine key  =", engine_key)
        print("Query       =", query)
        print("Precision   =", target_precision)
        print("Google Search Results:") 
        print("======================")
        
        # Display query & Collect feedback
        results = google_search(query, api_key, engine_key, num=10)
        relevant_results = collect_feedback(results)
        precision = len(relevant_results) / len(results)
        
        # Display feedback
        print("======================")
        print("FEEDBACK SUMMARY")
        print(f"Query {query}")
        print(f"Precision {precision}")

        # Evaluate precision
        if precision >= target_precision:
            print("Desired precision reached, done")
            break
        elif precision == 0:
            print("Below desired precision, but can no longer augment the query")
            break
        else: 
            print(f"Still below the desired precision of {target_precision}")
        
        # Perform query augmentation
        irrelevant_results = [result for result in results if result not in relevant_results]
        new_terms = refine_query_rocchio(query, relevant_results, irrelevant_results)
        print(f"Augmenting by {new_terms}")

        # Form the new query by combining original query and new terms
        query = f"{query} {new_terms}"
        

if __name__ == "__main__":
    main()
