from googleapiclient.discovery import build
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from itertools import permutations
from collections import Counter
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

    # If there is no query result
    if not "items" in result:
        return []
    
    return result['items']


def collect_feedback(results):
    """
    Collects feedback on the relevance of search results from the user.

    Parameter:
    - results (list): A list of search results.

    Return:
    - relevant_results (list): A list of search results that the user marked as relevant.
    - irrelevant_results (list): A list of search results that the user marked as irrelevant.
    - precision (float): A float showing precision@10, calculated only on html files.
    """

    relevant_results = []
    irrelevant_results = []
    
    for index, result in enumerate(results, start=1):
        # If file format is not in HTML, skip the document
        if result.get('fileFormat'):
            print("\n--------------------------------------------")
            print(f"NON-HTML File format detected. Skipping Result {index}...")
            print("--------------------------------------------\n")
            continue
        
        # Display Results
        print(f"Result {index}\n[")
        print(f" URL: {result['link']}")
        print(f" Title: {result['title']}")
        print(f" Summary: {result['snippet']}\n]\n")
        
        # Ask for user feedback
        feedback = input(f"Relevant (Y/N)?").strip().lower()
        if feedback == 'y':
            relevant_results.append(result)
        else:
            irrelevant_results.append(result)
    
    # Calculate precision@10, focusing only on html files
    precision = len(relevant_results) / (len(relevant_results) + len(irrelevant_results))

    return relevant_results, irrelevant_results, precision


def refine_query_rocchio(original_query, relevant_docs, irrelevant_docs, alpha=1.0, beta=0.75, gamma=0.15):
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
    - augment_by (str): A string containing 2 new terms to augment the original query.
    - new_query (str): A string containing all terms (new, original) in a new order (best permutation). 
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
    original_terms = re.findall(r'\w+', original_query.lower())
    new_terms = [feature_names[idx] for idx in sorted_indices if feature_names[idx] not in original_terms]

    # Add top-scoring 2 new terms to the new query
    new_query_words = (original_query, new_terms[0], new_terms[1])

    # Find the best permutation of the query words by the highest frequency
    query_perms = list(permutations(new_query_words))
    relevant_texts_str = " ".join(relevant_texts).lower()
    perm_counts = Counter()
    for perm in query_perms:
        sections = relevant_texts_str.split(".")
        perm_counts[perm] = sum(len(re.findall(rf"{perm[0]}.*{perm[1]}.*{perm[2]}", section, re.IGNORECASE)) for section in sections)

    # Find the highest count 
    max_count = perm_counts.most_common(1)[0][1]

    # Filter permutations with the maximum count
    max_count_perms = [perm for perm, count in perm_counts.items() if count == max_count]  

    if len(max_count_perms) > 1:
        # If we have a tie, prioritize permutation with the original query at the beginning
        best_permutation = next((perm for perm in max_count_perms if perm[0] == original_query), max_count_perms[0])
    elif len(max_count_perms) == 1:
        # One best permutation found
        best_permutation = max_count_perms[0]
    else:
        # Just in case of an error where the list is might be empty
        best_permutation = new_query_words
    
    augment_by = " ".join(new_terms[:2])
    new_query = " ".join(best_permutation)

    return augment_by, new_query


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
    iteration = 0
    
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
        
        # Get query results from the engine
        results = google_search(query, api_key, engine_key, num=10)
        
        # Terminate when there are less than 10 query results in the first iteration
        if len(results) < 10 and iteration == 0:
            print("Less than 10 query results. Please try a different query")
            break
        elif len(results) == 0:  # Terminate if in any iteration there are no relevant results 
            print("No relevant results. Terminating ...")
            
        # Collect user feedback
        relevant_results, irrelevant_results, precision = collect_feedback(results)
        
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
        augment_by, new_query = refine_query_rocchio(query, relevant_results, irrelevant_results)
        print(f"Augmenting by {augment_by}")
        query = new_query
        
        iteration +=1

if __name__ == "__main__":
    main()
