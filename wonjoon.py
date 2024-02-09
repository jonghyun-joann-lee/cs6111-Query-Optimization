from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from googleapiclient.discovery import build
import re
import sys


# API setup
def google_search(search_term, api_key, engine_key, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=engine_key, **kwargs).execute()
    return res['items']


def collect_feedback(results):
    relevant_results = []
    for index, result in enumerate(results, start=1):
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


def refine_query_with_rocchio(original_query, relevant_docs, irrelevant_docs, alpha=1, beta=0.75, gamma=0.15):
    # Process Texts
    relevant_texts = [doc['title'] + " " + doc['snippet'] for doc in relevant_docs]
    irrelevant_texts = [doc['title'] + " " + doc['snippet'] for doc in irrelevant_docs]
    all_texts = [original_query] + relevant_texts + irrelevant_texts
    
    # Define vectorizer and transform texts
    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(all_texts)
    tfidf_matrix = vectorizer.transform(all_texts)

    # Calculate the mean of the vectors
    original_query_vector = tfidf_matrix[0]
    relevant_vectors_mean = tfidf_matrix[1:1+len(relevant_docs)].mean(axis=0)
    irrelevant_vectors_mean = tfidf_matrix[1+len(relevant_docs):].mean(axis=0)
    
    # Adjust the original query vector
    new_query_vector = alpha * original_query_vector + beta * relevant_vectors_mean - gamma * irrelevant_vectors_mean
    
    # Get the terms that are not in the original query 
    feature_names = np.array(vectorizer.get_feature_names_out())
    sorted_indices = np.argsort(new_query_vector.A).flatten()[::-1]
    new_terms = [feature_names[idx] for idx in sorted_indices if feature_names[idx] not in set(re.findall(r'\w+', original_query.lower()))]

    # Return the top 2 new terms
    return ' '.join(new_terms[:2])


def main():
    # Get parameters
    api_key = sys.argv[1]
    engine_key = sys.argv[2]
    target_precision = float(sys.argv[3])
    query = sys.argv[4]
    
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

        if precision >= target_precision:
            print("Desired precision reached, done")
            break
        elif precision == 0:
            print("Below desired precision, but can no longer augment the query")
            break
        else: 
            print(f"Still below the desired precision of {target_precision}")
        
        irrelevant_results = [result for result in results if result not in relevant_results]
        new_keywords = refine_query_with_rocchio(query, relevant_results, irrelevant_results)
        print(f"Augmenting by {new_keywords}")

        # Form the new query by combining original query and new keywords
        query = f"{query} {new_keywords}"
        

if __name__ == "__main__":
    main()
