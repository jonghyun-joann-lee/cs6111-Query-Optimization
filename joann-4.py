from googleapiclient.discovery import build
from collections import Counter
import re
import sys
import math


# API setup
def google_search(search_term, api_key, engine_key, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=engine_key, **kwargs).execute()
    return res['items']


def collect_feedback(results):
    relevant_results = []
    non_relevant_results = []
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
        if feedback == 'n':
            non_relevant_results.append(result)

    return relevant_results, non_relevant_results


def load_stop_words(file_path):
    # Read a file containing stop words and return them as a set
    stop_words = set()
    with open(file_path, 'r') as file:
        for line in file:
            stop_word = line.strip()
            stop_words.add(stop_word)
    return stop_words


def refine_query(original_query, relevant_docs, non_relevant_docs, stop_words):
    original_keywords = set(re.findall(r'\w+', original_query.lower()))

    # Initialize structures for TF-IDF calculations
    relevant_tf = []  # List of term frequencies for each relevant document
    non_relevant_tf = []  # Same for non-relevant documents
    doc_freq = Counter()  # Document frequency for all terms across both relevant and non-relevant documents

    # Process relevant documents
    for docs in [relevant_docs, non_relevant_docs]:
        for doc in docs:
            content = f"{doc['title']} {doc['snippet']}"
            terms = re.findall(r'\w+', content.lower())
            filtered_terms = [term for term in terms if term not in stop_words and term not in original_keywords]

            # Calculate term frequency for the current document and update document frequency
            tf = Counter(filtered_terms)
            if docs == relevant_docs:
                relevant_tf.append(tf)
            else:
                non_relevant_tf.append(tf)

            for term in set(filtered_terms):
                doc_freq[term] += 1

    # FOR DEBUGGING
    print(relevant_tf)
    print(non_relevant_tf)

    # Total number of documents (for IDF calculation)
    total_docs = len(relevant_docs) + len(non_relevant_docs)

    # Rocchio adjustment factors
    alpha, beta, gamma = 1, 0.75, 0.15

    # Calculate TF-IDF scores and adjust according to Rocchio
    term_scores = Counter()
    # Handle relevant documents
    for tf in relevant_tf:
        for term, freq in tf.items():
            idf = math.log((total_docs + 1) / (doc_freq[term] + 1))
            tf_idf_score = (1 + math.log(freq)) * idf
            term_scores[term] += beta * tf_idf_score / len(relevant_docs)

    # Handle non-relevant documents
    for tf in non_relevant_tf:
        for term, freq in tf.items():
            idf = math.log((total_docs + 1) / (doc_freq[term] + 1))
            tf_idf_score = (1 + math.log(freq)) * idf
            term_scores[term] -= gamma * tf_idf_score / len(non_relevant_docs) if len(non_relevant_docs) > 0 else 0

    # Select top 2 terms for query expansion based on adjusted scores
    new_keywords = [word for word, _ in term_scores.most_common(2) if word not in original_keywords]

    # FOR DEBUGGING
    print(term_scores)

    return new_keywords


def main():
    # Load stop words
    stop_words_file = 'proj1-stop.txt'
    stop_words = load_stop_words(stop_words_file)

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
        relevant_results, non_relevant_results = collect_feedback(results)
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
        
        new_keywords = refine_query(query, relevant_results, non_relevant_results, stop_words)
        print(f"Augmenting by  {' '.join(new_keywords)}")

        # Form the new query by combining original query and new keywords
        query = f"{query} {' '.join(new_keywords)}"
        

if __name__ == "__main__":
    main()
