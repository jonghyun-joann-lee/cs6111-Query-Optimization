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

    # Set up for tf-idf calculations
    relevant_tf = []  # List of term frequencies for each relevant doc
    non_relevant_tf = []  # Same for non-relevant docs
    doc_freq = Counter()  # Document frequency for all terms across all docs
    total_docs = len(relevant_docs) + len(non_relevant_docs) # Total number of docs

    for docs in [relevant_docs, non_relevant_docs]:
        for doc in docs:
            # Extract all terms from title and snippet
            content = f"{doc['title']} {doc['snippet']}"
            terms = re.findall(r'\w+', content.lower())

            # Filter out stop words and original query words
            filtered_terms = [term for term in terms if term not in stop_words and term not in original_keywords]

            # Calculate tf for current doc and update doc freq
            tf = Counter(filtered_terms)
            if docs == relevant_docs:
                relevant_tf.append(tf)
            else:
                non_relevant_tf.append(tf)

            for term in set(filtered_terms): # Set of unique terms
                doc_freq[term] += 1

    # FOR DEBUGGING
    print(relevant_tf)
    print(non_relevant_tf)  

    # Rocchio weights (don't need alpha)
    beta, gamma = 0.75, 0.15

    # Calculate tf-idf scores and term scores using Rocchio
    term_scores = Counter()

    for tf in relevant_tf:
        for term, freq in tf.items():
            idf = math.log(total_docs / doc_freq[term])
            tf_idf_score = (1 + math.log(freq)) * idf
            term_scores[term] += beta * tf_idf_score / len(relevant_docs)

    for tf in non_relevant_tf:
        for term, freq in tf.items():
            idf = math.log(total_docs / doc_freq[term])
            tf_idf_score = (1 + math.log(freq)) * idf
            term_scores[term] -= gamma * tf_idf_score / len(non_relevant_docs)

    # Filter out negative and zero scores (non-relevant terms)
    term_scores = +term_scores

    # Select top 2 terms for query expansion based on adjusted scores
    new_keywords = [word for word, _ in term_scores.most_common(2)]

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
