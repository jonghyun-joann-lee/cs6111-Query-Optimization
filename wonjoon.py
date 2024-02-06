from googleapiclient.discovery import build
from collections import Counter
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
        print(f" Summary: {result['snippet']}\n]")
        
        # Ask for user feedback
        feedback = input(f"Relevant (Y/N) for Result {index}?").strip().lower()
        if feedback == 'y':
            relevant_results.append(result)
    return relevant_results


def refine_query(original_query, relevant_results):
    # Basic cleanup and word extraction from the original query to avoid duplication
    original_keywords = set(re.findall(r'\w+', original_query.lower()))

    # Collect words from relevant results
    words = []
    for result in relevant_results:
        # Extract words from title and snippet
        content = f"{result['title']} {result['snippet']}"
        words.extend(re.findall(r'\w+', content.lower()))

    # Count word frequencies, excluding original query words
    word_freq = Counter(words)
    for word in original_keywords:
        del word_freq[word]  # Exclude words already in the query

    # Pick the top 2 most frequent new words that are not in the original query
    new_keywords = [word for word, freq in word_freq.most_common(2)]

    # Form the new query by combining original query and new keywords
    new_query = f"{original_query} {' '.join(new_keywords)}"

    return new_query


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
        query = refine_query(query, relevant_results)
        print(f"Augmenting by {query}")


if __name__ == "__main__":
    main()
