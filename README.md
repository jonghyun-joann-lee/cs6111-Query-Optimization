# How to Run the Program
1. Install the following in your desired environment  
   ```
   pip3 install google-api-python-client
   pip3 install scikit-learn
   pip3 install numpy
   ```
2. Run the program using :
   ```
   python3 main.py [API_KEY] [ENGINE_ID] [TARGET_PRECISION] "[INITIAL_QUERY]"
   ```
   API_KEY: Google Custom Search JSON API key, ENGINE_ID: Google Custom Search Engine ID, TARGET_PRECISION: target precision (i.e., 0.9), [INITIAL_QUERY]: initial query

# Internal Design

## Libraries Used

- `googleapiclient.discovery`: Used to interact with Google Custom Search API.
- `sklearn.feature_extraction.text.TfidfVectorizer`: Used for transforming text data into TF-IDF vectors.
- `numpy`: Used for several numerical operations.
- *Other Basic libraries*: itertools, collections, sys, re

## Main Components (High Level Overview)

- `google_search`: Function to perform a Google search.
- `collect_feedback`: Function to collect user feedback on search results relevance.
- `refine_query_rocchio`: Function implementing the Rocchio algorithm to refine search queries based on user feedback.
- `main`: Runs the main loop of query refinement.

# Query Modification Method

We make use of the *Title* and *Snippets* provided by Google query results, not the full pages from the web.

The core of our project is within `refine_query_rocchio` function, which refines the search query based on user feedback using the Rocchio Algorithm introduced in class. The method involves the following:

1. Receive user feedback as relevant or irrelevant documents.
2. Transform all text data into TF-IDF vectors using TfidfVectorizer from scikit-learn.
3. Adjust the original query vector by considering the vectors of relevant and irrelevant documents (Rocchio Algorithm).
4. Select new keywords to add to the query based on the adjusted vector. We pick two with higher term importance (TF-IDF value).
5. Determine the order of whole query words (both new and original) for the next round based on their occurrence in relevant documents.

# Additional Information

- Our program does **not** consider **non-HTML** file format. Once detected, we notify the user and do not require any feedback from users.
- If there are fewer than 10 results overall in the first iteration, then our program terminates.
- Program also terminates if in any iteration there are no relevant results.

