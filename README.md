RAG pipeline to query wikis. Default is to query the OSRS wiki.

To run, install from the requirements file in backend. Create an API key with your chosen LLM (default is Gemini, may require some editing to work with others).
Create a .env file in the root folder that contains your api key.

Then simply do 'python backend/main.py "[YOUR_QUESTION_HERE]". 
The pipeline will parse the natural language question, identify any specific entities to search for and keywords to use, query the wiki, 
and then generate a response using the provided context. 
