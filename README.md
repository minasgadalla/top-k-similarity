# TOP-K text similarity 

This simple python project finds the TOP-K most similar documents on a dataset given by the user. Technically, it uses standard TF-IDF vector representation and cosine similarity as the distance metric.

More specifically,  as the user starts the execution, she gives the number of text files N and the number K needed for the method and also the directory of the folder in which contains the .txt files. After the successful inputs, the code optimizes the text for the later discovery of the similar documents and calculation of the tfidf value; most important words of the document.

The text optimization is mostly based on removing stopwords and specific special characters that are useless for the processing. Of course, you can easily change the stopwords and the way it handles the text by modifying the text_handler function.

Although this code was created in the context of an undergraduate course, it is made to be used by anyone looking for a simple program which implements that method.
