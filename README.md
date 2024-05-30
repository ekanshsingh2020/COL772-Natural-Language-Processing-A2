# Assignment 2 : Table Cell Classification for Question Answering
### Goal
- We had question,table pairs along with the gold labels in the training data
- We are supposed to filter out the correct cells from the table which correspond to the answer of the given question

### Idea Overview
- The idea was to used transformers (neural methods) for finding the columns and use some rule based methods for finding the correct rows and then combine them to get the cells
- We started with columns first and below is the overview of that approach
- We used glove embeddings to start with for all the words in the question and the table cells
- Used sinusoidal embeddings for the positional encoding of the words
- I also restricted the question length of 60 words so that we can have a fixed size input 
- Then set the embedding dimension to be 100, hidden dimension is 250, number of layers of encoder is 2, used a single head and set dropout to be 0.08
- We computed our column embeddings by summing over all the embeddings of individual column names and would feed that into the MLP along with the question embeddings
- After storing the glove embeddings of the questions, one hot label vectors of the columns and column embeddings, we loaded the classifier and optimizer 
- We then shuffled the complete training data to ensure no bias is encountered
- We trained in batches of 5000 in the training data and 1000 on the validation data provided
- Now we have our column prediction ready with us and now we started with row prediction which is much easier