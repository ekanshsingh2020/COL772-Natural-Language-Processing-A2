# Assignment 2 : Table Cell Classification for Question Answering
### Goal
- We had question,table pairs along with the gold labels in the training data
- We are supposed to filter out the correct cells from the table which correspond to the answer of the given question

### Idea Overview
- The final code can be found in ee1200490.zip
- The idea was to used transformers (neural methods) for finding the columns and use some rule based methods for finding the correct rows and then combine them to get the cells
  - Column Prediction
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
  - Row Prediction
    - We initially tried using neural methods for row prediction as well but since the limit on the number of columns was 64 but there was no as such limit on the number of rows, we were not able to understand how to model the rows and input into the transformers or biLSTM
    - Given the question and table pair, we gave score to each row, for the sake of simplicity we only considered one row to be correct (This is because I observed in the given data, only 5-6% data samples had multiple rows correct and I was ready to sacrifice them to make the model working)
    - One basic rule that we wrote for giving scores was to first match words in the question and then in the row , if matched then find the difference (if any) in the absolute value of the index where both words are found and penalise for higher such distance, you can look up the code
    - Also the words matched with higher length were given higher weightage which makes sense
    - After all this just take the row with maximum score
    - In the code, their is some neural network code for row prediction, you can ignore that as that was just for experimenting and is dead code
- These is our column and row prediction models for this unique problem statement

### Result
This landed me 4th place in the class with column accuracy of 84% and row accuracy of 89%, thanks to the rule based methods (This taught me that observing the data is highly useful for such projects)

Please let me know if you have anything to ask in the code :)