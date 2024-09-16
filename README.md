# Part-of-Speech Tagger using Bi-LSTM

This project implements a Part-of-Speech (PoS) Tagger using a Bidirectional Long Short-Term Memory (Bi-LSTM) model. The model is trained on a combination of datasets provided by NLTK and uses the universal tagset for labeling.

A Part-of-Speech (PoS) tagger is a tool used in natural language processing (NLP) to label each word in a sentence with its corresponding part of speech, such as noun, verb, adjective, etc. This process helps in understanding the syntactic structure of the sentence and the role each word plays within it.

### Example

Consider the sentence: `The quick brown fox jumps over the lazy dog.`

```
[('the', 'det'), ('quick', 'adj'), . . . ('lazy', 'adj'), ('dog', 'noun')]
```

Each word is tagged according to its part of speech, which helps in understanding how the words relate to each other. This tagging is crucial for various NLP tasks like parsing, text-to-speech systems, and information extraction.

## Datasets

The following datasets were combined for training the model:
- **Treebank**
- **Brown**
- **Conll2000**

## Tagset

The PoS tags used in this project follow the `universal_tagset`, which includes:
1. **ADJ** - Adjective
2. **ADP** - Adposition
3. **ADV** - Adverb
4. **CONJ** - Conjunction
5. **DET** - Determiner
6. **NOUN** - Noun
7. **NUM** - Numeral
8. **PRON** - Pronoun
9. **PRT** - Particle
10. **VERB** - Verb
11. **.** - Punctuation
12. **X** - Other (residual elements)

## Project Workflow

The project follows these steps to train the PoS tagger:

1. **Data Preparation**:
   - Combine datasets from NLTK.
   - Split the data into training, validation, and test sets.

2. **Tokenization**:
   - Create tokenizers for sentences (input `x`) and tags (output `y`).
   - Automatically generate vocabulary while creating tokenizers.

3. **Sequence Conversion**:
   - Convert sentences into sequences of tokens.
   - Pad sequences to ensure uniform input size.

4. **One-Hot Encoding**:
   - Convert tag sequences to one-hot encoding since they are categorical.

5. **Model Training**:
   - Train the Bi-LSTM model using the processed data.

6. **Testing and Inference**:
   - Test the model on the test dataset.
   - Perform inference on new sentences to predict PoS tags.

Here's an updated section to include in the README, covering the model architecture and the flow of data dimensions:

## Model Architecture

1. **Embedding Layer**: 
   - This layer converts input tokens into dense vector representations of a specified dimension (`embedding_dim`). It captures the semantic meaning of words.
   - **Input Shape**: `(Batch size, MAX_SEN_LEN)` - e.g., `(256, 161)`
   - **Output Shape**: `(Batch size, MAX_SEN_LEN, Embedding_Dim)` - e.g., `(256, 161, 128)`

2. **Bidirectional LSTM Layer**:
   - This layer processes the input sequences in both forward and backward directions to capture dependencies from both ends of the sentence.
   - **Input Shape**: `(Batch size, MAX_SEN_LEN, Embedding_Dim)` - e.g., `(256, 161, 128)`
   - **Output Shape**: `(Batch size, MAX_SEN_LEN, 2 * lstm_units)` - e.g., `(256, 161, 256)`

3. **TimeDistributed Dense Layer**:
   - The `TimeDistributed` Dense layer applies a fully connected layer to each time step independently, producing a probability distribution over the possible PoS tags (`num_classes`) for each token.
   - **Input Shape**: `(Batch size, MAX_SEN_LEN, 2 * lstm_units)` - e.g., `(256, 161, 256)`
   - **Output Shape**: `(Batch size, MAX_SEN_LEN, num_classes)` - e.g., `(256, 161, 10)`

4. **Compilation**:
   - The model is compiled using the `categorical_crossentropy` loss function, which is suitable for the multi-class classification problem of PoS tagging.
   - The `adam` optimizer is used for training, and the model's performance is evaluated using accuracy.

This architecture allows the model to effectively capture contextual information in sentences, making it suitable for the PoS tagging task.