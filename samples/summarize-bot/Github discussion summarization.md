# Github discussion summarization

## 1- Introduction

In this repo we provide a sample that span different summarization techniques, integrated into Microsoft bot-framework. The sample is aiming to summarize Github issue discussions to provide a compact and helpful version of the whole discussion.



## 2- Data

Github discussions consists of an initial comment or post that is submitted by some user. Later different users respond with comments that usually targets or provide some response to the original post.

We collected data from Github to use in our model development, however the collected data wasn't annotated due to time limitations. We decided to use a data that is similar in structure. We decided to use forum discussion dataset (https://arxiv.org/pdf/1805.10390.pdf) collected from tripadvisor website.  The dataset is annotated and have a human selected sentences as a summary.

Another data that later was used for evaluation is MsWord dataset. The dataset consists of Microsoft word documents alongside human annotation summary.

The data description is as follows.



|                               | Threads (docs) | Train | Val  | Test |
| ----------------------------- | -------------- | ----- | ---- | ---- |
| Github                        | 885            | -     | -    | -    |
| Forum discussion(tripadvisor) | 700            | 500   | 100  | 100  |
| MsWord                        | 532            | 266   | 138  | 128  |





## 3- Models

We either developed, used or modified a set of models for summarization. We incorporated a set of 5 models (2 non neural network based, and 3 neural network based). The models span a wide variety of techniques (Attention, Multitask learning, Contextual embeddings, etc..).

### 3.1- Sumy

Sumy is an open source python summarization package, it as used as a baseline.

### 3.2- LSA + Clustering

Another non-neural network baseline that is implemented is LSA + clustering baseline. To perform summarization all comments are converted into sentences, sentences are then converted into embeddings using LSA. Following sentence embedding, sentences are then clustered into *N* clusters using kmeans. The head of each cluster (closest to center) is then picked as the cluster representative and added to the summary.

### 3.3- SummaRuNNer

SummaRuNNer (https://arxiv.org/pdf/1611.04230.pdf) is an extractive summarization model proposed by Nallapati et al. The system consists of 2 Bi-LSTM layers for sentence and document embedding respectively. Following the sentence and document embedding, a fully layer is attached to act as a classifier to predict either to include sentence in summary or not.

![1563903090230](.\images\1563903090230.png)



We implemented the original model and used at as baseline. We later proposed a series of modifications to   enhance the models and render it more effective to summarize discussions like data.

#### 3.3.1- Proposed modifications:

##### 3.3.1.1- Pre-training:

Due to small size of our data, we decided to try the impact of pretraining the model using a larger dataset. Following the pretraining, finetuning using the forum discussion dataset is performed. We chose CNN/Dailymail dataset for pretraining, the CNN dataset contains more than 200K training samples, which makes it a reasonable choice for pretraining.

##### 3.3.1.2- Co-attention:

Another modification we proposed to benefit from the nature of discussion data is **co-attention**. In discussion data, threads usually consists of an initial comment (post) and following comments that reply to the original comment. Therefor it is essential to consider the relation between a comment and the original post in order to make a decision to include the comment in a the summary or not. With that in mind we decided to integrate a co-attention mechanism, the co-attention performs attention between the post sentences and each of the comment sentences. The co-attention result is then passed to the classifier to make a decision.

![1563903610377](.\images\1563903610377.png)

##### 3.3.1.3- Contextual Embeddings

Word embeddings are essential part in any Neural Network model, however shallow word embeddings are more liable to fail in capturing information that is encoded in surrounding words. Later on contextual embeddings emerged into the field, contextual embeddings capture not only word information, but also the information within the context surrounding each word.

With that said we decided to see the impact of using **contextual embeddings (BERT)** instead of shallow ones. BERT embeddings are used and kept fixed during the model training, sentence Bi-LSTM is removed and replaced with average pooling the BERT word embeddings. 

![1563903582960](.\images\1563903582960.png)

##### 3.3.1.4- Keywords extraction

Another modification we did to the SummaRuNNer model is adding additional input. We extracted keywords from each sentence using RAKE algorithm and then fed the keywords separately to the model. The sentence embedding is then concatenated with the keywords embedding.

The intuition behind that modification is that while it is more likely to select sentence in summary based on keywords, attention mechanism may fail to attend to important keywords. So we decided to give the model a push by providing the keywords separately as an input to the model.

![1563903554967](.\images\1563903554967.png)

#### 3.3.2- Results and Discussion

| **Model**                                    |  **R1**   |  **R2**   |  **RL**   | **Avg Rouge** |
| :------------------------------------------- | :-------: | :-------: | :-------: | :-----------: |
| Summy                                        |    38     |   15.06   |   21.95   |     25.00     |
| **LSA +** **kmeans**                         | **35.94** | **19.05** | **23.03** |   **26.06**   |
| LSA (wikipedia) + kmeans                     |   35.4    |   18.49   |   22.57   |     25.48     |
| SummaRuNNer                                  |   36.97   |   15.84   |   24.5    |     25.77     |
| SummaRuNNer (pretrained)                     |   31.81   |   10.88   |   19.61   |     20.76     |
| SummaRuNNer   (Forum)  + co-attention        |   37.46   |   16.17   |   24.5    |     26.04     |
| SummaRuNNer   (Forum) + BERT                 |   38.48   |   16.88   |   25.63   |     26.99     |
| SummaRuNNer   (Forum)  + co-attention + BERT | **39.36** | **17.71** | **26.78** |   **27.95**   |



We can see in the results that pretraining the model using CNN data didn't help the model, we argue that this behavior is most probably due to the fundamental difference between the two datasets, as CNN is news data while the forum discussion dataset consists of discussions over tripadvisor.

It is also clear that integrating co-attention mechanism and contextual embeddings are both helpful to the model.



### 3.4- BERTSum

![1563926657821](.\images\1563926657821.png)

We only tried doing pretraining on the BERTSum model. The results are as follows

| **Model**                                                  | **R1**    | **R2** | **RL**    | **Avg Rouge** |
| ---------------------------------------------------------- | --------- | ------ | --------- | ------------- |
| Fine-tune BERT for Extractive   Summarization (Forum)      | 31.96     | 24.98  | 29.93     | 28.95         |
| Fine-tune BERT for Extractive   Summarization (pretrained) | **32.95** | **26** | **31.07** | **30**        |

### 3.5- siatl

![1563926811266](.\images\1563926811266.png)

This model is introduced in "https://www.aclweb.org/anthology/N19-1213", the model is developed for sentence classification. We tried using it for extractive summarization, by classify a sentence as in summary or not. We also integrated slight modifications to it to attend for the data we have.

#### 3.5.1- Proposed modifications:

##### 3.5.1.1- co-attention:

Similar to what was done in SummaRuNNer model, co-attention mechanism was integrated in the model. We also tried using co-attention only, self attention only and co-attention followed by self attention.

#### 3.5.2- Results:



| Siatl   self-attention only                                  | 45.15     | 26.12     | 43.3      | 38.19     |
| ------------------------------------------------------------ | --------- | --------- | --------- | --------- |
| **Siatl**   **co-attention only**                            | **46.5**  | **28.53** | **44.65** | **39.89** |
| Siatl   co-attention + self-attention                        | 46.32     | 28.69     | 44.41     | 39.80     |
| Siatl   co-attention only (bidirectional)                    | 45.46     | 26.17     | 43.56     | 38.39     |
| **Siatl**   **co-attention + self-attention (bidirectional)** | **46.44** | **28.01** | **44.55** | **39.66** |

We can notice that using co-attention is actually helpful, combining co-attention and self attention can also be helpful. More experiments on different datasets needs to be performed to get a better conclusion of which of the variants (co-attention, or co-attention + self attention) is more suited which type of data.



## 4- Evaluation on MSWord dataset

We also carried out model evaluation using MSWrod dataset, we compared our results with other DNN baselines.



| **Model**                                                    |  **R1**   |  **R2**   |  **RL**   | **Avg Rouge** |
| :----------------------------------------------------------- | :-------: | :-------: | :-------: | :-----------: |
| SummaRuNNer   (self-implemented)                             |   63.48   |   54.51   |   61.66   |     59.88     |
| SummaRuNNer + co-attention                                   |   64.23   |   55.23   |   62.21   |     60.56     |
| SummaRuNNer + BERT                                           |   65.81   |   7.99    |   63.9    |     62.57     |
| **SummaRuNNer + co-attention + BERT**                        | **66.12** | **58.56** | **64.48** |   **63.05**   |
| SummaRuNNer   + co-attention + BERT + keywords               |   66.44   |   57.52   |   64.36   |     62.77     |
| Siatl co-attention + self-attention (trained on Forum)       |   44.81   |   27.02   |   42.79   |     38.20     |
| Siatl   co-attention + self-attention (trained on Forum - tuned on MsWord) |   43.44   |   24.52   |   41.27   |     36.41     |
| Lead3                                                        |   41.62   |   29.55   |   39.74   |     36.97     |
| Oracle3                                                      |   65.01   |   59.44   |   64.03   |     62.82     |
| **SummaRunner**                                              | **63.56** | **53.57** | **61.89** |   **59.67**   |
| Seq2Seq+RNN                                                  |   63.89   |   54.22   |   62.36   |     58.29     |
| Cheng&Lapata                                                 |   60.21   |   49.81   |   58.62   |     56.21     |
| BertSum+Classifier                                           |   58.63   |   47.75   |   56.95   |     54.44     |
| BertSum+Transformer                                          |   43.49   |   32.14   |   41.85   |     39.16     |

We can see that our implementation of SummaRuNNer is getting a very close scores to the other implementation, which suggests that both implementations are aligned. We can see a consistent behavior of adding co-attention and BERT, adding both modifications enhances model scores similar to the behavior on forum discussions dataset.

To integrate co-attention with MsWord dataset which doesn't contain an actual post (initial comment) we use the first 3 sentences of the document as the post. more analysis needs to be done to figure out the best number of sentences to use as post.



## 5- Simple Analysis

Why multitask learning model (siatl) suffers from a huge dropout in performance on MsWord dataset while at the same time performs exceptionally good on the Forum discussion dataset.

We did simple analysis to try to figure out why such behavior appears. One concern was the size of the output of siatl model (number of sentences classified as in summary). The following table shows the average number of sentences in (Gold standard summary, siatl output, SummaRuNNer output) for both datasets.



|                             |                      |       | Number of Sentences |      |      |      |
| ------ | :-----------------: | :---------------: | :------------------: | :---: | :--: | :--: |
|        | Gold                |  | sital |  | SummaRuNNer |  |
|        | avg                 | std               | avg                  | std   | avg | std |
| Forum  | 13.38               | 8.16              | 16                   | 6.48  | 8.2 | 3.52 |
| MSWord | 7.15                | 7.58              | 21.85                | 12.69 | 6.4 | 3.6 |

Since siatl was developed for sentence classification purpose, the model has no knowledge of history or document. So it classifies each sentence independently which results of a large number of sentences in the output. One modification we plan to integrate in future work is integrating a history concept in the model. history can be integrated by means of any recurrent NN based (LSTM, GRU, etc..)  node, it can be also integrated with different mechanism, we still need to run some experiments to find the best way.

## 6- Conclusion

- Document Summarizing models can be used effectively for discussions summarization. Which render Github discussions summarization as a valid and doable task. 

- Integrating **co-attention** mechanism between initial post and comments proved to be helpful not only for discussion based datasets.

- Using contextual embeddings **(BERT)** helps achieving better ROUGE scores compared to shallow embeddings.

- While injecting **keywords** into the model can be helpful sometimes, it is not always the case **(more analysis needs to be done)**

## 7- Future work

- Annotate Github collected data.

- re-train models on Github data, and get an insight of how models perform on Github data.
- integrate history in siatl model to limit the size of the output.
- Experiment with integrating backtranslation instead of keywords.



## 8- References

1- Nallapati, Ramesh, Feifei Zhai, and Bowen Zhou. "Summarunner: A recurrent neural network based sequence model for extractive summarization of documents." *Thirty-First AAAI Conference on Artificial Intelligence*. 2017

2- Tarnpradab, Sansiri, Fei Liu, and Kien A. Hua. "Toward extractive summarization of online forum discussions via hierarchical attention networks." *The Thirtieth International Flairs Conference*. 2017.

3- Chronopoulou, Alexandra, Christos Baziotis, and Alexandros Potamianos.  "An embarrassingly simple approach for transfer learning from pretrained language models." *arXiv preprint arXiv:1902.10547* (2019).

4- Liu, Yang. "Fine-tune BERT for Extractive Summarization." *arXiv preprint arXiv:1903.10318* (2019).

