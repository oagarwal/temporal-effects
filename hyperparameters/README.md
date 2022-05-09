Hyperparameters are optimized for each combination of dataset and model using the oldest temporal training and development set. The same hyperparameters are then used for the remaining temporal splits for the dataset and model combination.

The biLSTM-CRF models are trained using <a href="https://github.com/guillaumegenthial/tf_ner">tf_ner</a>, modified to add ELMo and to perform sentence classification. We will make the code with these modifications public. Hyperparameters are selected by a grid search over batch size, number of epochs and minimum number of steps with a set of predefined values based on dataset size and past recommendations. For NER, we use a batch size of 20, train for a maximum 50 epochs with minimum 2,500 steps. For sentiment classification, domain classification and truecasing, we use a batch size of 100, train for a maximum 10 epochs minimum 1,500 steps. For most tasks, these values are same across representations. For randomly initialized version of NER, sentiment classification and domain classification, the number of epochs is 100, 25 and 25 respectively and the minimum number of steps are 1500, 5000, 1500 respectively. For the remaining, we use the default values in the implementation. 

BERT and RoBERTa are finetuned using the implementation in <a href="https://github.com/huggingface/transformers">HuggingFace transformers</a> with a maximum sequence length of 128. We will make the link to the code with the task-specific changes public. Hyperparameters are selected by a grid search over batch size, number of epochs and learning rate with a set of predefined values based on dataset size and past recommendations. Learning rate is always one of 5e-05 and 5e-06. Batch size varies from 2 to 32 in powers of 2. Number of epochs is from 2 to 6. Following are the exact values for each combination (4 datasets x 3 models x 3 hyperparameters = 36 values). For the remaining, we use the default values.

<table>
<tr>
  <th>Dataset</th>
  <th>Model</th>
  <th>Learning rate</th>
  <th>Total batch Size</th>
  <th>Number of Epochs</th>
</tr>
<tr>
  <td>NER-TTC</td>
  <td>bert-base-cased</td>
  <td>5e-05</td>
  <td>2</td>
  <td>10</td>
</tr>
<tr>
  <td></td>
  <td>bert-large-cased</td>
  <td>5e-06</td>
  <td>2</td>
  <td>6</td>
</tr>
<tr>
  <td></td>
  <td>distilbert-base-cased</td>
  <td>5e-05</td>
  <td>2</td>
  <td>10</td>
</tr>
<tr>
  <td></td>
  <td>roberta-base</td>
  <td>5e-05</td>
  <td>2</td>
  <td>10</td>
</tr>
<tr>
  <td></td>
  <td>roberta-large</td>
  <td>5e-06</td>
  <td>2</td>
  <td>6</td>
</tr>
<tr>
  <td></td>
  <td>distilroberta-base</td>
  <td>5e-05</td>
  <td>2</td>
  <td>10</td>
</tr>
<tr>
  <td>Truecasing-NYT</td>
  <td>bert-base-cased</td>
  <td>5e-05</td>
  <td>16</td>
  <td>3</td>
</tr>
<tr>
  <td></td>
  <td>bert-large-cased</td>
  <td>5e-06</td>
  <td>16</td>
  <td>3</td>
</tr>
<tr>
  <td></td>
  <td>distilbert-base-cased</td>
  <td>5e-05</td>
  <td>16</td>
  <td>2</td>
</tr>
<tr>
  <td></td>
  <td>roberta-base</td>
  <td>5e-05</td>
  <td>16</td>
  <td>3</td>
</tr>
<tr>
  <td></td>
  <td>roberta-large</td>
  <td>5e-06</td>
  <td>16</td>
  <td>3</td>
</tr>
<tr>
  <td></td>
  <td>distilroberta-base</td>
  <td>5e-05</td>
  <td>16</td>
  <td>3</td>
</tr>
<tr>
  <td>Sentiment-Amazon</td>
  <td>bert-base-cased</td>
  <td>5e-06</td>
  <td>32</td>
  <td>3</td>
</tr>
<tr>
  <td></td>
  <td>distilbert-base-cased</td>
  <td>5e-05</td>
  <td>32</td>
  <td>3</td>
</tr>
<tr>
  <td></td>
  <td>roberta-base</td>
  <td>5e-06</td>
  <td>32</td>
  <td>3</td>
</tr>
<tr>
  <td></td>
  <td>distilroberta-base</td>
  <td>5e-05</td>
  <td>32</td>
  <td>3</td>
</tr>
<tr>
  <td>Domain-NYT</td>
  <td>bert-base-cased</td>
  <td>5e-05</td>
  <td>32</td>
  <td>2</td>
</tr>
<tr>
  <td></td>
  <td>distilbert-base-cased</td>
  <td>5e-05</td>
  <td>32</td>
  <td>2</td>
</tr>
<tr>
  <td></td>
  <td>roberta-base</td>
  <td>5e-05</td>
  <td>32</td>
  <td>4</td>
</tr>
<tr>
  <td></td>
  <td>distilroberta-base</td>
  <td>5e-05</td>
  <td>32</td>
  <td>3</td>
</tr>
</table>

Continual pre-training is also performed using HuggingFace. For sentiment and domain classification, we use a batch size of 8 for 1 epoch. For NER, we use a batch size of 4 with 10 epochs. Due to the small size of the NER dataset, each sentence is treated as a separate sequence, employing the line\_by\_line argument in the implementation. Remaining are default values. Since there can be significant variation due to seed, we measure the perplexity on the data used for continual pretraining. If it is an unusually large value, we discard the run and re-pre-train. For finetuning performed in the continual pretraining and self-labeling experiments, the same hyperparameters as the respective dataset+representative fineuning with just gold-standard data are used.

We restrict the review/article length to 50 words for both sentiment and domain classification. For domain classification, it is a design choice for the task i.e. given the first 1-2 paragraphs of an article, predict the domain. For sentiment classification, the length restriction is to be consistent across representations for comparison and the need for resources depending on the model. It is possible to train the GloVe model with more than 50 words but for other representations, using more words leads to either excessively slow training or not enough memory to train it at all.

For LSTM for text classification, we used just the representation of the first word as opposed to the representation of the last word in the forward LSTM concatenated with the representation of the first word in a reverse LSTM. There isn’t much difference between the two approaches, especially with no effect on the trend (verified on domain classification with GloVe).


