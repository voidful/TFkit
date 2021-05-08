## Models Overview

| task        | available models                    |
| ----------- | ------------------------------------ |
| text generation       | `seq2seq` `clm` `onebyone` `once` `oncectc`  |
| extractive question answering       | `qa` |
| multiple choice question answering       | `mcq` |
| sequence tagging        | `tag` `tagcrf` |
| sentence classification        | `clas` |  
| mask language model        | `clm` |  

## Text Generation
### `seq2seq`
[comment]: <> (::: tfkit.model.seq2seq.model.Model.forward)
[comment]: <> (::: tfkit.model.seq2seq.dataloader)
encoder decoder models for text generation, eg: T5/BART

### `clm`
causal language model, decoder only models for text generation, eg: GPT

### `onebyone`
onebyone text generation, for mask lm generation.

### `once`
once text generation

### `oncectc`
once text generation with ctc loss

## Extractive Question Answering
### `qa`
SQuAD like question answer

## Multiple Choice Question Answering
### `mcq`
softmax from mask token in input

## Sequence Tagging
### `tag`
token classification

### `tagcrf`
token classification with crf layer 

## Sentence Classification 
### `clas`
sentence classification using pooling head from transformer models.

## Mask Language Model
### `mask`
mask token prediction, for self-supervised learning