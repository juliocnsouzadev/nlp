# BERT
### Bidirectional Enconder Respresentation from Transformers
- Enconder Representations: language modeling system, pre-tranined with unlabeled dadta. Then fine-tuning
- from Transformer: based on powerful NLP algorithm. Defines the architecture of BERT.
- Bidirectional: uses with left and right context when dealing with a word. Defines the traingin process.

## Word embedding
### (Input *X* Embedding Matrix = N dimention Vector) * Context Matrix = Output Softmax
# Skip-gram model:
in sentence "In spite of everything, I still believe people are really good at heart". The word "good" produces pairs ("good", "are"), ("good", "really"), ("good", "at"), ("good", "heart") at target/context 