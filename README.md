## 04 - Analyzing Political Communication on Twitter with NLP (2024)
This project, which is currently being implemented, aims to use natural language processing to analyze over 330,000 self-collected tweets from all Bundestag candidates represented on Twitter (X) in 2021. The project is split into two parts and training/deployment runs via the Google Cloud Platform (GCP):

First, I have classified all tweets according to their political content with PyTorch using the already pre-trained manifestoberta model. These will now be analyzed with logistic multi-level regressions to examine the determinants of specific content across the election campaign. Manifestoberta is based on the multilingual XLM-RoBERTa-Large models, which were tuned on all annotated statements in the text corpus of the MANIFESTO project.

As a second project, I will train and deploy my own model for the classification of negative campaigning based on the tweets using SpaCy. The final goal is to analyze all classified tweets for determinants and dynamics of campaign communication using network analysis and regression models. 

Stay tuned for updates on the project's progress!

## Examples of current code

