# yaTL
Yet-Another-Training-Loop for Natural Language Processing (based on BERT-like models)
This project is aimed to provide a clean working example of training (fine-tuning) 
of neural models for Natural Language Processing. 
The current best-known practices rangefrom incorporating and indulging into a framework such as AllenNLP,
or trying to use directly the example scripts rovided with the transformers python library. 

Sometimes we need more flexibility in the training loop code then a library can provide through hooks. 
Moreover, we don't want to deal with the idiosyncracies outside the scope of our research or work project.
Orthogonal aspects to the algorithm, such as monitoring, checkpointing, logging, periodic evaluations and early stops
are what makes these deep-learning libraries shine.
Organized, readable, testable code are the major selling point for a large framework. 
But often, we would like tighter control on other aspects such as parallelism with multiple GPUs,
floating-point operations, scheduling learning rates, clipping gradients, and any other detail 
that should be reported in a research project. 

This repository is provided as an example code for direct use (don't subclass me!) and extend for your purposes.
A simple use case for Part-of-Speech classification is provided with sane defaults command line arguments.
You are the master of your own training loop, now.
 
