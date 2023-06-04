# Model-Based Deep Learning
In this repository we include the source code accompanying our recent book:

Nir Shlezinger and Yonina C. Eldar, "Model-Based Deep Learning", 2023.

## Book Abstract
Signal processing traditionally relies on classical statistical modeling techniques. Such model-based methods utilize mathematical formulations that represent the underlying physics, prior information and additional domain knowledge. Simple classical models are useful but sensitive to inaccuracies and may lead to poor performance when real systems display complex or dynamic behavior. 
More recently, deep learning approaches that use highly  parametric deep neural networks are becoming increasingly popular. Deep learning systems do not rely on mathematical modeling, and learn their mapping from data, which allows them to operate in complex environments. However, they lack the interpretability and reliability of model-based methods, typically require large training sets to obtain good performance,  and tend to be computationally complex.
 
Model-based signal processing methods and data-centric deep learning each have their pros and cons. These  paradigms can be characterized  as edges of a continuous spectrum varying in specificity and parameterization. The methodologies that lie in the middle ground of this spectrum, thus integrating model-based signal processing with deep learning, are referred to as model-based deep learning, and are the focus  here. 
  
This monograph provides a tutorial style presentation of model-based deep learning methodologies. These are families of algorithms  that combine principled mathematical models with data-driven systems to benefit from the advantages of both approaches. Such model-based deep learning methods exploit both partial domain knowledge, via mathematical structures designed for specific problems, as well as learning from limited data.  We accompany our presentation with  running signal processing examples, in super-resolution, tracking of dynamic systems, and array processing. We show how they are expressed using the provided characterization and specialized in each of the detailed methodologies.  Our aim is to facilitate the design and study of future systems at the intersection of signal processing and machine learning that incorporate the advantages of both domains. The source code of our numerical examples are available and reproducible as Python notebooks.

## Simulation Code
The simulation code is given in Python Notebook format. All notebooks are completely self-contained and can be launched without any external dependcies using online platforms such as Google Colab. As a result, no hardware or software requirements are needed, and only a Google Colab account is sufficient for being able to run and display the code in a step-by-step manner. The notebooks are particularly suitable to accompany a course on model-based deep learning, and were designed to support in-class display.

The notebooks are enumerated based on their example index in the book. They are thus divided based on the chapters including numerical examples, which are Chapter 3 (model-based methods) and Chapter 5 (model-based deep learning) of the book.

## Acknowledgements
We are grateful to our students who helped us in fleshing up this repository. We particularly thank Elad Sofer, who lead the experimental effort and its packing as accessible notebooks.


