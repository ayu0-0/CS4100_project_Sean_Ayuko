# CS 4100 Course Project: Fall 2025

Sean Ediger and Ayuko Okuzawa

**Abstract**

Our goal in this project was to identify different ways to be able to classify songs into different genres using different ML technologies. The dataset we used for this was the GTZAN dataset (https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data) with 10 music genre labels. We mainly used 2 different types of data in this dataset, Tabular and Image data. For the Tabular features we decided to use smaller logistic regression models and then make our way up to more advanced neural networks. For image data (spectrograms), as you might be able to guess, we used CNN's for the task.

**Overview**

Problem:

The problem we are trying to solve here is building a baseline on predicting and classifying songs based on genre. Our motivation for this is as avid music listeners and enthusiasts, sometimes it can be difficult to pinpoint the genre of a song you are listening to, especially if you are completely unaware of the genre of a song you are listening to. Being able to predict and classify these song means gaining more clarity on music exploration and can potentially give clarity on knowing what kind of songs you like more or not like. This problem is interesting because there are many different avenues in which to pursue this. You have so many types of data that can represent music and it's interesting investigating which does best for some things and what might do better for others. I also think that this problem is interesting because it is mixing AI and technology with music / art. The arts are one use case in which people claim AI is behind in compared to other subjects so it's interesting to see the transformation from music -> usable data -> trained model.

Approaches:

Since we had access to a few different data sources, we decided to take a few different approaches to tackle this problem. The three types of data we had access to in this dataset were extracted tabular data, spectrograms (image data), and then the actual audio files. Our way of thinking leading up to this project was start as simple as possible and then work our way up to more complicated models. Our first model we used was simple logistic regression with the tabular data. We thought this would be the easiest way to start off and it was. Next we moved onto using the same data and fitting it with a neural network. We chose to use these approaches first becuase they are very interpretable and it gave us a good baseline for slightly more complex models we might train in the future. We then moved onto using the spectrograms (image data), and we thought the best model for this approach would be CNNs. We used CNNS for this because they they allow for faster and more efficient training of the large data that needs to be processed.

There are plenty of references online of similar ways that we could have gone about solving this problem, but we both thought it might be more fun to
**Expected workload**

- 1st milestone: ~30 lines of code or ~5 hours of work.
- 2nd milestone: ~20 lines of code or ~4 hours of work.
- 3rd milestone: ~30 lines of code or ~6 hours of work.
- 4th milestone: ~30 lines of code or ~5 hours of work.

**Project workflow**

We now introduce the workflow. For example, suppose we want to develop an agent system capable of retrieving information from a database to answer user questions.

- We will first create a small collection of Wikipedia-like documents and then use a search method to find related information to the query. The search method will be based on [**TF-IDF**](https://en.wikipedia.org/wiki/Tf–idf), a technique used in search engines to rank documents according to their relevance to a user’s query.
- We will design prompting formats to guide a language model in generating responses and calling the previously defined search method.
- We will then use a language model from Hugging Face. By applying the loading and generation functions, the model will process the retrieved documents to find answers within the information.
- We will implement a workflow that enables the agent to iteratively generate search actions using the language model and retrieve new information from the database.

**Expected tools and platforms**

We will use Python as the programming language for this project. For data processing, we will work with NumPy and Pandas. To handle text data, we will apply Python’s built-in string operations to process queries and documents. Additionally, we will utilize pretrained language model implementations from the Hugging Face Transformers library.

**Next steps**

1. Form a team with two or three classmates.
2. Make a plan to work on the project, such as setting up a weekly meeting time, a project document / overleaf write-up, etc.
3. Brainstorm about potential project ideas and make a decision by the end of next Friday, Oct 17.
4. Make a presentation file to present the overall project idea and share with the rest of the class, and sign up for a presentation slot on Oct 20 or Oct 23!

## Python Environment

- [Google Colab](https://colab.research.google.com/).
- Local computing ([instructions](https://github.com/VirtuosoResearch/CS4100_project/blob/main/Resources/Set-up-a-Local-Python-Environment.md)) using [Anaconda](https://www.anaconda.com/download).
- Discover cluster: Discovery is a high-performance computing (HPC) resource for the Northeastern University research community. If you need computation resources for your course project, you can apply for access to the Discovery cluster. We provide the instructions for accessing a Discover cluster [in the document here](https://github.com/VirtuosoResearch/CS4100_project/blob/main/Resources/Accessing-and-Using-Discovery-Clusters.md).

## Examples of AI Agents

We describe a few examples of modern AI agents. An AI agent is a software system that utilizes language models to automate tasks.

A travel assistant agent helps plan a trip from start to finish by interpreting a traveler’s request, including dates, budget, and interests. Companies like [Mindtrip AI](https://mindtrip.ai/) and [Booked AI](https://www.booked.ai/) have already built such AI-powered travel planners. These agents search for flights and hotels, check basic rules, and suggest itineraries.

A software engineering agent helps developers build, debug, and maintain software projects more efficiently. Examples include [GitHub Copilot ](https://github.com/features/copilot)and [Tabnine Coding Assistant](https://www.tabnine.com/). Such an agent assists users in understanding codebases, fixing bugs, and managing development workflows.

A customer service agent assists users by answering questions and resolving issues quickly and accurately. Examples include [Zendesk AI Assist](https://www.zendesk.com/service/ai/) and [Intercom Fin AI Agent](https://fin.ai/drlp/ai-agent), which automatically handle customer inquiries and escalate complex cases to human staff. Such an agent’s role is to interpret customer messages, locate useful information, and send helpful responses.

## Related Papers

- [Toolformer](https://arxiv.org/abs/2302.04761): Language Models Can Teach Themselves to Use Tools
- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401): Combining generation with non-parametric memory; useful baseline/variant for your tool-use agent.[ ](https://arxiv.org/abs/2005.11401?utm_source=chatgpt.com)
- [Self-Consistency](https://arxiv.org/abs/2203.11171) Improves Chain of Thought Reasoning in Language Models

- [ReAct](https://arxiv.org/abs/2210.03629?utm_source=chatgpt.com): Synergizing Reasoning and Acting in Language Models.
