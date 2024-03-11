---
title: 📊 Spectral and Spatial Graph Neural Networks
linkTitle: Spectral Graph
summary: An example of using Hugo Blox Builder's Book layout for publishing online courses.
date: '2024-01-24'
type: book
tags:
  - current
---


{{< toc hide_on="xl" >}}

{{< figure src="intro.png" >}}


<!-- ## What you will learn -->

<!-- - Fundamental {{<hl>}}Python programming skills{{</hl>}}
- {{<hl>}}Statistical concepts{{</hl>}} and how to apply them in practice
- Gain experience with the {{<hl>}}Scikit{{</hl>}}, including data visualization with {{<hl>}}Plotly{{</hl>}} and data wrangling with {{<hl>}}Pandas{{</hl>}} -->

## Intro

Over recent years, Graph Neural Networks (GNNs) have garnered significant attention.
However, the proliferation of diverse GNN models, underpinned by various theoretical approaches, complicates the process of model selection, as they are not readily comprehensible within a uniform framework.
Specifically, early GNNs were implemented using spectral theory, while others were developed based on spatial theory . This divergence between spectral and spatial methodologies renders direct comparisons challenging. Moreover, the multitude of models within each domain further complicates the evaluation of their respective strengths and weaknesses.

In this _half-day_ tutorial, we examine the state-of-the-art in GNNs and introduce a comprehensive framework that bridges the spatial and spectral domains, elucidating their complex interrelationship. This emphasis on a comprehensive framework enhances our understanding of GNN operations.
The tutorial's objective is to explore the interplay between key paradigms, such as spatial and spectral-based methods, through a synthesis of spectral graph theory and approximation theory.
We provide an in-depth analysis of the latest research developments in GNNs in this tutorial, including discussions on emerging issues like over-smoothing. A range of well-established GNN models will be utilized to illustrate the universality of our proposed framework.


## Materials/Resources to be distributed
This tutorial is mainly based on our [paper acccepted by ACM Computing Survey](https://dl.acm.org/doi/full/10.1145/3627816) and research papers in spectral graph neural networks 
We have a set of previous [tutorial slides for 1-hour talk](https://imczq.com/csur.pdf), which will be extended to a 3-hour tutorial.
We have been maintaining a collection of related sources at [GitHub](https://github.com/XGraph-Team/Spectral-Graph-Survey).

## Meet your instructor 
{{< spoiler text="Presenter: **Dr. Zhiqian Chen** @ Mississippi State University" >}}
Dr. Zhiqian Chen is an Assistant Professor in the Department of Computer Science and Engineering at Mississippi State University. Specializing in graph machine learning and its applications, Dr. Chen has an impressive portfolio of research published in esteemed journals and conferences such as AAAI, IJCAI, IEEE ICDM, EMNLP, ACM Computing Surveys, and Nature Communication. Beyond his scholarly contributions, he has been an active reviewer for prestigious academic platforms including AAAI, ICML, ICLR, NeuralPS, and SIGKDD. Dr. Chen's research endeavors have garnered support from the National Science Foundation (NSF) and the United States Department of Agriculture (USDA). His accolades include an Outstanding Contribution Award from Toyota Research North America in 2016 and a Best Paper Award from ACM SIGSPATIAL.
{{< /spoiler >}}

{{< spoiler text="Presenter: **Dr. Lei Zhang** @ Virginia Tech" >}}
Dr. Lei Zhang is a Ph.D candidate in Computer Science at Virginia Tech. He is supervised by Dr. Chang-Tien Lu. His research interests span the expansive domains of machine learning and data mining, with a specific emphasis on graph neural networks, graph structure learning, bi-level optimization, neural architecture search, and social network mining. He has published papers in top-tier conferences such as AAAI, ICDM, and IJCAI. He was awarded the Cunningham Fellowship from Virginia Tech in summer 2023.
{{< /spoiler >}}

{{< spoiler text="Presenter: **Dr. Liang Zhao** @ Emory University" >}}
Dr. Liang Zhao is an associate professor at the Department of Computer Science at Emory University. He was an assistant professor at the Department of IST and CS at George Mason University. He obtained his Ph.D. degree as an Outstanding Doctoral Student in the Department of Computer Science at Virginia Tech in 2017. His research interests include data mining, artificial intelligence, and machine learning, with special interests in spatiotemporal and network data mining, deep learning on graphs, nonconvex optimization, and interpretable machine learning. He has published over a hundred papers in top-tier conferences and journals such as KDD, ICDM, TKDE, NeurIPS, Proceedings of the IEEE, TKDD, TSAS, IJCAI, AAAI, WWW, CIKM, SIGSPATIAL, and SDM. He won NSF CAREER Award in 2020. He has also won Cisco Faculty Research Award in 2023, Meta Research Award in 2022, Amazon Research Award in 2020, and Jeffress Trust Award in 2019, , and was ranked as one of the "Top 20 Rising Star in Data Mining" by Microsoft Search in 2016. He has won best paper award in ICDM 2022, best poster runner-up in ACM SIGSPATIAL 2022, Best Paper Award Shortlist in WWW 2021, Best Paper Candidate in ACM SIGSPATIAL 2022, Best Paper Award in ICDM 2019, and best paper candidate in ICDM 2021. He is recognized as "Computing Innovative Fellow Mentor" in 2021 by Computing Research Association. He is a senior member of IEEE. 
{{< /spoiler >}}

## Tentitative Program overview

The total duration comprises 165 minutes allocated for the tutorial and 15 minutes for a break, cumulatively equating to 180 minutes or 3 hours.

### 1, Background: Spectral and Spatial Graph Neural Networks (20 min)

Graph Neural Networks (GNNs) can be bifurcated into two primary categories based on their underlying computational mechanisms: spectral and spatial approaches. Spectral GNNs, grounded in graph signal processing, hinge upon the eigen-decomposition of the graph Laplacian, with foundational contributions such as ChebNet. These networks furnish a theoretically robust framework for articulating convolution operations on graphs, albeit potentially exacting in terms of computational resources. In contrast, spatial GNNs operate within the vertex domain, aggregating information from neighboring nodes, exemplified by GraphSAGE and GCN. This tutorial provides an extensive overview of the recent advancements in both spectral- and spatial-based GNNs, and embarks on a detailed analysis of the evolutionary trajectories of these two GNN paradigms from a technical perspective. Within each category of GNNs, there exists a multitude of theoretical underpinnings guiding their design. This diversity presents challenges in conducting a unified analysis across disparate GNN frameworks. To surmount this barrier, we will investigate a novel perspective we propose, which could potentially harmonize the treatment of both types within a singular framework, and subsequently discuss its merits and demerits.

- Current research of graph neural network (5 min)
- Overview of spatial-based GNN (5 min)
- Overview of spectral-based GNN (5 min)
- The challenge, benefit of studying the connection between spatial and spectral GNN (5 min)

### 2, Spectral graph theory, approximation theory and GNN (50 min)

In this section, we will elucidate the proposed unified perspective through case studies and comprehensive theoretical explanations. For instance, as a fundamental operation, graph matrix normalization will be examined from both spectral and spatial viewpoints, thoroughly justifying the necessity of normalization. Furthermore, we will delve into graph convolution from both perspectives: Spectral graph theory is committed to the analysis of graph structures via the eigenvectors and eigenvalues of matrices associated with the graph, notably the adjacency or Laplacian matrices. Approximation theory, which focuses on identifying the optimal approximation for a function within a designated class of functions, plays a pivotal role in revealing the expressiveness in terms of function of eigenvalues. By leveraging the spectral domain in conjunction with approximation methods, we can categorize all GNNs into linear, polynomial, and rational functions. Correspondingly, the spatial approach can be segregated into first-order, higher-order, and skip-connection strategies among neighbors. These two sets of three-tier categorizations precisely correspond, providing an additional layer of insights, rendering our proposed perspective both unified and conducive to elucidating a broader spectrum of GNNs.

- Spectral graph theory (15 min)
- Approximation theory for GNNs: linear, polynomial and rational functions (15 min)
- Case Study: normalization, GCN and DeepWalk (15 min)
- Unified framework to bridge spatial and spectral GNNs (5 min)

### 3, Theoretical study: viewpoints of uncertainty, sampling oversmoothing, and inverse (60 min)

This section offers a profound and innovative theoretical perspective on prevalent topics within the domain of Graph Neural Networks (GNNs), exploring areas such as uncertainty, sampling, oversmoothing, and inverse problems to furnish a comprehensive framework for analyzing and comprehending various phenomena in GNNs. Moreover, these new insights can be interpreted through both spectral and spatial lenses. The uncertainty principle is employed to understand the distinct global and local effects observable in the spectral and spatial domains, respectively. The examination of sampling theory in the context of graphs is facilitated by a unified framework that integrates explicit closed-form solutions. The phenomenon of oversmoothing in GNNs, characterized by the diminution of node feature distinctiveness as successive layers are applied, can be interpreted within the same conceptual framework. The realm of inverse problems, focusing on resolving edge-related challenges in graphs, presents an alternate viewpoint for understanding GNNs.

- Uncertainty principle: Global vs. Local Perspectives (10 min)
- Theoretical comparison between spatial and spectral methods (10 min)
- Sampling Point of View (15 min)
- Over-smoothing Point of View (15 min)
- Inverse Problem Point of View (10 min)

### 4, Future Directions (20 min)

Recent advancements in using Partial Differential Equations (PDEs) for graph analysis have shown promise. Specifically, the behavior of polynomial and rational functions in this context bears resemblance to diffusion and wave functions, offering innovative tools to study GNNs leveraging these functions. However, current spectral graph theories predominantly cater to simple graphs. Extending these theories to encompass signed, directed, and hypergraphs remains an underdeveloped area. While there has been progress, a comprehensive theoretical framework integrating these graph types is still in development. Future research endeavors will explore the feasibility of formulating a unified spectral theory. This theory would not only encompass various graph types, including simple, signed, directed, and hypergraphs, but also address the dynamics intrinsic to these graphs. The aim is to create a holistic and integrated theoretical understanding of graph behaviors and properties across a diverse range of graph types and their dynamics.

- PDE on graphs (10 min)
- Spectral Graph Theories for Non-simple graphs (5 min)
- Graph Dynamics (5 min)

### 5, Conclusion and Discussion (15 min)

- Conclusion (5 min)
- Q and A (10 min)



## Content in this program

{{< list_children >}}

<!-- ## Meet your instructor -->
<!-- {{< mention "admin" >}} -->

## FAQs

{{< spoiler text="Are there prerequisites?" >}}
Our target audience includes all levles researchers and practitioners in machine learning over graphs. The prerequisites for this tutorial are basic calculus, linear algebra, machine learning, and graph theory. We plan to cover half of the materials for beginners (estimated 30+) and the rest for intermediate and experts (estimated 30+). We expect the audience to come away with an overview of the state-of-art models of spectral graph neural networks. While knowledge in spectral graph theory, approximation theory will facilitate a deeper understanding of the proposed framework, the tutorial can be digested without knowledge of specific GNN models.
{{< /spoiler >}}

<!-- {{< spoiler text="How often do the courses run?" >}}
Continuously, at your own pace.
{{< /spoiler >}} -->

<!-- {{< cta cta_text="Begin the course" cta_link="python" >}} -->
