# Awesome MLOps [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of awesome MLOps tools.

Inspired by [awesome-python](https://github.com/vinta/awesome-python).

- [Awesome MLOps](#awesome-mlops)
    - [CI/CD for Machine Learning](#cicd-for-machine-learning)
    - [Cron Job Monitoring](#cron-job-monitoring)
    - [Data Exploration](#data-exploration)
    - [Data Management](#data-management)
    - [Data Processing](#data-processing)
    - [Data Visualization](#data-visualization)
    - [Feature Store](#feature-store)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Knowledge Sharing](#knowledge-sharing)
    - [Machine Learning Platform](#machine-learning-platform)
    - [Model Lifecycle](#model-lifecycle)
    - [Model Serving](#model-serving)
    - [Optimization Tools](#optimization-tools)
    - [Workflow Tools](#workflow-tools)
- [Resources](#resources)
    - [Articles](#articles)
    - [Other Lists](#other-lists)
    - [Podcasts](#podcasts)
    - [Slack](#slack)
    - [Websites](#websites)
- [Contributing](#contributing)

---

## CI/CD for Machine Learning

*Tools for performing CI/CD for Machine Learning.*

* [CML](https://github.com/iterative/cml) - Open-source library for implementing CI/CD in machine learning projects.

## Cron Job Monitoring

*Tools for monitoring cron jobs (recurring jobs).*

* [HealthchecksIO](https://healthchecks.io/) - Simple and effective cron job monitoring.

## Data Exploration

*Tools for performing data exploration.*

- [Google Colab](https://colab.research.google.com) - Hosted Jupyter notebook service that requires no setup to use.
- [Jupyter Notebook](https://jupyter.org/) - Web-based notebook environment for interactive computing.
- [JupyterLab](https://jupyterlab.readthedocs.io) - The next-generation user interface for Project Jupyter.
- [Jupytext](https://github.com/mwouts/jupytext) - Jupyter Notebooks as Markdown Documents, Julia, Python or R scripts.

## Data Management

*Tools for performing data management.*

* [Arrikto](https://www.arrikto.com/) - Dead simple, ultra fast storage for the hybrid Kubernetes world.
* [DVC](https://dvc.org/) - Management and versioning of datasets and machine learning models.

## Data Processing

*Tools related to data processing and data pipelines.*

* [Airflow](https://airflow.apache.org/) - Platform to programmatically author, schedule, and monitor workflows.
* [Hadoop](https://hadoop.apache.org/) - Framework that allows for the distributed processing of large data sets across clusters of computers.
* [Spark](https://spark.apache.org/) - Unified analytics engine for large-scale data processing.

## Data Visualization

*Tools for data visualization, reports and dashboards.*

* [Data Studio](https://datastudio.google.com) - Reporting solution for power users who want to go beyond the data and dashboards of GA.
* [Metabase](https://www.metabase.com/) - The simplest, fastest way to get business intelligence and analytics to everyone.
* [Redash](https://redash.io/) - Connect to any data source, easily visualize, dashboard and share your data.
* [Superset](https://superset.incubator.apache.org/) - Modern, enterprise-ready business intelligence web application.
* [Tableau](https://www.tableau.com) - Powerful and fastest growing data visualization tool used in the business intelligence industry.

## Feature Store

*Feature store tools for data serving.*

* [Butterfree](https://github.com/quintoandar/butterfree) - A tool for building feature stores. Transform your raw data into beautiful features.
* [Feast](https://feast.dev/) - End-to-end open source feature store for machine learning.

## Hyperparameter Tuning

*Tools and libraries to perform hyperparameter tuning.*

* [Katib](https://github.com/kubeflow/katib) - Kubernetes-based system for hyperparameter tuning and neural architecture search.
* [Tune](https://docs.ray.io/en/latest/tune.html) - Python library for experiment execution and hyperparameter tuning at any scale.

## Knowledge Sharing

*Tools for sharing knowledge to the entire team/company.*

* [Knowledge Repo](https://github.com/airbnb/knowledge-repo) - Knowledge sharing platform for data scientists and other technical professions.
* [Kyso](https://kyso.io/) - One place for data insights so your entire team can learn from your data.

## Machine Learning Platform

*Complete machine learning platform solutions.*

* [Algorithmia](https://algorithmia.com/) - Securely govern your machine learning operations with a healthy ML lifecycle.
* [Allegro AI](https://allegro.ai/) - Transform ML/DL research into products. Faster.
* [CNVRG](https://cnvrg.io/) - An end-to-end machine learning platform to build and deploy AI models at scale.
* [Cubonacci](https://cubonacci.com/) - Intuitive code-first MLOps platform that streamlines the end-to-end machine learning workflow.
* [DAGsHub](https://dagshub.com/) - A platform built on open source tools for data, model and pipeline management.
* [Dataiku](https://www.dataiku.com/) - Platform democratizing access to data and enabling enterprises to build their own path to AI.
* [DataRobot](https://www.datarobot.com/) - AI platform that democratizes data science and automates the end-to-end machine learning at scale.
* [Domino](https://www.dominodatalab.com/) - One place for your data science tools, apps, results, models, and knowledge.
* [Gradient](https://gradient.paperspace.com/) - Multicloud CI/CD and MLOps platform for machine learning teams.
* [H2O](https://www.h2o.ai/) - Open source leader in AI with a mission to democratize AI for everyone.
* [Hopsworks](https://www.hopsworks.ai/) - Open-source platform for developing and operating machine learning models at scale.
* [Iguazio](https://www.iguazio.com/) - Data science platform that automates MLOps with end-to-end machine learning pipelines.
* [Knime](https://www.knime.com/) - Create and productionize data science using one easy and intuitive environment.
* [Kubeflow](https://www.kubeflow.org/) - Making deployments of ML workflows on Kubernetes simple, portable and scalable.
* [LynxKite](https://lynxkite.com/) - A complete graph data science platform for very large graphs and other datasets.
* [Modzy](https://www.modzy.com/) - AI platform and marketplace offering scalable, secure, and ready-to-deploy AI models.
* [Pachyderm](https://www.pachyderm.com/) - Combines data lineage with end-to-end pipelines on Kubernetes, engineered for the enterprise.
* [Sagemaker](https://aws.amazon.com/sagemaker/) - Fully managed service that provides the ability to build, train, and deploy ML models quickly.
* [Valohai](https://valohai.com/) - Takes you from POC to production while managing the whole model lifecycle.

## Model Lifecycle

*Tools for managing model lifecycle (tracking experiments, parameters and metrics).*

* [Comet](https://www.comet.ml/site/) - Track your datasets, code changes, experimentation history, and models.
* [Mlflow](https://mlflow.org/) - Open source platform for the machine learning lifecycle.
* [Neptune AI](https://neptune.ai/) - The most lightweight experiment management tool that fits any workflow.
* [Sacred](https://github.com/IDSIA/sacred) - A tool to help you configure, organize, log and reproduce experiments.

## Model Serving

*Tools for serving models in production.*

* [BentoML](https://bentoml.org) - Open-source platform for high-performance ML model serving.
* [Cortex](https://www.cortex.dev/) - Machine learning model serving infrastructure.
* [GraphPipe](https://oracle.github.io/graphpipe) - Machine learning model deployment made simple.
* [KFServing](https://github.com/kubeflow/kfserving) - Kubernetes custom resource definition for serving ML models on arbitrary frameworks.
* [PredictionIO](https://github.com/apache/predictionio) - Event collection, deployment of algorithms, evaluation, querying predictive results via APIs.
* [Seldon](https://www.seldon.io/) - Take your ML projects from POC to production with maximum efficiency and minimal risk.
* [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) - Flexible, high-performance serving system for ML models, designed for production.

## Optimization Tools

*Optimization tools related to model scalability in production.*

* [Dask](https://dask.org/) - Provides advanced parallelism for analytics, enabling performance at scale for the tools you love.
* [Fiber](https://uber.github.io/fiber/) - Python distributed computing library for modern computer clusters.
* [Mahout](https://mahout.apache.org/) - Distributed linear algebra framework and mathematically expressive Scala DSL.
* [MLlib](https://spark.apache.org/mllib/) - Apache Spark's scalable machine learning library.
* [Modin](https://github.com/modin-project/modin) - Speed up your Pandas workflows by changing a single line of code.
* [Ray](https://github.com/ray-project/ray) - Fast and simple framework for building and running distributed applications.
* [Singa](http://singa.apache.org/en/index.html) - Apache top level project, focusing on distributed training of deep learning and machine learning models.
* [Tpot](https://github.com/EpistasisLab/tpot) - Automated machine learning tool that optimizes machine learning pipelines using genetic programming.

## Workflow Tools

*Tools and frameworks to create workflows or pipelines in the machine learning context.*

* [Argo](https://github.com/argoproj/argo) - Open source container-native workflow engine for orchestrating parallel jobs on Kubernetes.
* [Kedro](https://github.com/quantumblacklabs/kedro) - Library that implements software engineering best-practice for data and ML pipelines.
* [Metaflow](https://metaflow.org/) - Human-friendly lib that helps scientists and engineers build and manage data science projects.
* [Prefect](https://docs.prefect.io/) - A workflow management system, designed for modern infrastructure.

---

# Resources

Where to discover new tools and discuss about existing ones.

## Articles

* [AI Wiki](https://docs.paperspace.com/machine-learning/) (Paperspace)
* [Continuous Delivery for Machine Learning](https://martinfowler.com/articles/cd4ml.html) (Martin Fowler)
* [Continuous delivery for machine learning](https://www.thoughtworks.com/pt/insights/articles/intelligent-enterprise-series-cd4ml) (ThoughtWorks)
* [Delivering on the Vision of MLOps: A maturity-based approach](https://azure.microsoft.com/mediahandler/files/resourcefiles/gigaom-Delivering-on-the-Vision-of-MLOps/Delivering%20on%20the%20Vision%20of%20MLOps.pdf) (GigaOm)
* [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/solutions/machine-learning/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) (Google)

## Other Lists

* [Awesome MLOps](https://github.com/visenger/awesome-mlops)
* [Awesome Production Machine Learning](https://github.com/EthicalML/awesome-production-machine-learning)

## Podcasts

* [Kubernetes Podcast from Google](https://kubernetespodcast.com/)
* [Machine Learning – Software Engineering Daily](https://podcasts.google.com/?feed=aHR0cHM6Ly9zb2Z0d2FyZWVuZ2luZWVyaW5nZGFpbHkuY29tL2NhdGVnb3J5L21hY2hpbmUtbGVhcm5pbmcvZmVlZC8)
* [MLOps.community](https://podcasts.google.com/?feed=aHR0cHM6Ly9hbmNob3IuZm0vcy8xNzRjYjFiOC9wb2RjYXN0L3Jzcw)
* [This Week in Machine Learning & AI](https://twimlai.com/)

## Slack

* [Kubeflow Workspace](https://kubeflow.slack.com/#/)
* [MLOPs Community Wokspace](https://mlops-community.slack.com)

## Websites

* [Feature Stores for ML](http://featurestore.org/)
* [MLOps](https://mlops.org/)
* [MLOps Community](https://mlops.community/)

# Contributing

All contributions are welcome! Please take a look at the [contribution guidelines](https://github.com/kelvins/awesome-mlops/blob/master/CONTRIBUTING.md) first.
