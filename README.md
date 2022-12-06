# Awesome MLOps [![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)

A curated list of awesome MLOps tools.

Inspired by [awesome-python](https://github.com/vinta/awesome-python).

- [Awesome MLOps](#awesome-mlops)
    - [AutoML](#automl)
    - [CI/CD for Machine Learning](#cicd-for-machine-learning)
    - [Cron Job Monitoring](#cron-job-monitoring)
    - [Data Catalog](#data-catalog)
    - [Data Enrichment](#data-enrichment)
    - [Data Exploration](#data-exploration)
    - [Data Management](#data-management)
    - [Data Processing](#data-processing)
    - [Data Validation](#data-validation)
    - [Data Visualization](#data-visualization)
    - [Feature Engineering](#feature-engineering)
    - [Feature Store](#feature-store)
    - [Hyperparameter Tuning](#hyperparameter-tuning)
    - [Knowledge Sharing](#knowledge-sharing)
    - [Machine Learning Platform](#machine-learning-platform)
    - [Model Fairness and Privacy](#model-fairness-and-privacy)
    - [Model Interpretability](#model-interpretability)
    - [Model Lifecycle](#model-lifecycle)
    - [Model Serving](#model-serving)
    - [Model Testing & Validation](#model-testing--validation)
    - [Optimization Tools](#optimization-tools)
    - [Simplification Tools](#simplification-tools)
    - [Visual Analysis and Debugging](#visual-analysis-and-debugging)
    - [Workflow Tools](#workflow-tools)
- [Resources](#resources)
    - [Articles](#articles)
    - [Books](#books)
    - [Events](#events)
    - [Other Lists](#other-lists)
    - [Podcasts](#podcasts)
    - [Slack](#slack)
    - [Websites](#websites)
- [Contributing](#contributing)

---

## AutoML

*Tools for performing AutoML.*

* [AutoGluon](https://github.com/awslabs/autogluon) - Automates machine learning tasks enabling you to easily achieve strong predictive performance.
* [AutoKeras](https://github.com/keras-team/autokeras) - AutoKeras goal is to make machine learning accessible for everyone.
* [AutoPyTorch](https://github.com/automl/Auto-PyTorch) - Automatic architecture search and hyperparameter optimization for PyTorch.
* [AutoSKLearn](https://github.com/automl/auto-sklearn) - Automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator.
* [EvalML](https://github.com/alteryx/evalml) - A library that builds, optimizes, and evaluates ML pipelines using domain-specific functions.
* [FLAML](https://github.com/microsoft/FLAML) - Finds accurate ML models automatically, efficiently and economically.
* [H2O AutoML](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) - Automates ML workflow, which includes automatic training and tuning of models.
* [MindsDB](https://github.com/mindsdb/mindsdb) - AI layer for databases that allows you to effortlessly develop, train and deploy ML models.
* [MLBox](https://github.com/AxeldeRomblay/MLBox) - MLBox is a powerful Automated Machine Learning python library.
* [Model Search](https://github.com/google/model_search) - Framework that implements AutoML algorithms for model architecture search at scale.
* [NNI](https://github.com/microsoft/nni) - An open source AutoML toolkit for automate machine learning lifecycle.

## CI/CD for Machine Learning

*Tools for performing CI/CD for Machine Learning.*

* [ClearML](https://github.com/allegroai/clearml) - Auto-Magical CI/CD to streamline your ML workflow.
* [CML](https://github.com/iterative/cml) - Open-source library for implementing CI/CD in machine learning projects.

## Cron Job Monitoring

*Tools for monitoring cron jobs (recurring jobs).*

* [Cronitor](https://cronitor.io/cron-job-monitoring) - Monitor any cron job or scheduled task.
* [HealthchecksIO](https://healthchecks.io/) - Simple and effective cron job monitoring.

## Data Catalog

*Tools for data cataloging.*

* [Amundsen](https://www.amundsen.io/) - Data discovery and metadata engine for improving the productivity when interacting with data.
* [Apache Atlas](https://atlas.apache.org) - Provides open metadata management and governance capabilities to build a data catalog.
* [CKAN](https://github.com/ckan/ckan) - Open-source DMS (data management system) for powering data hubs and data portals.
* [DataHub](https://github.com/linkedin/datahub) - LinkedIn's generalized metadata search & discovery tool.
* [Magda](https://github.com/magda-io/magda) - A federated, open-source data catalog for all your big data and small data.
* [Metacat](https://github.com/Netflix/metacat) - Unified metadata exploration API service for Hive, RDS, Teradata, Redshift, S3 and Cassandra.
* [OpenMetadata](https://open-metadata.org/) - A Single place to discover, collaborate and get your data right.

## Data Enrichment

*Tools and libraries for data enrichment.*

* [Upgini](https://github.com/upgini/upgini) - Enriches training datasets with features from public and community shared data sources.

## Data Exploration

*Tools for performing data exploration.*

* [Apache Zeppelin](https://zeppelin.apache.org/) - Enables data-driven, interactive data analytics and collaborative documents.
* [BambooLib](https://github.com/tkrabel/bamboolib) -  An intuitive GUI for Pandas DataFrames.
* [Google Colab](https://colab.research.google.com) - Hosted Jupyter notebook service that requires no setup to use.
* [Jupyter Notebook](https://jupyter.org/) - Web-based notebook environment for interactive computing.
* [JupyterLab](https://jupyterlab.readthedocs.io) - The next-generation user interface for Project Jupyter.
* [Jupytext](https://github.com/mwouts/jupytext) - Jupyter Notebooks as Markdown Documents, Julia, Python or R scripts.
* [Polynote](https://polynote.org/) - The polyglot notebook with first-class Scala support.

## Data Management

*Tools for performing data management.*

* [Arrikto](https://www.arrikto.com/) - Dead simple, ultra fast storage for the hybrid Kubernetes world.
* [BlazingSQL](https://github.com/BlazingDB/blazingsql) - A lightweight, GPU accelerated, SQL engine for Python. Built on RAPIDS cuDF.
* [Delta Lake](https://github.com/delta-io/delta) - Storage layer that brings scalable, ACID transactions to Apache Spark and other engines.
* [Dolt](https://github.com/dolthub/dolt) - SQL database that you can fork, clone, branch, merge, push and pull just like a git repository.
* [Dud](https://github.com/kevin-hanselman/dud) - A lightweight CLI tool for versioning data alongside source code and building data pipelines.
* [DVC](https://dvc.org/) - Management and versioning of datasets and machine learning models.
* [Git LFS](https://git-lfs.github.com) - An open source Git extension for versioning large files.
* [Hub](https://github.com/activeloopai/Hub) - A dataset format for creating, storing, and collaborating on AI datasets of any size.
* [Intake](https://github.com/intake/intake) - A lightweight set of tools for loading and sharing data in data science projects.
* [lakeFS](https://github.com/treeverse/lakeFS) - Repeatable, atomic and versioned data lake on top of object storage.
* [Marquez](https://github.com/MarquezProject/marquez) - Collect, aggregate, and visualize a data ecosystem's metadata.
* [Milvus](https://github.com/milvus-io/milvus/) - An open source embedding vector similarity search engine powered by Faiss, NMSLIB and Annoy.
* [Pinecone](https://www.pinecone.io) - Managed and distributed vector similarity search used with a lightweight SDK.
* [Qdrant](https://github.com/qdrant/qdrant) - An open source vector similarity search engine with extended filtering support.
* [Quilt](https://github.com/quiltdata/quilt) - A self-organizing data hub with S3 support.

## Data Processing

*Tools related to data processing and data pipelines.*

* [Airflow](https://airflow.apache.org/) - Platform to programmatically author, schedule, and monitor workflows.
* [Azkaban](https://github.com/azkaban/azkaban) - Batch workflow job scheduler created at LinkedIn to run Hadoop jobs.
* [Dagster](https://github.com/dagster-io/dagster) - A data orchestrator for machine learning, analytics, and ETL.
* [Hadoop](https://hadoop.apache.org/) - Framework that allows for the distributed processing of large data sets across clusters.
* [Spark](https://spark.apache.org/) - Unified analytics engine for large-scale data processing.

## Data Validation

*Tools related to data validation.*

* [Cerberus](https://github.com/pyeve/cerberus) - Lightweight, extensible data validation library for Python.
* [Great Expectations](https://greatexpectations.io) - A Python data validation framework that allows to test your data against datasets.
* [JSON Schema](https://json-schema.org/) - A vocabulary that allows you to annotate and validate JSON documents.
* [TFDV](https://github.com/tensorflow/data-validation) - An library for exploring and validating machine learning data.

## Data Visualization

*Tools for data visualization, reports and dashboards.*

* [Count](https://count.co) - SQL/drag-and-drop querying and visualisation tool based on notebooks.
* [Dash](https://github.com/plotly/dash) - Analytical Web Apps for Python, R, Julia, and Jupyter.
* [Data Studio](https://datastudio.google.com) - Reporting solution for power users who want to go beyond the data and dashboards of GA.
* [Facets](https://github.com/PAIR-code/facets) - Visualizations for understanding and analyzing machine learning datasets.
* [Lux](https://github.com/lux-org/lux) - Fast and easy data exploration by automating the visualization and data analysis process.
* [Metabase](https://www.metabase.com/) - The simplest, fastest way to get business intelligence and analytics to everyone.
* [Redash](https://redash.io/) - Connect to any data source, easily visualize, dashboard and share your data.
* [Superset](https://superset.incubator.apache.org/) - Modern, enterprise-ready business intelligence web application.
* [Tableau](https://www.tableau.com) - Powerful and fastest growing data visualization tool used in the business intelligence industry.

## Feature Engineering

*Tools and libraries related to feature engineering.*

* [Feature Engine](https://github.com/feature-engine/feature_engine) - Feature engineering package with SKlearn like functionality.
* [Featuretools](https://github.com/alteryx/featuretools) - Python library for automated feature engineering.
* [TSFresh](https://github.com/blue-yonder/tsfresh) - Python library for automatic extraction of relevant features from time series.

## Feature Store

*Feature store tools for data serving.*

* [Butterfree](https://github.com/quintoandar/butterfree) - A tool for building feature stores. Transform your raw data into beautiful features.
* [ByteHub](https://github.com/bytehub-ai/bytehub) - An easy-to-use feature store. Optimized for time-series data.
* [Feast](https://feast.dev/) - End-to-end open source feature store for machine learning.
* [Feathr](https://github.com/linkedin/feathr) - An enterprise-grade, high performance feature store.
* [Featureform](https://github.com/featureform/featureform) - A Virtual Feature Store. Turn your existing data infrastructure into a feature store.
* [Tecton](https://www.tecton.ai/) - A fully-managed feature platform built to orchestrate the complete lifecycle of features.

## Hyperparameter Tuning

*Tools and libraries to perform hyperparameter tuning.*

* [Advisor](https://github.com/tobegit3hub/advisor) - Open-source implementation of Google Vizier for hyper parameters tuning.
* [Hyperas](https://github.com/maxpumperla/hyperas) - A very simple wrapper for convenient hyperparameter optimization.
* [Hyperopt](https://github.com/hyperopt/hyperopt) - Distributed Asynchronous Hyperparameter Optimization in Python.
* [Katib](https://github.com/kubeflow/katib) - Kubernetes-based system for hyperparameter tuning and neural architecture search.
* [KerasTuner](https://github.com/keras-team/keras-tuner) - Easy-to-use, scalable hyperparameter optimization framework.
* [Optuna](https://optuna.org/) - Open source hyperparameter optimization framework to automate hyperparameter search.
* [Scikit Optimize](https://github.com/scikit-optimize/scikit-optimize) - Simple and efficient library to minimize expensive and noisy black-box functions.
* [Talos](https://github.com/autonomio/talos) - Hyperparameter Optimization for TensorFlow, Keras and PyTorch.
* [Tune](https://docs.ray.io/en/latest/tune.html) - Python library for experiment execution and hyperparameter tuning at any scale.

## Knowledge Sharing

*Tools for sharing knowledge to the entire team/company.*

* [Knowledge Repo](https://github.com/airbnb/knowledge-repo) - Knowledge sharing platform for data scientists and other technical professions.
* [Kyso](https://kyso.io/) - One place for data insights so your entire team can learn from your data.

## Machine Learning Platform

*Complete machine learning platform solutions.*

* [aiWARE](https://www.veritone.com/aiware/aiware-os/) - aiWARE helps MLOps teams evaluate, deploy, integrate, scale & monitor ML models.
* [Algorithmia](https://algorithmia.com/) - Securely govern your machine learning operations with a healthy ML lifecycle.
* [Allegro AI](https://allegro.ai/) - Transform ML/DL research into products. Faster.
* [Bodywork](https://bodywork.readthedocs.io/en/latest/) - Deploys machine learning projects developed in Python, to Kubernetes.
* [CNVRG](https://cnvrg.io/) - An end-to-end machine learning platform to build and deploy AI models at scale.
* [DAGsHub](https://dagshub.com/) - A platform built on open source tools for data, model and pipeline management.
* [Dataiku](https://www.dataiku.com/) - Platform democratizing access to data and enabling enterprises to build their own path to AI.
* [DataRobot](https://www.datarobot.com/) - AI platform that democratizes data science and automates the end-to-end ML at scale.
* [Domino](https://www.dominodatalab.com/) - One place for your data science tools, apps, results, models, and knowledge.
* [Edge Impulse](https://edgeimpulse.com/) - Platform for creating, optimizing, and deploying AI/ML algorithms for edge devices.
* [envd](https://github.com/tensorchord/envd) - Machine learning development environment for data science and AI/ML engineering teams.
* [FedML](https://fedml.ai/) - Simplifies the workflow of federated learning anywhere at any scale.
* [Gradient](https://gradient.paperspace.com/) - Multicloud CI/CD and MLOps platform for machine learning teams.
* [H2O](https://www.h2o.ai/) - Open source leader in AI with a mission to democratize AI for everyone.
* [Hopsworks](https://www.hopsworks.ai/) - Open-source platform for developing and operating machine learning models at scale.
* [Iguazio](https://www.iguazio.com/) - Data science platform that automates MLOps with end-to-end machine learning pipelines.
* [Katonic](https://katonic.ai/) - Automate your cycle of intelligence with Katonic MLOps Platform.
* [Knime](https://www.knime.com/) - Create and productionize data science using one easy and intuitive environment.
* [Kubeflow](https://www.kubeflow.org/) - Making deployments of ML workflows on Kubernetes simple, portable and scalable.
* [LynxKite](https://lynxkite.com/) - A complete graph data science platform for very large graphs and other datasets.
* [ML Workspace](https://github.com/ml-tooling/ml-workspace) - All-in-one web-based IDE specialized for machine learning and data science.
* [MLReef](https://github.com/MLReef/mlreef) - Open source MLOps platform that helps you collaborate, reproduce and share your ML work.
* [Modzy](https://www.modzy.com/) - AI platform and marketplace offering scalable, secure, and ready-to-deploy AI models.
* [Neu.ro](https://neu.ro) - MLOps platform that integrates open-source and proprietary tools into client-oriented systems.
* [Omnimizer](https://www.omniml.ai) - Simplifies and accelerates MLOps by bridging the gap between ML models and edge hardware.
* [Pachyderm](https://www.pachyderm.com/) - Combines data lineage with end-to-end pipelines on Kubernetes, engineered for the enterprise.
* [Polyaxon](https://www.github.com/polyaxon/polyaxon/) - A platform for reproducible and scalable machine learning and deep learning on kubernetes.
* [Sagemaker](https://aws.amazon.com/sagemaker/) - Fully managed service that provides the ability to build, train, and deploy ML models quickly.
* [Sematic](https://sematic.dev) - An open-source end-to-end pipelining tool to go from laptop prototype to cloud in no time.
* [SigOpt](https://sigopt.com/) - A platform that makes it easy to track runs, visualize training, and scale hyperparameter tuning.
* [Valohai](https://valohai.com/) - Takes you from POC to production while managing the whole model lifecycle.

## Model Fairness and Privacy

*Tools for performing model fairness and privacy in production.*

* [AIF360](https://github.com/Trusted-AI/AIF360) - A comprehensive set of fairness metrics for datasets and machine learning models.
* [Fairlearn](https://github.com/fairlearn/fairlearn) - A Python package to assess and improve fairness of machine learning models.
* [Opacus](https://github.com/pytorch/opacus) - A library that enables training PyTorch models with differential privacy.
* [TensorFlow Privacy](https://github.com/tensorflow/privacy) - Library for training machine learning models with privacy for training data.

## Model Interpretability

*Tools for performing model interpretability/explainability.*

* [Alibi](https://github.com/SeldonIO/alibi) - Open-source Python library enabling ML model inspection and interpretation.
* [Captum](https://github.com/pytorch/captum) - Model interpretability and understanding library for PyTorch.
* [ELI5](https://github.com/eli5-org/eli5) - Python package which helps to debug machine learning classifiers and explain their predictions.
* [InterpretML](https://github.com/interpretml/interpret) - A toolkit to help understand models and enable responsible machine learning.
* [LIME](https://github.com/marcotcr/lime) - Explaining the predictions of any machine learning classifier.
* [Lucid](https://github.com/tensorflow/lucid) - Collection of infrastructure and tools for research in neural network interpretability.
* [SAGE](https://github.com/iancovert/sage) - For calculating global feature importance using Shapley values.
* [SHAP](https://github.com/slundberg/shap) - A game theoretic approach to explain the output of any machine learning model.
* [Skater](https://github.com/oracle/Skater) - Unified framework to enable Model Interpretation for all forms of model.

## Model Lifecycle

*Tools for managing model lifecycle (tracking experiments, parameters and metrics).*

* [Aim](https://github.com/aimhubio/aim) - A super-easy way to record, search and compare 1000s of ML training runs.
* [Cascade](https://github.com/Oxid15/cascade) - Library of ML-Engineering tools for rapid prototyping and experiment management.
* [Comet](https://github.com/comet-ml) - Track your datasets, code changes, experimentation history, and models.
* [Guild AI](https://guild.ai/) - Open source experiment tracking, pipeline automation, and hyperparameter tuning.
* [Keepsake](https://github.com/replicate/keepsake) - Version control for machine learning with support to Amazon S3 and Google Cloud Storage.
* [Losswise](https://losswise.com) - Makes it easy to track the progress of a machine learning project.
* [Mlflow](https://mlflow.org/) - Open source platform for the machine learning lifecycle.
* [ModelDB](https://github.com/VertaAI/modeldb/) - Open source ML model versioning, metadata, and experiment management.
* [Neptune AI](https://neptune.ai/) - The most lightweight experiment management tool that fits any workflow.
* [Replicate](https://github.com/replicate/replicate) - Library that uploads files and metadata (like hyperparameters) to S3 or GCS.
* [Sacred](https://github.com/IDSIA/sacred) - A tool to help you configure, organize, log and reproduce experiments.
* [Weights and Biases](https://github.com/wandb/client) - A tool for visualizing and tracking your machine learning experiments.

## Model Serving

*Tools for serving models in production.*

* [Banana](https://banana.dev) - Host your ML inference code on serverless GPUs and integrate it into your app with one line of code.
* [BentoML](https://github.com/bentoml/BentoML) - Open-source platform for high-performance ML model serving.
* [BudgetML](https://github.com/ebhy/budgetml) - Deploy a ML inference service on a budget in less than 10 lines of code.
* [Cortex](https://www.cortex.dev/) - Machine learning model serving infrastructure.
* [Gradio](https://github.com/gradio-app/gradio) - Create customizable UI components around your models.
* [GraphPipe](https://oracle.github.io/graphpipe) - Machine learning model deployment made simple.
* [Hydrosphere](https://github.com/Hydrospheredata/hydro-serving) - Platform for deploying your Machine Learning to production.
* [KFServing](https://github.com/kubeflow/kfserving) - Kubernetes custom resource definition for serving ML models on arbitrary frameworks.
* [Merlin](https://github.com/gojek/merlin) - A platform for deploying and serving machine learning models.
* [MLEM](https://github.com/iterative/mlem) - Version and deploy your ML models following GitOps principles.
* [Opyrator](https://github.com/ml-tooling/opyrator) - Turns your ML code into microservices with web API, interactive GUI, and more.
* [PredictionIO](https://github.com/apache/predictionio) - Event collection, deployment of algorithms, evaluation, querying predictive results via APIs.
* [Rune](https://github.com/hotg-ai/rune) - Provides containers to encapsulate and deploy EdgeML pipelines and applications.
* [Seldon](https://www.seldon.io/) - Take your ML projects from POC to production with maximum efficiency and minimal risk.
* [Streamlit](https://github.com/streamlit/streamlit) - Lets you create apps for your ML projects with deceptively simple Python scripts.
* [TensorFlow Serving](https://www.tensorflow.org/tfx/guide/serving) - Flexible, high-performance serving system for ML models, designed for production.
* [TorchServe](https://github.com/pytorch/serve) - A flexible and easy to use tool for serving PyTorch models.
* [Triton Inference Server](https://github.com/triton-inference-server/server) - Provides an optimized cloud and edge inferencing solution.
* [Vespa](https://github.com/vespa-engine/vespa) - Store, search, organize and make machine-learned inferences over big data at serving time.

## Model Testing & Validation

*Tools for testing and validating models.*

* [Deepchecks](https://github.com/deepchecks/deepchecks) - Open-source package for validating ML models & data, with various checks and suites.
* [Trubrics](https://github.com/trubrics/trubrics-sdk) - Validate machine learning with data science and domain expert feedback.

## Optimization Tools

*Optimization tools related to model scalability in production.*

* [Accelerate](https://github.com/huggingface/accelerate) - A simple way to train and use PyTorch models with multi-GPU, TPU, mixed-precision.
* [Dask](https://dask.org/) - Provides advanced parallelism for analytics, enabling performance at scale for the tools you love.
* [DeepSpeed](https://github.com/microsoft/DeepSpeed) - Deep learning optimization library that makes distributed training easy, efficient, and effective.
* [Fiber](https://uber.github.io/fiber/) - Python distributed computing library for modern computer clusters.
* [Horovod](https://github.com/horovod/horovod) - Distributed deep learning training framework for TensorFlow, Keras, PyTorch, and Apache MXNet.
* [Mahout](https://mahout.apache.org/) - Distributed linear algebra framework and mathematically expressive Scala DSL.
* [MLlib](https://spark.apache.org/mllib/) - Apache Spark's scalable machine learning library.
* [Modin](https://github.com/modin-project/modin) - Speed up your Pandas workflows by changing a single line of code.
* [Nebulgym](https://github.com/nebuly-ai/nebulgym) - Easy-to-use library to accelerate AI training.
* [Nebullvm](https://github.com/nebuly-ai/nebullvm) - Easy-to-use library to boost AI inference.
* [Petastorm](https://github.com/uber/petastorm) - Enables single machine or distributed training and evaluation of deep learning models.
* [Rapids](https://rapids.ai/index.html) - Gives the ability to execute end-to-end data science and analytics pipelines entirely on GPUs.
* [Ray](https://github.com/ray-project/ray) - Fast and simple framework for building and running distributed applications.
* [Singa](http://singa.apache.org/en/index.html) - Apache top level project, focusing on distributed training of DL and ML models.
* [Tpot](https://github.com/EpistasisLab/tpot) - Automated ML tool that optimizes machine learning pipelines using genetic programming.

## Simplification Tools

*Tools related to machine learning simplification and standardization.*

* [Hermione](https://github.com/a3data/hermione) - Help Data Scientists on setting up more organized codes, in a quicker and simpler way.
* [Hydra](https://github.com/facebookresearch/hydra) - A framework for elegantly configuring complex applications.
* [Koalas](https://github.com/databricks/koalas) - Pandas API on Apache Spark. Makes data scientists more productive when interacting with big data.
* [Ludwig](https://github.com/uber/ludwig) - Allows users to train and test deep learning models without the need to write code.
* [MLNotify](https://github.com/aporia-ai/mlnotify) - No need to keep checking your training, just one import line and you'll know the second it's done.
* [PyCaret](https://pycaret.org/) - Open source, low-code machine learning library in Python.
* [Sagify](https://github.com/Kenza-AI/sagify) - A CLI utility to train and deploy ML/DL models on AWS SageMaker.
* [Soopervisor](https://github.com/ploomber/soopervisor) - Export ML projects to Kubernetes (Argo workflows), Airflow, AWS Batch, and SLURM.
* [Soorgeon](https://github.com/ploomber/soorgeon) - Convert monolithic Jupyter notebooks into maintainable pipelines.
* [TrainGenerator](https://github.com/jrieke/traingenerator) - A web app to generate template code for machine learning.
* [Turi Create](https://github.com/apple/turicreate) - Simplifies the development of custom machine learning models.

## Visual Analysis and Debugging

*Tools for performing visual analysis and debugging of ML/DL models.*

* [Aporia](https://www.aporia.com/) - Observability with customized monitoring and explainability for ML models.
* [Arize](https://www.arize.com/) - A free end-to-end ML observability and model monitoring platform.
* [Evidently](https://github.com/evidentlyai/evidently) - Interactive reports to analyze ML models during validation or production monitoring.
* [Fiddler](https://www.fiddler.ai/) - Monitor, explain, and analyze your AI in production.
* [Manifold](https://github.com/uber/manifold) - A model-agnostic visual debugging tool for machine learning.
* [NannyML](https://github.com/NannyML/nannyml) - Algorithm capable of fully capturing the impact of data drift on performance.
* [Netron](https://github.com/lutzroeder/netron) - Visualizer for neural network, deep learning, and machine learning models.
* [Superwise](https://www.superwise.ai) - Fully automated, enterprise-grade model observability in a self-service SaaS platform.
* [Whylogs](https://github.com/whylabs/whylogs) - The open source standard for data logging. Enables ML monitoring and observability.
* [Yellowbrick](https://github.com/DistrictDataLabs/yellowbrick) - Visual analysis and diagnostic tools to facilitate machine learning model selection.

## Workflow Tools

*Tools and frameworks to create workflows or pipelines in the machine learning context.*

* [Argo](https://github.com/argoproj/argo) - Open source container-native workflow engine for orchestrating parallel jobs on Kubernetes.
* [Automate Studio](https://www.veritone.com/applications/automate-studio/) - Rapidly build & deploy AI-powered workflows.
* [Couler](https://github.com/couler-proj/couler) - Unified interface for constructing and managing workflows on different workflow engines.
* [dstack](https://github.com/dstackai/dstack) - An open-core tool to automate data and training workflows.
* [Flyte](https://flyte.org/) - Easy to create concurrent, scalable, and maintainable workflows for machine learning.
* [Kale](https://github.com/kubeflow-kale/kale) - Aims at simplifying the Data Science experience of deploying Kubeflow Pipelines workflows.
* [Kedro](https://github.com/quantumblacklabs/kedro) - Library that implements software engineering best-practice for data and ML pipelines.
* [Luigi](https://github.com/spotify/luigi) - Python module that helps you build complex pipelines of batch jobs.
* [Metaflow](https://metaflow.org/) - Human-friendly lib that helps scientists and engineers build and manage data science projects.
* [MLRun](https://github.com/mlrun/mlrun) - Generic mechanism for data scientists to build, run, and monitor ML tasks and pipelines.
* [Orchest](https://github.com/orchest/orchest/) - Visual pipeline editor and workflow orchestrator with an easy to use UI and based on Kubernetes.
* [Ploomber](https://github.com/ploomber/ploomber) - Write maintainable, production-ready pipelines. Develop locally, deploy to the cloud.
* [Prefect](https://docs.prefect.io/) - A workflow management system, designed for modern infrastructure.
* [ZenML](https://github.com/maiot-io/zenml) - An extensible open-source MLOps framework to create reproducible pipelines.

---

# Resources

Where to discover new tools and discuss about existing ones.

## Articles

* [A Tour of End-to-End Machine Learning Platforms](https://databaseline.tech/a-tour-of-end-to-end-ml-platforms/) (Databaseline)
* [Continuous Delivery for Machine Learning](https://martinfowler.com/articles/cd4ml.html) (Martin Fowler)
* [Delivering on the Vision of MLOps: A maturity-based approach](https://azure.microsoft.com/mediahandler/files/resourcefiles/gigaom-Delivering-on-the-Vision-of-MLOps/Delivering%20on%20the%20Vision%20of%20MLOps.pdf) (GigaOm)
* [Machine Learning Operations (MLOps): Overview, Definition, and Architecture](https://arxiv.org/abs/2205.02302) (arXiv)
* [MLOps: Continuous delivery and automation pipelines in machine learning](https://cloud.google.com/solutions/machine-learning/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) (Google)
* [MLOps: Machine Learning as an Engineering Discipline](https://towardsdatascience.com/ml-ops-machine-learning-as-an-engineering-discipline-b86ca4874a3f) (Medium)
* [Rules of Machine Learning: Best Practices for ML Engineering](https://developers.google.com/machine-learning/guides/rules-of-ml) (Google)
* [The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/aad9f93b86b7addfea4c419b9100c6cdd26cacea.pdf) (Google)
* [What Is MLOps?](https://blogs.nvidia.com/blog/2020/09/03/what-is-mlops/) (NVIDIA)

## Books

* [Beginning MLOps with MLFlow](https://www.amazon.com/Beginning-MLOps-MLFlow-SageMaker-Microsoft/dp/1484265483) (Apress)
* [Building Machine Learning Pipelines](https://www.oreilly.com/library/view/building-machine-learning/9781492053187) (O'Reilly)
* [Building Machine Learning Powered Applications](https://www.oreilly.com/library/view/building-machine-learning/9781492045106) (O'Reilly)
* [Deep Learning in Production](https://www.amazon.com/gp/product/6180033773) (AI Summer)
* [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956) (O'Reilly)
* [Engineering MLOps](https://www.packtpub.com/product/engineering-mlops/9781800562882) (Packt)
* [Introducing MLOps](https://www.oreilly.com/library/view/introducing-mlops/9781492083283) (O'Reilly)
* [Kubeflow for Machine Learning](https://www.oreilly.com/library/view/kubeflow-for-machine/9781492050117) (O'Reilly)
* [Kubeflow Operations Guide](https://www.oreilly.com/library/view/kubeflow-operations-guide/9781492053262) (O'Reilly)
* [Machine Learning Design Patterns](https://www.oreilly.com/library/view/machine-learning-design/9781098115777) (O'Reilly)
* [Machine Learning Engineering in Action](https://www.manning.com/books/machine-learning-engineering-in-action) (Manning)
* [ML Ops: Operationalizing Data Science](https://www.oreilly.com/library/view/ml-ops-operationalizing/9781492074663) (O'Reilly)
* [MLOps Engineering at Scale](https://www.manning.com/books/mlops-engineering-at-scale) (Manning)
* [Practical Deep Learning at Scale with MLflow](https://www.packtpub.com/product/practical-deep-learning-at-scale-with-mlflow/9781803241333) (Packt)
* [Practical MLOps](https://www.oreilly.com/library/view/practical-mlops/9781098103002) (O'Reilly)
* [Production-Ready Applied Deep Learning](https://www.packtpub.com/product/production-ready-applied-deep-learning/9781803243665) (Packt)
* [Reliable Machine Learning](https://www.oreilly.com/library/view/reliable-machine-learning/9781098106218) (O'Reilly)
* [The Machine Learning Solutions Architect Handbook](https://www.packtpub.com/product/the-machine-learning-solutions-architect-handbook/9781801072168) (Packt)

## Events

* [apply() - The ML data engineering conference](https://www.applyconf.com/)
* [MLOps Conference - Keynotes and Panels](https://www.youtube.com/playlist?list=PLH8M0UOY0uy6d_n3vEQe6J_gRBUrISF9m)
* [MLOps World: Machine Learning in Production Conference](https://mlopsworld.com/)
* [Stanford MLSys Seminar Series](https://mlsys.stanford.edu/)

## Other Lists

* [Applied ML](https://github.com/eugeneyan/applied-ml)
* [Awesome AutoML Papers](https://github.com/hibayesian/awesome-automl-papers)
* [Awesome AutoML](https://github.com/windmaple/awesome-AutoML)
* [Awesome Data Science](https://github.com/academic/awesome-datascience)
* [Awesome DataOps](https://github.com/kelvins/awesome-dataops)
* [Awesome Deep Learning](https://github.com/ChristosChristofidis/awesome-deep-learning)
* [Awesome Game Datasets](https://github.com/leomaurodesenv/game-datasets) (includes AI content)
* [Awesome Machine Learning](https://github.com/josephmisiti/awesome-machine-learning)
* [Awesome MLOps](https://github.com/visenger/awesome-mlops)
* [Awesome Production Machine Learning](https://github.com/EthicalML/awesome-production-machine-learning)
* [Awesome Python](https://github.com/vinta/awesome-python)
* [Deep Learning in Production](https://github.com/ahkarami/Deep-Learning-in-Production)

## Podcasts

* [How AI Built This](https://how-ai-built-this.captivate.fm/)
* [Kubernetes Podcast from Google](https://kubernetespodcast.com/)
* [Machine Learning – Software Engineering Daily](https://podcasts.google.com/?feed=aHR0cHM6Ly9zb2Z0d2FyZWVuZ2luZWVyaW5nZGFpbHkuY29tL2NhdGVnb3J5L21hY2hpbmUtbGVhcm5pbmcvZmVlZC8)
* [MLOps.community](https://podcasts.google.com/?feed=aHR0cHM6Ly9hbmNob3IuZm0vcy8xNzRjYjFiOC9wb2RjYXN0L3Jzcw)
* [Pipeline Conversation](https://podcast.zenml.io/)
* [Practical AI: Machine Learning, Data Science](https://changelog.com/practicalai)
* [This Week in Machine Learning & AI](https://twimlai.com/)

## Slack

* [Kubeflow Workspace](https://kubeflow.slack.com/#/)
* [MLOps Community Wokspace](https://mlops-community.slack.com)

## Websites

* [Feature Stores for ML](http://featurestore.org/)
* [Made with ML](https://madewithml.com/)
* [ML-Ops](https://ml-ops.org/)
* [MLOps Community](https://mlops.community/)
* [MLOps Guide](https://mlops-guide.github.io/)

# Contributing

All contributions are welcome! Please take a look at the [contribution guidelines](https://github.com/kelvins/awesome-mlops/blob/main/CONTRIBUTING.md) first.
