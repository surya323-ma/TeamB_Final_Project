# BotTrainer NLU — Model Trainer & Evaluator

Python + Streamlit app for training, evaluating and testing NLU intent classifiers.

## Setup

    pip install -r requirements.txt
    streamlit run app.py

## Pages

    1_Dataset_Manager.py    Upload/explore/annotate training data
    2_Trainer.py            Train NLU classifier with configurable hyperparams
    3_Evaluator.py          Metrics, confusion matrix, CV, confidence analysis
    4_Live_Testing.py       Real-time predictions + entity extraction
    5_Compare_Algorithms.py Benchmark all 5 algorithms side-by-side
    6_Model_Registry.py     Save, load, compare and export trained models

## Algorithms

    Logistic Regression | Linear SVM | Naive Bayes | Random Forest | Gradient Boosting

## Production Usage

    import joblib
    from utils.nlu_engine import predict

    m = joblib.load("model.joblib")
    result = predict(m["pipeline"], m["label_encoder"], "book a flight")
    print(result["intent"], result["conf"], result["entities"])
