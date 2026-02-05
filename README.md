# Twitter Sentiment Analysis for Brand Monitoring

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.6+-green.svg)](https://www.nltk.org/)

*Automated Real-Time Sentiment Classification System for Social Media Brand Monitoring*

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Features](#features)
- [Model Performance](#model-performance)
- [Results & Insights](#results--insights)
- [Recommendations](#recommendations)
- [Future Work](#future-work)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Contributors](#contributors)

---

## Overview

This project develops an **automated sentiment analysis system** that processes Twitter data in real-time to classify public sentiment toward major technology brands (Apple and Google). Built using **Natural Language Processing (NLP)** and **Machine Learning**, the system enables companies to monitor brand health, detect emerging crises, and prioritize customer service responses efficiently.

The solution transforms manual, time-intensive social media monitoring into an automated, scalable process capable of analyzing thousands of tweets daily with **68.5% accuracy** and robust negative sentiment detection.

---

## Business Problem

### The Challenge

In today's digital marketplace, social media platforms like Twitter serve as primary channels where customers express opinions about products and brands. However, companies face critical challenges:

- **Volume Overload**: Thousands of brand mentions daily make manual analysis impossible
- **Response Delays**: Traditional monitoring methods identify issues only after brand damage occurs
- **Resource Constraints**: Customer service teams are overwhelmed by the sheer volume of feedback
- **Prioritization Issues**: Difficulty distinguishing urgent complaints from general chatter

### Our Solution

We developed an **intelligent classification system** that automatically categorizes tweets into three sentiment classes:
- ✅ **Positive** - Enthusiastic endorsements and satisfaction
- ⚠️ **Negative** - Complaints and dissatisfaction requiring immediate attention
- ⚪ **Neutral** - Informational mentions without clear emotion

### Business Value

1. **Real-Time Crisis Detection** - Identify emerging negative sentiment before it escalates
2. **Automated Monitoring** - Scale from analyzing hundreds to thousands of tweets daily
3. **Smart Prioritization** - Route critical complaints directly to specialized response teams
4. **Data-Driven Insights** - Measure campaign effectiveness and product reception quantitatively

---

## Dataset

### Source & Context
- **Origin**: Tweets collected during the **South by Southwest (SXSW)** conference
- **Size**: 9,093 tweets (after cleaning: 9,070)
- **Target Brands**: Apple and Google products (iPhone, iPad, Android, Google services)
- **Labeling**: Human-annotated sentiment classifications
- **Format**: CSV with tweet text, product mentions, and sentiment labels

### Class Distribution

| Sentiment Class | Count | Percentage |
|----------------|-------|------------|
| **Neutral** | 5,531 | 60.98% |
| **Positive** | 2,970 | 32.75% |
| **Negative** | 569 | 6.27% |

**Challenge**: Significant class imbalance requires specialized handling to prevent bias toward majority class.

### Data Characteristics
- **Language**: English with social media slang, abbreviations, emojis
- **Noise**: Contains URLs, user mentions (@), hashtags (#), typos, informal grammar
- **Real-World Complexity**: Sarcasm, context-dependent sentiment, mixed emotions

---

## Methodology

### 1. Data Understanding & Exploration
- **Initial Analysis**: Examined dataset structure, missing values, and duplicates
- **Sentiment Distribution**: Analyzed class imbalance patterns
- **Product Mention Analysis**: Identified most discussed products (iPad, Apple, Google)
- **Tweet Characteristics**: Studied length patterns and word count distributions

### 2. Data Preparation & Cleaning
```python
# Key cleaning steps implemented:
✓ Removed 1 missing tweet
✓ Filled 5,802 missing product mentions with 'Unknown'
✓ Eliminated 22 duplicate entries
✓ Simplified 4-class to 3-class sentiment problem
```

### 3. Advanced Text Preprocessing (NLTK)

Our preprocessing pipeline includes:

#### Basic Cleaning
- **URL Removal**: Stripped http/https links
- **Mention Removal**: Removed @username references
- **Hashtag Processing**: Kept hashtag text, removed # symbol
- **Special Characters**: Eliminated punctuation and numbers
- **Case Normalization**: Converted to lowercase

#### Advanced NLP Processing
- **Tokenization**: Split text into individual words using NLTK
- **Lemmatization**: Reduced words to root form (e.g., "running" → "run")
- **Stopword Removal**: Eliminated common words while preserving sentiment-bearing negations
- **Custom Retention**: Kept important sentiment words like "not", "no", "but", "against"

### 4. Feature Engineering

#### TF-IDF Vectorization
- **Max Features**: 7,000 most important terms
- **N-gram Range**: Unigrams + Bigrams (1-2 words) to capture phrases like "not good"
- **Min Document Frequency**: 2 (eliminates extremely rare words)
- **Sublinear TF Scaling**: Logarithmic term frequency for better weight distribution

#### N-gram Analysis
Analyzed bigrams and trigrams to identify sentiment-bearing phrases:
- **Positive**: "apple store", "sxsw link", "come see"
- **Negative**: "design headache", "google circle", "ipad design"
- **Neutral**: "social network", "new social", "network called"

### 5. Model Development & Comparison

We implemented and evaluated **five distinct machine learning models**:

#### Model 1: Baseline Naive Bayes
- **Purpose**: Fast baseline with interpretable probability estimates
- **Configuration**: MultinomialNB with unigram TF-IDF (max_features=5000)
- **Result**: 65.49% accuracy, struggled with negative class (0.88% recall)

#### Model 2: Enhanced Naive Bayes
- **Improvement**: Added bigrams, increased vocabulary size to 7,000
- **Configuration**: Reduced alpha smoothing (0.1), sublinear TF scaling
- **Result**: 67.48% accuracy, improved negative recall to 22.81%

#### Model 3: Logistic Regression
- **Strength**: Handles feature dependence better than Naive Bayes
- **Configuration**: Balanced class weights, L2 regularization (C=1.0)
- **Result**: 64.55% accuracy, **best negative recall (56.14%)**

#### Model 4: Linear SVM ⭐ **SELECTED MODEL**
- **Strength**: Optimal decision boundaries in high-dimensional text space
- **Configuration**: Balanced class weights, C=0.5 regularization
- **Result**: **68.52% accuracy**, best overall F1-score, robust negative detection

#### Model 5: XGBoost
- **Strength**: Ensemble boosting with tree-based learning
- **Configuration**: 100 estimators, learning_rate=0.1, max_depth=6
- **Result**: 67.48% accuracy, high precision but low negative recall (7.02%)

### 6. Model Evaluation

#### Validation Strategy
- **Train-Test Split**: 80% training (7,255 samples) / 20% testing (1,814 samples)
- **Stratification**: Preserved original sentiment distribution in both sets
- **Cross-Validation**: 5-fold CV on top models for robustness assessment

#### Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: How many predicted positives were actually positive
- **Recall**: How many actual positives were correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed class-wise error analysis

---

## ✨ Features

### Core Capabilities
✅ **Automated Sentiment Classification**: Three-class prediction (Positive/Negative/Neutral)  
✅ **Real-Time Processing**: Fast inference suitable for streaming Twitter data  
✅ **Robust Text Preprocessing**: Handles social media noise (slang, typos, emojis)  
✅ **Class Imbalance Handling**: Balanced weighting ensures minority class detection  
✅ **Scalable Architecture**: Pipeline design ready for production deployment  

### Advanced NLP Features
🔤 **Lemmatization**: Root word extraction using WordNet  
📝 **Bigram Support**: Captures multi-word sentiment expressions  
🎯 **TF-IDF Weighting**: Emphasizes distinctive words over common terms  
🚫 **Smart Stopword Removal**: Preserves negations critical for sentiment  

### Model Interpretability
📊 **Feature Importance**: Identify most influential words per sentiment  
📈 **Confusion Matrix Analysis**: Understand model error patterns  
🎯 **Per-Class Metrics**: Separate performance tracking for each sentiment  

---

## 📊 Main Notebook

```
jupyter notebook Main.ipynb
```

The notebook contains:
- 📊 Comprehensive exploratory data analysis
- 🧹 Step-by-step preprocessing pipeline
- 🤖 Model training and comparison
- 📈 Detailed performance visualizations
- 💡 Business insights and recommendations

---

## 📈 Model Performance

### Final Model: Linear SVM

<table>
<tr>
<th>Metric</th>
<th>Value</th>
<th>Business Interpretation</th>
</tr>
<tr>
<td><b>Overall Accuracy</b></td>
<td>68.52%</td>
<td>Correct classification rate across all sentiments</td>
</tr>
<tr>
<td><b>Macro F1-Score</b></td>
<td>0.5971</td>
<td>Balanced performance across all classes</td>
</tr>
<tr>
<td><b>Negative Recall</b></td>
<td>48.25%</td>
<td>Catches nearly half of all customer complaints</td>
</tr>
<tr>
<td><b>Negative F1-Score</b></td>
<td>0.4435</td>
<td>Best balance for crisis detection</td>
</tr>
</table>

### Detailed Classification Report

```
                  precision    recall  f1-score   support

    Negative       0.41      0.48      0.44       114
     Neutral       0.76      0.77      0.77      1106
    Positive       0.60      0.57      0.58       594

    accuracy                           0.69      1814
   macro avg       0.59      0.61      0.60      1814
weighted avg       0.69      0.69      0.69      1814
```

### Model Comparison Matrix

| Model | Accuracy | Macro F1 | Negative Recall | Negative F1 | Best For |
|-------|----------|----------|-----------------|-------------|----------|
| **Linear SVM** ⭐ | **68.52%** | **0.5971** | 48.25% | **0.4435** | **Production** |
| Logistic Regression | 64.55% | 0.5649 | **56.14%** | 0.3879 | Crisis Detection |
| Enhanced Naive Bayes | 67.48% | 0.5406 | 22.81% | 0.3059 | Fast Baseline |
| XGBoost | 67.48% | 0.4421 | 7.02% | 0.1270 | High Precision |
| Baseline Naive Bayes | 65.49% | 0.3921 | 0.88% | 0.0172 | Benchmark |

### Cross-Validation Results

```
5-Fold Cross-Validation (F1-Macro Scores):

Linear SVM:
  Fold Scores: [0.5590, 0.5338, 0.5704, 0.5488, 0.5592]
  Mean F1: 0.5542 (±0.0246)
  
Logistic Regression:
  Fold Scores: [0.5404, 0.5341, 0.5533, 0.5483, 0.5427]
  Mean F1: 0.5437 (±0.0132)
```

**Interpretation**: Low variance confirms stable generalization to unseen data.

---

## 🎯 Results & Insights

### Key Findings

#### 1. **The Accuracy Paradox**
While XGBoost shows comparable accuracy (67.48%) to Linear SVM (68.52%), it misses **93% of negative sentiment** (7% recall). This demonstrates why accuracy alone is misleading for imbalanced datasets. **Linear SVM achieves the best business balance** by maintaining high overall performance while detecting nearly half of all complaints.

#### 2. **Sentiment Distribution Patterns**

**Product-Specific Insights**:
- **iPad**: Most discussed product (945 mentions), predominantly positive
- **Apple**: Strong brand loyalty with 659 mentions, mostly neutral-to-positive
- **Google**: 428 mentions with balanced sentiment distribution
- **iPhone**: 296 mentions, more polarized (higher negative proportion)

**Temporal Context**: SXSW conference setting created:
- High volume of promotional/informational tweets (explaining 61% neutral class)
- Enthusiastic early adopter sentiment (33% positive)
- Lower complaint rate typical of tech-savvy, brand-loyal audience (6% negative)

#### 3. **Linguistic Patterns Discovered**

**Positive Sentiment Indicators**:
- Bigrams: "apple store", "sxsw link", "awesome app"
- Action verbs: "love", "excited", "amazing", "check out"
- Product launch enthusiasm

**Negative Sentiment Indicators**:
- Bigrams: "design headache", "battery dead", "crashy app"
- Complaint markers: "hope", "issue", "problem", "not working"
- Longer tweet length (average 109 chars vs 104 for neutral)

**Neutral Sentiment Indicators**:
- Informational: "new social network", "google launch", "major new"
- Product names without emotion words
- Shortest average tweet length

#### 4. **Model Trade-offs Analysis**

| Dimension | Logistic Regression | Linear SVM (SELECTED) |
|-----------|--------------------|-----------------------|
| **Strength** | Highest negative recall (56%) | Best overall balance |
| **Weakness** | Lower overall accuracy | Moderate negative recall |
| **Use Case** | Maximize complaint detection | Production deployment |
| **Cost** | More false alarms | Fewer false alarms |

**Business Decision**: We selected **Linear SVM** because:
1. ✅ Highest overall accuracy (68.52%) ensures reliable predictions
2. ✅ Strong negative F1-score (0.44) balances recall and precision
3. ✅ Cross-validation confirmed stability (low variance)
4. ✅ Fewer false positives reduces customer service workload

---

## 💡 Recommendations

### 1. **Immediate Deployment Actions**

#### A. Operationalize Crisis Alerts
```python
# Pseudo-code for production implementation
if predicted_sentiment == 'Negative' and confidence > 0.7:
    send_alert_to_customer_service_team()
    flag_for_priority_response()
    track_sentiment_spike()
```

**Implementation**:
- Integrate model with Twitter API for real-time streaming
- Set confidence thresholds (recommend 0.7 for high-priority alerts)
- Route negative tweets to specialized response dashboard
- Implement 15-minute response SLA for flagged complaints

#### B. Customer Service Integration
- **Smart Routing**: Automatically assign negative tweets to appropriate product teams
- **Priority Queue**: Surface high-confidence complaints first
- **Context Provision**: Include product mention, tweet length, and confidence score
- **Response Templates**: Pre-populate empathetic responses for common issues

#### C. Sentiment Dashboard
Build executive dashboard showing:
- **Real-Time Sentiment Pulse**: Live gauge showing sentiment distribution
- **Product Breakdown**: Sentiment by product (iPad, iPhone, Google, etc.)
- **Trend Analysis**: Sentiment shifts over time (hourly/daily)
- **Top Issues**: Most common bigrams in negative tweets
- **Competitive Comparison**: Apple vs. Google sentiment head-to-head

### 2. **Model Improvement Roadmap**

#### Phase 1: Short-Term (1-3 months)
✅ **Feedback Loop**: 
- Have customer service reps label misclassified tweets
- Retrain model monthly with corrected labels
- Track performance improvement over time

✅ **Threshold Tuning**:
- Conduct A/B test with confidence thresholds (0.6 vs 0.7 vs 0.8)
- Measure impact on customer service workload vs. complaint capture rate

✅ **Error Analysis**:
- Manually review 100 misclassified tweets per class
- Identify systematic failure patterns (e.g., sarcasm, product comparisons)
- Add targeted features to address gaps

#### Phase 2: Medium-Term (3-6 months)
🚀 **Emoji Sentiment Analysis**:
- Extract emojis as separate features (😊, 😡, 🔥)
- Map emojis to sentiment scores using emoji sentiment lexicon
- Add emoji count and type as metadata features

🚀 **Aspect-Based Sentiment**:
- Identify *what* users discuss (battery, screen, price, design)
- Classify sentiment per aspect, not just overall tweet
- Example: "Great screen [Positive], terrible battery [Negative]"

🚀 **Class Imbalance Mitigation**:
- Implement SMOTE (Synthetic Minority Over-sampling Technique)
- Collect more negative examples through targeted scraping
- Use cost-sensitive learning with asymmetric penalties

#### Phase 3: Long-Term (6-12 months)
🔬 **Deep Learning Transition**:
- Experiment with **BERT** (Bidirectional Encoder Representations from Transformers)
- Fine-tune pre-trained language models on Twitter data
- Expected improvement: 5-10% accuracy gain, better sarcasm detection

🔬 **Multi-Modal Analysis**:
- Process images and videos in tweets (requires computer vision)
- Combine text and visual sentiment signals
- Detect sentiment from facial expressions in embedded videos

🔬 **Conversation Context**:
- Analyze tweet threads (replies and quoted tweets)
- Capture sentiment evolution over conversation
- Detect escalating complaints requiring intervention

### 3. **Business Process Recommendations**

#### A. Crisis Management Protocol
```
1. Model detects 20% spike in negative sentiment → Alert Level Yellow
2. Human review confirms emerging issue → Alert Level Orange
3. PR team drafts holding statement → Prepare for Level Red
4. Public acknowledgment if trend continues → Alert Level Red
```

#### B. Product Launch Monitoring
- **Pre-Launch**: Establish sentiment baseline 2 weeks before
- **Launch Day**: Monitor in real-time with 5-minute refresh
- **Post-Launch**: Track sentiment decay curve over 30 days
- **Insights Report**: Deliver executive summary within 48 hours

#### C. Competitive Intelligence
- Extend model to competitors (Samsung, Microsoft, Amazon)
- Generate weekly sentiment comparison reports
- Identify competitor weaknesses to exploit in marketing
- Track sentiment during competitor product launches

---

## 🔮 Future Work

### Technical Enhancements

#### 1. **Sarcasm Detection**
**Challenge**: "Oh great, my iPhone crashed again. Thanks Apple 🙄" is labeled negative but contains positive words.

**Solution Approaches**:
- Implement contrastive learning to detect sentiment reversal
- Use user history to identify sarcastic patterns
- Integrate emoji-text mismatch detection

#### 2. **Multilingual Support**
**Current Limitation**: English-only model

**Expansion Plan**:
- Spanish (LATAM markets)
- Mandarin (China/Taiwan)
- Japanese (tech-savvy market)
- Use multilingual BERT (mBERT) for transfer learning

#### 3. **Real-Time Streaming Pipeline**
**Architecture**:
```
Twitter API → Apache Kafka → Preprocessing Service → 
Model Inference → PostgreSQL → Dashboard (Grafana)
```

**Infrastructure Requirements**:
- Dockerized microservices
- Kubernetes orchestration
- Auto-scaling based on tweet volume
- 99.9% uptime SLA

#### 4. **Explainable AI (XAI)**
**Goal**: Make model predictions interpretable for non-technical stakeholders

**Methods**:
- **LIME** (Local Interpretable Model-agnostic Explanations)
- **SHAP** (SHapley Additive exPlanations)
- Highlight words contributing most to sentiment prediction
- Generate plain-English explanations: "This tweet is negative because of words: 'crash', 'frustrating', 'terrible'"

### Research Directions

#### 1. **Temporal Sentiment Dynamics**
- Analyze how sentiment evolves during product lifecycle
- Predict sentiment trajectory based on early adopter reactions
- Identify "tipping points" where sentiment shifts rapidly

#### 2. **Influencer Impact Analysis**
- Weight tweets by user follower count and engagement rate
- Model sentiment contagion (how influencer opinions spread)
- Prioritize responses to high-reach negative tweets

#### 3. **Causal Sentiment Attribution**
- Beyond classification: *why* is sentiment negative?
- Extract root causes: "battery", "price", "customer service"
- Enable targeted product improvement roadmaps

---

## 📁 Project Structure

```
twitter-sentiment-analysis/
│
├── data/
│ ├── judge-1377884607_tweet_product_company.csv # Raw dataset
│ └── README.md # Data documentation
│
├── notebooks/
│ ├── Main.ipynb # Complete analysis workflow
│ ├── Steve.ipynb # Individual analysis (Steve)
│ ├── Salma.ipynb # Individual analysis (Salma)
│ └── Grace.ipynb # Individual analysis (Grace)
│
├── src/
│ ├── init.py
│ ├── preprocessing.py # Text cleaning pipeline
│ ├── feature_engineering.py # TF-IDF, n-grams
│ ├── models.py # Model training functions
│ └── evaluation.py # Metrics and visualization
│
├── models/
│ ├── linear_svm_model.pkl # Trained Linear SVM
│ ├── logistic_regression_model.pkl # Alternative model
│ └── tfidf_vectorizer.pkl # Fitted vectorizer
│
├── visualizations/
│ ├── confusion_matrix_svm.png
│ ├── sentiment_distribution.png
│ ├── model_comparison.png
│ └── wordcloud_negative.png
│
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Git ignore rules                                     # Git ignore rules
```

---

## 🛠️ Technologies Used

### Core Machine Learning
- **scikit-learn** 🤖 - Model training and evaluation
- **XGBoost** 🚀 - Gradient boosting implementation
- **NLTK** 📝 - Natural language processing toolkit

### Data Processing
- **pandas** 🐼 - Data manipulation and analysis
- **numpy** 🔢 - Numerical computing

### Visualization
- **matplotlib** 📊 - Static plotting
- **seaborn** 🎨 - Statistical visualization
- **wordcloud** ☁️ - Text visualization

### Development Tools
- **Jupyter** 📓 - Interactive notebooks
- **joblib** 💾 - Model serialization
- **Git** 🌿 - Version control

---

## 👥 Contributors

**Muema Stephen** - musyokas753@gmail.com  
**Salma Mwende** - salma.mwende@student.moringaschool.com  
**Grace Wangui** - gracewangui251@gmail.com

---

## 🌟 Star This Repository!

If you found this project helpful or interesting, please consider giving it a ⭐ on GitHub!

---

<div align="center">

**Built by Group 1**

*Transforming social media noise into actionable business intelligence*

[Back to Top ↑](#twitter-sentiment-analysis-for-brand-monitoring)

</div>
