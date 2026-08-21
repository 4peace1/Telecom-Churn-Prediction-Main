# Telecom Churn Prediction

I built this project for my 3MTT capstone to explore a practical question: can customer information help a telecom company spot people who may be thinking of leaving? The project follows the full process, from exploring the data and training the model to presenting the result in a simple Streamlit app.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Model-F7931E?logo=scikitlearn&logoColor=white)

**Name:** Sunday Babatunde  
**Fellow ID:** FE/26/2850020886  
**3MTT Email:** 4peace1@gmail.com  
**Gender:** Male  
**Cohort:** NextGen Cohort, Lagos

## Why I chose this problem

Losing customers affects revenue, but contacting every customer with a retention offer is not realistic. A churn model can help a retention team decide where to start. My aim was not only to predict churn, but also to show the probability behind each prediction and turn it into a useful next step.

## What I built

- I explored 7,043 customer records to understand the main churn patterns.
- I created one preprocessing pipeline for missing values, scaling and categorical encoding.
- I trained a logistic-regression model because it is suitable for a binary outcome and remains easy to explain.
- I tested the model on data it had not seen during training.
- I built a Streamlit app where a user can enter a customer profile and receive a risk estimate.
- I saved the trained pipeline so the app can load it directly instead of training again on every refresh.

## Validated results

At first, I used the usual 50% cut-off. It produced higher accuracy, but it missed too many customers who actually churned. Because the purpose of this project is early retention, I used a separate validation set to choose a 33% cut-off. I then tested that decision once on the untouched 20% test set.

| Metric | Holdout result |
|---|---:|
| Accuracy | 76.30% |
| Precision | 54.00% |
| Recall | 72.19% |
| F1 score | 61.78% |
| ROC-AUC | 84.19% |
| Average precision | 63.34% |
| Operating threshold | 33% |

With the 33% cut-off, the model found about 72% of the actual churners in the test set. The trade-off is that some customers who would have stayed are also flagged. In a real company, the final cut-off should depend on the cost of an offer, the value of the customer and how many people the retention team can contact.

## Dataset and scope

The CSV is the commonly used IBM-style Telco Customer Churn sample. `Churn` is the outcome I am trying to predict. I removed `customerID` because an identification number should not help the model decide whether a person will leave. Blank entries in `TotalCharges` are converted to missing values and handled inside the pipeline.

I discuss how the idea could be useful in Nigeria, but the dataset does not come from a Nigerian network provider. The results should therefore be treated as a learning exercise, not as evidence about Nigerian subscribers. A real deployment would need recent local data, a privacy review, fairness checks and regular monitoring.

## Repository structure

```text
.
├── app.py                         # Streamlit application
├── train_model.py                 # Reproducible training and validation
├── telecom_churn_capstone.ipynb   # Executed analysis notebook
├── Telecom Churn Model1.csv       # Input dataset
├── artifacts/
│   └── telecom_churn_model.joblib # Trained pipeline and metadata
├── .github/workflows/ci.yml        # Automated GitHub checks
├── .streamlit/config.toml          # Streamlit theme and server config
├── check_project.py                # Local and CI validation suite
├── SECURITY.md                     # Secret and data-handling guidance
├── runtime.txt                     # Deployment Python version
├── requirements.txt
└── README.md
```

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Validate the complete repository before committing:

```bash
python check_project.py
```

## Publish to GitHub

Create an empty GitHub repository named `telecom-churn-prediction`, then run these commands inside this project folder:

```bash
git init
git branch -M main
git add .
git commit -m "Build telecom churn prediction capstone"
git remote add origin https://github.com/4peace1/telecom-churn-prediction.git
git push -u origin main
```

If the GitHub repository already contains a README or licence, clone that repository first and copy these project files into it before committing. Do not force-push over existing work.

## Deploy on Streamlit Community Cloud

1. Sign in to Streamlit Community Cloud with GitHub.
2. Select the `4peace1/telecom-churn-prediction` repository.
3. Choose branch `main` and entry point `app.py`.
4. Deploy. No application secrets are required for this educational version.
5. Add the deployed URL to the top of this README.

## Reproduce the notebook

Open `telecom_churn_capstone.ipynb` in Jupyter or VS Code and run all cells from top to bottom. The notebook uses only the packages listed in `requirements.txt` plus Jupyter itself.

## What I learned from the data

- About 26.5% of the customers in the sample had churned.
- Customers on month-to-month contracts had a much higher churn rate than those on longer contracts.
- Churn was also more common among newer customers and customers without technical-support or online-security services.
- Higher monthly charges, fibre-optic service and electronic-check payments were useful signals for the model.
- These patterns help with prediction, but they do not prove what caused an individual customer to leave.

## Using the result responsibly

- A risk score should guide a conversation, not make a final decision about a customer.
- The model should not be used to deny service or apply unfair treatment.
- Only data that is necessary and properly authorised should be collected.
- Performance should be checked across customer groups and monitored as behaviour changes.
- A probability is an estimate, not a guarantee that someone will churn.

## Submission checklist

- [x] Problem statement and business objective
- [x] Data cleaning and exploratory analysis
- [x] Reproducible preprocessing pipeline
- [x] Model evaluation beyond accuracy
- [x] Validation-based decision threshold
- [x] Working Streamlit application
- [x] Deployment-ready requirements and saved model
- [x] Limitations and responsible-use guidance
- [x] Fellow information confirmed and updated consistently
- [ ] Add a public repository URL and deployed Streamlit URL after publishing

## Security note

Secrets and access tokens must never be stored in notebooks or committed to Git. Use environment variables or the hosting platform's secret manager.

## Disclaimer

This is an educational capstone and portfolio project. It would need fresh company data, testing and governance before it could be used in a real retention programme.
