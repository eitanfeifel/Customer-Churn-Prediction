from openai import OpenAI
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import utils
import os

client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get('GROQ_API_KEY'))


def load_model(filename):
  with open(filename, "rb") as file:
    return pickle.load(file)


xgboost_model = load_model('xgb_model.pkl')

naive_bayes_model = load_model('nb_model.pkl')

random_forest_model = load_model('rf_model.pkl')

decision_tree_model = load_model('dt_model.pkl')

svm_model = load_model('svm_model.pkl')

knn_model = load_model('knn_model.pkl')

voting_classifier_model = load_model('voting_clf.pkl')

xgboost_SMOTE_model = load_model('xgboost_model-smote.pkl')

xbgoost_fe_model = load_model('xgboost_model-fe.pkl')


def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_products, has_credit_card, is_active_member,
                  estimated_salary):

  input_dict = {
      'CreditScore': credit_score,
      'Age': age,
      'Tenure': tenure,
      'Balance': balance,
      'NumOfProducts': num_products,
      'HasCrCard': int(has_credit_card),
      'IsActiveMember': int(is_active_member),
      'EstimatedSalary': estimated_salary,
      'Geography_France': 1 if location == 'France' else 0,
      'Geography_Spain': 1 if location == 'Spain' else 0,
      'Geography_Germany': 1 if location == 'Germany' else 0,
      'Gender_Male': 1 if gender == 'Male' else 0,
      'Gender_Female': 1 if gender == 'Female' else 0
  }

  input_df = pd.DataFrame([input_dict])

  return input_df, input_dict

def calculate_percentiles( selected_customer, df ):
  features = [ 'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
              'EstimatedSalary' ]
  percentiles = {}

  for feature in features:
      customer_value = selected_customer[ feature ]

      percentile = (df[feature] <= customer_value).mean() * 100
      percentiles[feature] = percentile

  return percentiles
  
def make_predictions(input_df, input_dict, df):

  probabilities = {
      'XGBoost': xgboost_SMOTE_model.predict_proba(input_df)[0][1],
      'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
      'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
      'SVM': svm_model.predict_proba(input_df)[0][1],
    
  }

  avg_probability = np.mean(list(probabilities.values()))

  col1, col2 = st.columns(2)

  with col1:
    fig = utils.create_gauge_chart(avg_probability)
    st.plotly_chart(fig, use_container_width=True)
    st.write(
        f"The customer has a {avg_probability:.2%} probability of churning.")

  with col2:
    fig = utils.create_model_probability_chart(probabilities)
    st.plotly_chart(fig, use_container_width=True)

  percentiles = calculate_percentiles( selected_customer, df )

  # print("perc", percentiles)
  percentile_chart = utils.create_percentile_bar_chart( percentiles )
  st.plotly_chart( percentile_chart, use_container_width = True )

  # st.markdown( "### Model Probabilites")
  # for model, prob in probabilities.items():
  #   st.write( f"{ model } { prob }" )
  # st.write( f"Average Probability: { avg_probability }" )

  return avg_probability



def explain_prediction(probability, input_dict, surname):

  prompt = f""" You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of Machine Learning Models.

  Your Machine Learning Model has predicted that a customer named { surname } has a { round( probability * 100 ) }% probability of churning, based on the following information provided below:
  { input_dict }

Here are the Machine Learning Model's top 10 most important features for predicting churn, along with their respecitve importance scores:

NumOfProducts:	0.323888
IsActiveMember:	0.164146
Age:	0.109550
Geography_Germany:	0.091373
Balance:	0.052786
Geography_France:	0.046463
Gender_Female:	0.045283
Geography_Spain:	0.036855
CreditScore:	0.035005
EstimatedSalary:	0.032655
HasCrCard:	0.031940
Tenure:	0.030054
Gender_Male:	0.000000

{ pd.set_option( 'display.max_columns', None ) }

Here are summary statistics for the churned customers:
{ df[ df[ 'Exited' ] == 1 ] .describe() }

Here are summary statistics for the non-chuner customers:
{ df[ df[ 'Exited' ] == 0 ] .describe() }

Given the customer's churn probability, determine if the customer has over or under a 40% risk of churning.
If they are over 40%, then generate a 3 sentence explanation of why they are at risk for churning.

However, if the customer has less than a 40% risk of churning, generate a 3 sentence explanation of why they might not be at risk for churning.
Your explanation should be based on the customer's information provided above, the summary statistics of churned and non-churned customers, and the feature importances provided.

Here are some important things to keep in mind:
Don't mention the probability of churning, or the machine learning model, and don't say anything like "Based on the Machine Learning Model's predictions, and top 10 most important features, ..." just explain the prediction in a natural and digestable fashion. Be informative, and explain any assumptions made in the explanation or names that may be misleading. Provide enough information to give a reader in-depth insight into how they can interact with the customer to either prevent churn, or retain loyalty.
  """

  # print("EXPLANATION_PROMPT", prompt)

  raw_response = client.chat.completions.create(
      model="llama-3.1-8b-instant",
      messages=[{
          "role": "user",
          "content": prompt
      }],
  )

  return raw_response.choices[0].message.content


def generate_email(probability, input_dict, explanation, surname):
  prompt = f""" Your role is being the manager at HS Bank. You are responsible for ensuring customers stay commited to the bank and are incentivized through various offers.

  You have been assigned with emailing customers, based on their churn probability, to incentivize them to stay loyal to your bank.

  The cutsomer, { surname }, has a { round( probability * 100 ) }% probability of churning.
  The customer's information is as follows:
  { explanation }

  Your task is to generate an email to the customer, based on their given information, asking them to stay loyal to the bank if they are at risk of churning, or offering them incentives so that they become more loyal to the bank.

  Your offers and incentives, should be heavily based on factors that might contribute to their risk of churning. If, for example, not having a credit card, or not being an active member of the bank, or having a low credit-score then you should offer them incentives to change those factors for the benefit of their churn probability.

  Make sure to list out a set of incentives relevant to them based on their information, in bullet point format -- provide at most 4 incentives and make sure the bullets are formatted properly. Don't ever mention their given probability of churning, or the machine learning model to the customer.
  """

  raw_response = client.chat.completions.create(
      model="llama-3.2-3b-preview",
      messages=[{
          "role": "user",
          "content": prompt
      }],
  )

  # print("\n\nEMAIL PROMPT", prompt)

  return raw_response.choices[0].message.content


st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

customers = [
    f"{ row['CustomerId' ] } - {row[ 'Surname' ] } "
    for _, row in df.iterrows()
]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
  selected_customer_id = int(selected_customer_option.split(" - ")[0])
  print("Selected Customer ID: ", selected_customer_id)

  selected_surname = selected_customer_option.split(" - ")[1]
  print("Surname: ", selected_surname)

  selected_customer = df.loc[df['CustomerId'] == selected_customer_id].iloc[0]
  print(selected_customer)

  col1, col2 = st.columns(2)

  with col1:

    credit_score = st.number_input("Credit Score",
                                   min_value=300,
                                   max_value=850,
                                   value=int(selected_customer['CreditScore']))

    location = st.selectbox("Location", ["Spain", "France", "Germany"],
                            index=["Spain", "France", "Germany"
                                   ].index(selected_customer["Geography"]))

    gender = st.radio("Gender", ["Male", "Female"],
                      index=0 if selected_customer["Gender"] == "Male" else 1)

    age = st.number_input("Age",
                          min_value=18,
                          max_value=100,
                          value=int(selected_customer["Age"]))

    tenure = st.number_input("Tenure (Years)",
                             min_value=0,
                             max_value=50,
                             value=int(selected_customer["Tenure"]))

  with col2:

    balance = st.number_input("Balance",
                              min_value=0.0,
                              value=float(selected_customer["Balance"]))

    num_products = st.number_input("Number of Products",
                                   min_value=1,
                                   max_value=10,
                                   value=int(
                                       selected_customer["NumOfProducts"]))

    has_credit_card = st.checkbox("Has Credit Card",
                                  value=bool(selected_customer["HasCrCard"]))

    is_active_member = st.checkbox("Is Active Member",
                                   value=bool(
                                       selected_customer["IsActiveMember"]))

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=float(selected_customer["EstimatedSalary"]))

  input_df, input_dict = prepare_input(credit_score, location, gender, age,
                                       tenure, balance, num_products,
                                       has_credit_card, is_active_member,
                                       estimated_salary)

  avg_probability = make_predictions(input_df, input_dict, df)

  explanation = explain_prediction(avg_probability, input_dict,
                                   selected_customer["Surname"])

  st.markdown("---")
  st.subheader("Explanation of Prediction")

  st.markdown(explanation)

  email = generate_email(avg_probability, input_dict, explanation,
                         selected_customer["Surname"])

  st.markdown("---")
  st.subheader("Personalized Email")

  st.markdown(email)
