{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Karen J Yang\n",
    "October 22, 2016"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Logistic Regression Model to Predict Homelessness Risk"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Problem 1: \n",
    "\n",
    "Detect or even predict who in the community is at risk of becoming homeless before they end up on the streets, and raplidly connect them with services providers who can help. Remember that many families and individuals become homeless from sudden loss of a job, sexual or physical abuse, addiction, medical debt, mental illness, or unpaid utility bills."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "General Observation 1:\n",
    "Sample dataset was composed of multiple excel sheets that had data from multiple agencies that collected different types of data from their own organization. User IDs were not consistent across the sheets and there were missing data. Datasets ranged in number of observations from 300 - 500.  There were no data in the dataset for people who were non-homeless, which made the target(outcome) variable lack variation.\n",
    "\n",
    "General Observation 2:\n",
    "After collaborating with St. Patrick's staff, I learned that the homeless figure nationally is about half a million. This comes to a rough ballpark estimate of 2% homeless nationally. This includes folks that are \"couch surfers\" as well as those counted in homeless shelters and on the street. Roughly 98% of the population is non-homeless. Data is skewed so fallout may happen during the calculations due to insufficient data.\n",
    "\n",
    "General Observation 3:\n",
    "The current sample dataset is unusable for the project but it is still possible to create code for a future feasible dataset. A mock dataset was created to build prediction models. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# read dataset2.csv\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import statsmodels.api as sm\n",
    "\n",
    "path = '/Users/karenyang/Desktop/dataset2.csv'\n",
    "df = pd.read_csv(path)  # create dataframe"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1000"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(df)  # 1000 observations"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    980\n",
       "1     20\n",
       "Name: risk, dtype: int64"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# examine the class distribution on the outcome variable\n",
    "# Counts match to national estimate of 2% homeless(coded as 1) and 98% non-homeless(coded as 0)\n",
    "df.risk.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1    501\n",
       "0    499\n",
       "Name: sudden_job_loss, dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# examine the class distribution on the predictors\n",
    "df.sudden_job_loss.value_counts()  # sudden job loss==1, 0 otherwise"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    825\n",
       "1    175\n",
       "Name: sex_phys_abuse, dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# examine the class distribution on the predictors\n",
    "df.sex_phys_abuse.value_counts()  # sex or physical abuse==1, 0 otherwise"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    900\n",
       "1    100\n",
       "Name: addiction, dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.addiction.value_counts() # addiction==1, 0 otherwise"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    900\n",
       "1    100\n",
       "Name: med_debt, dtype: int64"
      ]
     },
     "execution_count": 10,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.med_debt.value_counts() # medical debt==1, 0 otherwise"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    900\n",
       "1    100\n",
       "Name: mental_ill, dtype: int64"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.mental_ill.value_counts() # mental illness==1, 0 otherwise"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0    900\n",
       "1    100\n",
       "Name: unpaid_utility_bills, dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.unpaid_utility_bills.value_counts() # unpaid utility bills==1, 0 otherwise"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Create formula for regression. Build model using glm() function which is part of the formula submodule (statsmodel)\n",
    "formula = 'risk ~ sudden_job_loss+sex_phys_abuse+addiction+med_debt+mental_ill+unpaid_utility_bills'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# glm() fits generalized linear models, a class of model that includes logistic regression\n",
    "import statsmodels.formula.api as smf\n",
    "model = smf.glm(formula=formula, data=df, family=sm.families.Binomial())\n",
    "result = model.fit()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                 Generalized Linear Model Regression Results                  \n",
      "==============================================================================\n",
      "Dep. Variable:                   risk   No. Observations:                 1000\n",
      "Model:                            GLM   Df Residuals:                      993\n",
      "Model Family:                Binomial   Df Model:                            6\n",
      "Link Function:                  logit   Scale:                             1.0\n",
      "Method:                          IRLS   Log-Likelihood:                -79.405\n",
      "Date:                Sat, 22 Oct 2016   Deviance:                       158.81\n",
      "Time:                        16:36:37   Pearson chi2:                     430.\n",
      "No. Iterations:                    27                                         \n",
      "========================================================================================\n",
      "                           coef    std err          z      P>|z|      [95.0% Conf. Int.]\n",
      "----------------------------------------------------------------------------------------\n",
      "Intercept               -3.1023      0.297    -10.459      0.000        -3.684    -2.521\n",
      "sudden_job_loss        -25.4979   4.18e+04     -0.001      1.000     -8.19e+04  8.18e+04\n",
      "sex_phys_abuse           1.0240      0.512      2.002      0.045         0.021     2.027\n",
      "addiction               -1.0666      1.039     -1.026      0.305        -3.104     0.971\n",
      "med_debt               -24.1148   8.49e+04     -0.000      1.000     -1.67e+05  1.66e+05\n",
      "mental_ill              -0.3537      0.767     -0.461      0.645        -1.857     1.150\n",
      "unpaid_utility_bills    -0.5643      0.762     -0.740      0.459        -2.059     0.930\n",
      "========================================================================================\n"
     ]
    }
   ],
   "source": [
    "print(result.summary())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Intercept               -3.102287\n",
      "sudden_job_loss        -25.497909\n",
      "sex_phys_abuse           1.023986\n",
      "addiction               -1.066563\n",
      "med_debt               -24.114835\n",
      "mental_ill              -0.353743\n",
      "unpaid_utility_bills    -0.564323\n",
      "dtype: float64\n"
     ]
    }
   ],
   "source": [
    "#print(\"Coefficients\")\n",
    "print(result.params)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Intercept               1.338693e-25\n",
      "sudden_job_loss         9.995129e-01\n",
      "sex_phys_abuse          4.531242e-02\n",
      "addiction               3.048437e-01\n",
      "med_debt                9.997735e-01\n",
      "mental_ill              6.446779e-01\n",
      "unpaid_utility_bills    4.592105e-01\n",
      "dtype: float64\n"
     ]
    }
   ],
   "source": [
    "#print(\"p-Values\")\n",
    "print(result.pvalues)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "risk\n"
     ]
    }
   ],
   "source": [
    "#print(\"Dependent Variables\")\n",
    "print(result.model.endog_names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[  4.30129999e-02   1.06200227e-12   1.11223756e-01   1.52343576e-02\n",
      "   2.49258008e-02   4.30129999e-02   1.52343576e-02   4.30129999e-02\n",
      "   1.11223756e-01   1.07441346e-02]\n"
     ]
    }
   ],
   "source": [
    "predictions=result.predict()\n",
    "# Show the first 10 probabilities\n",
    "print(predictions[0:10])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Convert predicted probabilites into class labels so that we can make predictions\n",
    "predictions_nominal = [1 if x > 0.5 else 0 for x in predictions]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[980   0]\n",
      " [ 20   0]]\n"
     ]
    }
   ],
   "source": [
    "# Determine how many were correctly or incorrectly classified\n",
    "# The on-diagonal numbers reflect correct preditions while the off-diagonal numbers reflect incorrect predictions\n",
    "from sklearn.metrics import confusion_matrix, classification_report\n",
    "print(confusion_matrix(df['risk'], predictions_nominal))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             precision    recall  f1-score   support\n",
      "\n",
      "          0      0.980     1.000     0.990       980\n",
      "          1      0.000     0.000     0.000        20\n",
      "\n",
      "avg / total      0.960     0.980     0.970      1000\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Applications/anaconda/lib/python3.5/site-packages/sklearn/metrics/classification.py:1074: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.\n",
      "  'precision', 'predicted', average, warn_for)\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(df['risk'], predictions_nominal, digits=3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Trained and tested on the same dataset.\n",
    "# Training error rate is typically over-optimistic since it underestimates the test error rate.\n",
    "# Next step is to split dataset and train to build model on 1st set and then test model off of the 2nd dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from sklearn.cross_validation import train_test_split\n",
    "train, test = train_test_split(df, test_size = 0.2)\n",
    "x_train = train[['sudden_job_loss', 'sex_phys_abuse', 'addiction', 'med_debt', 'mental_ill', 'unpaid_utility_bills']]\n",
    "y_train = train.risk\n",
    "\n",
    "x_test = test[['sudden_job_loss', 'sex_phys_abuse', 'addiction', 'med_debt', 'mental_ill', 'unpaid_utility_bills']]\n",
    "y_test = test.risk"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Specify the formula\n",
    "# Build model using glm() function which is part of the formula submodule (statsmodel)\n",
    "formula1 = 'risk ~ sudden_job_loss+sex_phys_abuse+addiction+med_debt+mental_ill+unpaid_utility_bills'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# train model on 1st dataset\n",
    "model1 = smf.glm(formula=formula1, data=train, family=sm.families.Binomial())\n",
    "result1 = model1.fit()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Use model to predict homelessness risk, using test set data\n",
    "predictions1 = result1.predict(x_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Convert predicted probabilites into class labels so that we can make predictions\n",
    "predictions_nominal_test = [1 if x > 0.5 else 0 for x in predictions1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             precision    recall  f1-score   support\n",
      "\n",
      "          0      0.980     1.000     0.990       196\n",
      "          1      0.000     0.000     0.000         4\n",
      "\n",
      "avg / total      0.960     0.980     0.970       200\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Applications/anaconda/lib/python3.5/site-packages/sklearn/metrics/classification.py:1074: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.\n",
      "  'precision', 'predicted', average, warn_for)\n"
     ]
    }
   ],
   "source": [
    "print(classification_report(y_test, predictions_nominal_test, digits=3))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# Test error rate is 1 - recall, which is zero. Data is too skewed for test error rate to be a good metric. \n",
    "# Purpose of mock dataset is to build the frameworks for the models. Results will be different for the actual dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[  4.49438202e-02   4.67798616e-13   4.67798616e-13   4.49438202e-02\n",
      "   4.67798616e-13   4.67798616e-13   4.49438202e-02   4.67798616e-13\n",
      "   4.49438202e-02   4.49438202e-02]\n"
     ]
    }
   ],
   "source": [
    "# Refit a logistic model using only 2 variables, sudden job loss and medical debt, for illustration only.\n",
    "# When using a real dataset, use top predictors to re-run models to see if prediction results are improved.\n",
    "train2 = train[['sudden_job_loss', 'med_debt', 'risk']]\n",
    "x_train2 = train[['sudden_job_loss', 'med_debt']]\n",
    "y_train2 = train.risk\n",
    "\n",
    "x_test = test[['sudden_job_loss', 'med_debt']]\n",
    "y_test = test.risk\n",
    "\n",
    "# Build model using glm() function which is part of the formula submodule (statsmodel)\n",
    "formula2 = 'risk ~ sudden_job_loss+med_debt'\n",
    "\n",
    "model2 = smf.glm(formula=formula2, data=train2, family=sm.families.Binomial())\n",
    "result2 = model2.fit()\n",
    "predictions2=result2.predict(x_test)\n",
    "# Show the first 10 probabilities\n",
    "print(predictions2[0:10])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             precision    recall  f1-score   support\n",
      "\n",
      "          0      0.980     1.000     0.990       196\n",
      "          1      0.000     0.000     0.000         4\n",
      "\n",
      "avg / total      0.960     0.980     0.970       200\n",
      "\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Applications/anaconda/lib/python3.5/site-packages/sklearn/metrics/classification.py:1074: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.\n",
      "  'precision', 'predicted', average, warn_for)\n"
     ]
    }
   ],
   "source": [
    "predictions_nominal_test2 = [1 if x > 0.5 else 0 for x in predictions2]\n",
    "print(classification_report(y_test, predictions_nominal_test2, digits=3))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "# predicting homeless or non-homeless with new data on sudden job loss and medical debt variables only \n",
    "y_pred = result2.predict(pd.DataFrame([[0,0],[1,1]], columns = [\"sudden_job_loss\",\"med_debt\"]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[  4.49438202e-02   1.31853313e-23]\n"
     ]
    }
   ],
   "source": [
    "print(y_pred)  # predicted outcomes for new data turn out to be 0, which is non-homeless"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Point 1:\n",
    "\n",
    "Null accuracy is predicting off of the outcome variable based on its proportion. Without knowing the factors for homelessness, if one were to predict person is non-homeless 100% of the time, the error rate would be roughly 2%. Still, it is worth pursuing prevention steps and identifying the on onset of homelessness prior to its happening since it is a substantive issue that can affect as many as 0.5 million lives per year."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Point 2:\n",
    "\n",
    "Future data collection efforts needs to focus on standardizing the features that are used across organizations to better track people vulnerable to homelessness and to offer a better understanding of these factors."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Point 3:\n",
    "\n",
    "The roughly 2% estimate of homelessness is relatively small, suggesting human resourcefulness and ingenuity to solve their own housing problems by \"couch surfing\", \"managing\", or taking other measures."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Point 4:\n",
    "\n",
    "Risk factors for homelessness can be used as a prevention measure to help improve lives, provide shelter, and to create stability within the community. With an estimated 47% of jobs predicted to be taken over by computers over the next 10 to 20 years, local leaders need to create a massive housing plan, possibly rehabbing vacant and abandoned city-owned properties into readily made available housing units for temporary or permanent housing for a future homeless population that will most likely exceed 2%. \n",
    "    \n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [Root]",
   "language": "python",
   "name": "Python [Root]"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.5.2"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
