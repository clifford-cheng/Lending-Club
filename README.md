# Lending-Club

![](images/Front_.png)

> Project Goals:
- Create a loan success and default prediction model. 
- Compare returns of built model with loan grade aggregates. 
---

### Table of Contents
- [Description](#description)
- [Data Understanding](#data-understanding)
- [Model Prediction and Job Title Analysis](#data-preparation-and-NLP-prediction)
- [Model Returns](#NMF-Topic-Analysis)
- [Author-Info](#author-info)

---
## Description

Over the past 20 years interest rates have fallen precipitously with the 5 year US treasury yield at 0.29%. With debt obligations projected to reach many multiples of GDP, it is unlikely governments will return to higher interest rates in order to be able to service their massive debts. As a result, individuals and institutional investors alike are desperately seeking alternative safe investment opportunities to generate returns. The appeal of peer-to-peer lending has surged as it allows investors to chase higher yields while mitigating their risk profiles.   

LendingClub is the first and largest SEC registered peer-to-peer lending platform that allows borrowers to create unsecured personal loans for investors to fund. LendingClub operates by approving loan applicants, assigning a risk grade to each loan (A-G), and then packages the loan for funding by outside investors. While safer than the volitile nature of stocks, peer-to-peer loans still carry an inherent level of risk directly related to the security of the future cashflow of the borrower. This project seeks to dive deeper into quantifying the inherent risk of each personal loan in order to build a machine learning model that can predict whether a loan will be successfully paid or not. The success metric of this project can be easily determined by comparing portfolio returns of the risk grade aggregate and the portfolio arising from the model's prediction. 

[Back To The Top](#Semantic-Topic-Analysis)

---

## Data Understanding

The data consisted of 1,816,217 loans (1GB) from LendingClub's website. 39 features were extracted from the data provided for each loan, a few of the features are provided as an example below:

Issue year: (Numerical) Year loan was issued
Grade: (Categorical) Lending Club grade assigned to the loan
Subgrade: (Categorical) Lending Club sub-grade assigned to the loan
Funded amount: (Numerical) Loan amount funded
Term: (Categorical) Term length of loan, can only take on values of either 36 months or 60 months
Purpose: (Categorical) Purpose of the loan, includes credit card consolidation, deb
Application type: (Categorical) Type of application, can take on either Individual or Joint Application
FICO: (Numerical) FICO score of borrower, created by averaging the low and high range of the borrower's FICO scores
DTI: (Numerical) Debt to Income Ratio of the borrower based on the monthly debt payments charged to the borrower versus - monthly income
Annual income: (Numerical) Annual salary of borrower
Employment length: (Numerical) Years in current role
Home ownership: (Categorical) Indicates whether a borrower owns a home, is paying off a mortgage, rents or has some other living situation
Address state: (Categorical) Indicates state the borrower is applying from
Earliest credit line: (Numerical) The earliest year the borrower had a credit line
Negative activity: (Numerical) Combination of counts of public record bankruptcies and other credit adverse events
Inquiries within the last 6 months: (Numerical) Count of times borrower's credit report was inquired upon within the last 6 months
Delinquencies within the last 6 months: (Numerical) Count of delinquencies within the last 6 months
Open accounts: (Numerical) Count of open accounts the borrower has
Total Current Balance: (Numerical) Total outstanding credit of borrower
Loan Status: (Categorical) Indicates whether loan was Fully Paid or Charged-Off

<p align="center">
<img src="images/data_overview.png" width="600" height="400">
<p/>


There were additional meta-data such as date/time of the review, information of the reviewer, and image/product description (mostly NaNs). For this specific project, this data was not used. After cleaning and removing duplicate reviews roughly 450,000 reviews remained. In order to properly weight the importance of upvotes, the square root of the number of upvotes was used to duplicate the reviews. A logarithmic approach was considered but deemed too harsh for higher upvoted reviews as small scale upvoting and downvoting product attacks are quite common in this intense market environment. After compensating for upvoted reviews, the total number of reviews was 558,924. 

<p align="center">
<img src="images/1_Histogram_Per.png" width="550" height="450">
<p/>

[Back To The Top](#Semantic-Topic-Analysis)

---

## Data Preparation and NLP Prediction

I utilized Google Colab for the running of the models as I had Colab Pro. However, Amazon SageMaker or creating an AWS Jupyter Notebook EC2 Server with a static IP address were equally robust and viable options.   

The data was split into two categories: Low Rating (1-3 stars) and High Rating (4-5 stars). Splitting into 3 categories, Low Rating (1-2 stars), Average (3 stars), and High Rating (4-5), was considered but greatly dimished the power of the analysis due to limited number of 3 star samples in proportion to the other categories. After the split, the data was grouped into 24.8% (141,091) Low Rating and 74.2% (417,834) High Rating reviews. A quarter of 141,091 reviews were randomly sampled (35,272) and the same number was randomly sampled from the High Rating reviews. This would be used for training and testing the best model and had a balance of 50% Low Rating / 50% High Rating. The model would then be tested on the remaining 21.7% (105,819) Low Rating and 78.3% High Rating (382,563) review samples that were untouched throughout this process. 

<p align="center">
<img src="images/data_split.png" width="600" height="500">
<p/>

Logistic Regression, Random Forest, Linear SVC, and Multi-Naive Bayes were train/tested on the sub-sample data and then tested on the remaining untouched data. Linear SVC was the best model with an f1-score of 91% and 73% for predicting High Ratings and Low Ratings respectively. This is quite impressive considering the remaining data had a porportion of 78.3% High Rating and 21.7% Low Rating.

<p align="center">
<img src="images/NLP_TESTING.png" width="700" height="450">
<p/>

[Back To The Top](#Semantic-Topic-Analysis)


---

## NMF Topic Analysis

The next task was to obtain product specific insights from review text. K-Means, LDA, and NMF were used to find latent topics within the text data. Stemming (snowball stemmer, stemmer, etc...) and Lemmitization were tested but the most effective method was to build the database of stopwords by removing any mention of product, thereby focusing specifically on the feeling of the customer to the product. After much testing, 15 components were chosen as it optimized the balance between having product details specific enough to obtain actionable insights but general enough to properly group text data. The NMF model would then output the top words associated with each latent topic. The NMF model required the aid of human pattern recognition in order to properly create classifications derived from understanding the relationship among the key words within each latent topic. The categorized latent topics would then be mapped to their corresponding reviews. 

<p align="center">
<img src="images/NMF_Comp_High.png" width="700" height="500">
<p/>

<p align="center">
<img src="images/NMF_Comp_Low.png" width="700" height="500">
<p/>

In order to visualize the higher dimensional vectorized text data, a 1,050,000+ dimensional space needed to be mapped to an interpretable 2D plot. TSNE was used as a dimensional-reduction method that preserved the distance between latent topics. TSNE's underlying probablitic methology counter-acted the curse of dimensionality as it preserved informational loss as the data was moved from higher to lower dimensions. The end result was a color-coded 2D map of the latent topics and their corresponding key words. 


![](images/TSNE_High.png)
![](images/TSNE_Low.png)

The classification of text data using semi-supervised methods enables product sellers to obtain greater insights from review text. Not only can sentiment be obtained from review text data (High Rating | Low Rating), but the reason for that customer sentiment can also be discovered. This means product sellers can understand the main reasons behind a high or low rating and make adjustments accordingly. For example, if an increased number of reviews were being classified as "Extremely Disappointed" a product seller could see the main cause of the low rating was that the product arrived dented and broken. This could translate to improving packing and shipping procedures in order to ensure the product arrived safely at the customers' doorstep. Alternatively, the model could also pick up on the fact that a particular product was purchased as gifts for other family members. This could lead to targeted promotions of the product during special events.

The world of NLP is an exciting space and I hope to continue to build similar projects that can be applied to solve business solutions and gather important insights to allow business to understand their customers on a deeper level.


---

## Author Info

- LinkedIn - [clifford-cheng](https://www.linkedin.com/in/clifford-cheng/)
- Email - cliffpcheng@gmail.com



