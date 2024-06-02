# Description of submission

Chosen model: Random Forest with SMOTE - Chosen predictors:

'nomem_encr', #Unique number

'outcome_available', #1 = outcome is available --> merge with outcome dataset

'cf08a024', 'cf10c024', 'cf13f024', 'cf14g024', 'cf15h024', 'cf16i024', 'cf17j024', 'cf18k024', 'cf19l024', 'cf20m024', # Do you currently have a partner? 

'cf11d025', 'cf14g025', 'cf15h025', 'cf18k025', 'cf20m025', # Do you live together with this partner? 

'cf08a026', 'cf10c026', 'cf14g026', 'cf15h026', 'cf16i026', 'cf17j026', 'cf18k026', 'cf19l026', 'cf20m026', # What is his or her year of birth? 

'cf12e030', 'cf13f030', 'cf14g030', 'cf15h030', 'cf18k030', 'cf19l030', 'cf20m030', # Are you married to this partner? 

'cf08a032', 'cf14g032', 'cf17j032', 'cf18k032', 'cf19l032', 'cf20m032', # What is your partner's gender? 

'cf08a035', 'cf14g035', # Have you had any children? 

'cf09b128', 'cf12e128', 'cf14g128', 'cf15h128', 'cf16i128', 'cf17j128', 'cf18k128', 'cf19l128', 'cf20m128', # Do you think you will have children in the future? 

'cf10c129', 'cf12e129', 'cf13f129', 'cf14g129', 'cf15h129', 'cf16i129', 'cf17j129', 'cf18k129', 'cf19l129', 'cf20m129', # How many children do you think you will have in the future? 

'cf08a130', 'cf10c130', 'cf11d130', 'cf12e130', 'cf14g130', 'cf15h130', 'cf16i130', 'cf17j130', 'cf18k130', 'cf19l130', 'cf20m130', # Within how many years do you hope to have your (first-next) child? 

'cf11d180', 'cf14g180', 'cf15h180', 'cf16i180', 'cf17j180', 'cf18k180', 'cf19l180', 'cf20m180', # How satisfied are you with your current relationship? 

'cf10c181', 'cf11d181', 'cf14g181', 'cf17j181', 'cf18k181', 'cf19l181', 'cf20m181', # How satisfied are you with your family life? 

'cf14g188', # How is the household work divided between you and your partner? - laundry, ironing 

'cf12e189', 'cf13f189', # How is the household work divided between you and your partner? - house cleaning 

'cf14g190', # How is the household work divided between you and your partner? - odd jobs in and around the house 

'cf10c191', # How is the household work divided between you and your partner? - financial administration 

'cf13f192', # How is the household work divided between you and your partner? - grocery shopping 

'cf15h402', 'cf18k402', # Is this partner the same partner you entered in the questionnaire last year? 

'cf14g432', 'cf15h432', 'cf16i432', 'cf17j432', 'cf18k432', 'cf19l432',  # How would you generally describe the relationship with your family?

'cf15h454', 'cf17j454', 'cf18k454', 'cf19l454', 'cf20m454', # Did you ever have any children?

'cf15h455', 'cf16i455', 'cf17j455', 'cf18k455', 'cf20m455', # How many living children do you have in total?

'cf15h471', 'cf17j471', 'cf18k471', 'cf19l471', 'cf20m471', # Did you ever have (a) child(ren) who passed away after being born?

'cf15h483', 'cf16i483', 'cf17j483', 'cf18k483', 'cf19l483', 'cf20m483', # How is the household work divided between you and your partner? - preparing food

'cf15h484', 'cf16i484', 'cf17j484', 'cf18k484', 'cf19l484', 'cf20m484', # How is the household work divided between you and your partner? - laundry, ironing

'cf15h485', 'cf16i485', 'cf17j485', 'cf18k485', 'cf19l485', 'cf20m485', # How is the household work divided between you and your partner? - house cleaning

'cf15h486', 'cf19l486', 'cf20m486', # How is the household work divided between you and your partner? - odd jobs in and around the house

'cf15h487', 'cf16i487', 'cf17j487', 'cf18k487', 'cf19l487', 'cf20m487', # How is the household work divided between you and your partner? - financial administration

'cf15h488', 'cf16i488', 'cf17j488', 'cf18k488', 'cf19l488', 'cf20m488', # How is the household work divided between you and your partner? - grocery shopping

'cf20m526', # How would you generally describe the relationship with your family?

'cf20m527', # How have you and your partner arranged the work of raising and caring for the children? - storyreading, playing games, other forms of play

'cf20m528', # How have you and your partner arranged the work of raising and caring for the children? - bringing to/fetching from daycare or school, attending sports activities, clubs, etc.

'cf20m529', # How have you and your partner arranged the work of raising and caring for the children? - talking about problems in school

'cf20m530', # How have you and your partner arranged the work of raising and caring for the children? - small outings, as to the cinema, zoo, etc.

'cf20m531', # How did you and your partner arrange the work of raising and caring for the children? - storyreading, playing games, other forms of play

'cf20m532', # How did you and your partner arrange the work of raising and caring for the children? - bringing to/fetching from daycare or school, attending sports activities, clubs, etc.

'cf20m534', # How did you and your partner arrange the work of raising and caring for the children? - small outings, as to the cinema, zoo, etc.

'ci12e004', 'ci13f004', 'ci15h004', 'ci16i004', 'ci17j004', # Respondent works, according to household box

'ci08a006', 'ci09b006', 'ci10c006', 'ci11d006', 'ci12e006', 'ci14g006', 'ci15h006', 'ci16i006', 'ci17j006', 'ci18k006', 'ci19l006', 'ci20m006' # How satisfied are you with your financial situatioin?

'gender_imp' # Gender

'birthyear_imp' # Birthyear

'oplzon' # Highest level of education irrespective of diploma

'oplmet'# Highest level of education with diploma

'woonvorm' # Domestic situation

'woning' # Type of dwelling that the household inhabits

'brutoink_f' # Personal gross monthly income in Euros

'new_child' # Whether respondent had child in 2021-2023

Predicting Fertility Data Challenge - June 3rd 2024

1	Introduction

1.1	Introduction and Problem Statement

Fertility has been studied in the recent years, but there is still room for improvement especially about predicting about fertility intentions in the Netherlands (Sivak et al., 2024a, 2024c). The prevalence of childlessness has experienced a substantial rise in the past few decades. In the Netherlands, the percentage of women born between 1935 and 1945 who did not have children was approximately 11%. However, among women born between 1955 and 1975, this percentage increased to approximately 18% (Brakel et al., 2020). Among males, there is a comparable rise from approximately 17% to 26% (Stulp, 2024). Conducting study on fertility intention in the Netherlands is not uncommon, as several other countries (e.g. China (M. Li & Xu, 2022; Zhu et al., 2022), Israel (Preis et al., 2020), Italy (Mussino et al., 2023), Niger (Ahinkorah et al., 2021), Norway (Lappegård et al., 2022), Romania (Ciritel et al., 2019), South-Korea (Kim, 2023), United States (Beaujouan & Berghammer, 2019)) also conduct similar studies about the topic fertility intentions.

Not only has childlessness undergone a severe transformation, but the average number of children per family has also greatly declined, and the average age at which individuals become parents has experienced a steep increase. The primary factor contributing to the rise in childlessness is the substantial postponement of parenthood by couples (Birch Petersen et al., 2015; Stulp, 2024). Women may choose to delay childbearing for various reasons, including career ambitions, relationship status, and their degree of education (Birch Petersen et al., 2015). Frequently, this delay could results in complete infertility. The decline in fertility among couples, especially women aged 35 and above, is attributed to a decrease in their biological ability to conceive and bear children. The longing for offspring is frequently delayed until a time when the ability to conceive becomes more challenging. Medically assisted reproduction provides limited resolution to this problem (Birch Petersen et al., 2015; Stulp, 2024).

The study of fertility is extensively researched across various academic fields due to its significance to both individuals and societies. Several factors have been identified that are associated with fertility findings. However, these significant determinants only account for a small portion of the variations in fertility results, and researchers are unable to even comprehend their short-term changes.

Therefore, PreFer (Predicting Fertility data challenge) wants to know if people in the Netherlands will have children in 2021, 2022 and/or 2023. The objective of the challenge is to assess the level of predictability in fertility intentions in the Netherlands in the (near) future using machine learning (ML) techniques and to address which features are relevant to make this prediction (Sivak et al., 2024a, 2024c).

1.2	Motivation and Relevance

1.2.1	Societal Relevance

The objective of this research about predicting individual fertility intentions in the Netherlands is important because it gives more information and understanding about fertility intentions and possible increase/decrease in birth rate in the (near) future. It could give valuable insights for governments and policymakers to make adaptions on time if necessary.

1.2.2	Scientific Relevance

Research about predicting fertility intention in the Netherlands has not been done before using machine learning algorithms, therefore this research provides new knowledge and brings novelty.

2	Literature Review / Related Work

2.1	Theoretical Framework: Traits-Desires-Intentions-Behaviour (T-D-I-B)

The Traits-Desires-Intentions-Behaviour (T-D-I-B) framework defines the sequential progression of motivations and psychological states that impact human decision-making toward childbearing (Miller, 2011b). According to the research of Miller (2011a), this framework begins with motivational traits related to the reasoning to have children, which may consist of positive and negative traits, including emotions and beliefs about children, as well as our behaviors in interactions with them. These factors play an essential part in creating relationships. The crucial element here is the concept of nurturance, which involves emotions such as fear and affection. These emotions serve as the driving force behind the desire to love and to protect one’s children. These traits subsequently lead to particular desires toward having children, having how many children, and the timing of having offspring. These three components of desires influence the fertility intentions resulting in whether having a child or not having a child. Individual’s fertility intention determines their behaviors focusing on at either achieving or preventing conception. There are three primary behavioral approaches: actively pursuing conception, actively avoiding conception, and being neutral or inactive towards conception. Each behavioral approach is linked to its own distinct behavioral preferences per person.

2.2	Fertility Intentions Factors

Upon the study of Miller (2011b), fertility intention refers to an individual’s deliberate choice or decision regarding whether to have children or not. This decision is usually based on factors such as their current situation, available resources, and plans for the future. Fertility intentions typically are based on practical considerations and frequently involve evaluating factors such as financial stability, relationship status, physical wellbeing, and career goals.

Lappegård et al. (2022) revealed that both men and women showed similar responses in the study, suggesting that there are no significant differences between genders in the impact of economic narratives on fertility intentions. Nevertheless, the research conducted by Ahinkorah et al. (2021) revealed that both men and women exhibit a significant fertility desire, but it was observed that men seem to have a little stronger tendency towards having (more) children compared to women. While fertility intentions, specifically the decision to have (a)nother child and the timing of that decision, are not the main focus in this study, the strong desire for children can provide valuable insights on the prevalent high fertility intentions among both men and women.

According to Mussino et al. (2023) and Ajzen and Klobas (2013), the research had shown that demographic and socio-economic factors influenced the fertility intentions. Research of Kim (2023) and Zhu et al. (2022) showed that work-life balance is important and its influence on fertility decisions. Employers and policymakers should establish a supportive environment for family planning within households, including providing financial assistance (such as subsidies, affordable childcare, tax benefits for child-rearing, and future educational resources), and particularly for women with maternity support and extended maternity leave. Furthermore, dividing household task could increase the likelihood of childbearing (Riederer et al., 2019). According to Preis et al. (2020), an important predictor that was utilized, was unpleasant happening afterbirth (postpartum). It could be a predictor whether to have more children in the (near) future.

2.3	Previous Studies

2.3.1	Previous Studies with Statistical Methods

The method that was used for the research of Ciritel et al. (2019) and Zhu et al. (2022) about fertility intention, was statistical methods, specifically logistic regression model, to analyze the data. Logistic Regression is a technique that was used for modeling binary outcomes. Despite the research of Ciritel et al. (2019) was about analyzing data and making it more understandable about fertility intentions, there was no prediction made. The target variable of the study was ’intention to have a child within the next three years’. The predictors that were used, were attitudes, subjective norms, perceived behavioral control, economic characteristics (e.g. dwelling size, income) and background factors (e.g. age, gender, partnership status). The predictors that were utilized in the study of Zhu et al. (2022), are ’age’, ’household income’, ’working conditions’, ’childcare barriers’ and ’socioeconomic status’. Both studies of Mussino et al. (2023) and Riederer et al. (2019) utilized statistical methods focusing on logistic regression model as well, whereas Mussino et al. utilized the predictors ’demographic factors’, ’marital status’, ’labour market status (work)’ and ’home ownership’ and Riederer et al. utilized ’division household tasks’ as predictor.

2.3.2	Previous Studies with Machine Learning methods

In the study of M. Li and Xu (2022), Random Forest (RF) and Extreme Gradient Boosting (XGBoost) were the algorithms that were utilized in this research to make predictions about fertility intentions. Kim (2023) implemented Extreme Gradient Boosting (XGBoost) for making prediction about fertility intention among working women. Grid search techniques was also utilized to optimize the hyperparameters. Kim showed another significant predictors, which was ’work’. The Random Forest model, including hyperparameter tuning and Out-of-Bag , achieves an accuracy of 60.5%, a precision of 60.1%, a recall of 60.6%, and a f1-score of 60.3%. It consistently performs well in all metrics. By utilizing hyperparameter tuning, XGBoost demonstrated comparable outcomes to Random Forest. The XGBoost model achieved an accuracy of 59.8%, precision of 61.3%, recall of 58.1%, and a f1-score of 59.7% (M. Li & Xu, 2022). According to the research of Kim (2023), however, the accuracy of XGBoost was found to be 51.6%, with a precision of 16.6%, recall of 78.3%, and a f1-score of 27.4%. The performances of the last mentioned XGBoost model are poor, but there is confidence that XGBoost will improve due to its utilization of trees as base learners and its ability to correct errors caused by prior trees Chen and Guestrin (2016).

Although, the difference in model performance, M. Li and Xu (2022) and Kim (2023) showed almost similar significant predictors in their research. The significant predictors in both studies are ’age (birth year)’, ’income’, ’education’ and ’marital status (partnership status)’. M. Li and Xu (2022) have utilized 11 predictors for training the models and Kim (2023) has utilized 13 predictors for training the models.

2.4	Predictors

Upon presenting the findings in the previous studies, the research of Hashemzadeh et al. (2021) showed that these predictors ’age’, ’gender’, ’education’, ’partnership status’, ’income’, ’work’, ’dwelling’, ’unpleasant postpartum’ and ’division household tasks’ were also in their study. Moreover, Hashemzadeh et al. (2021) even displayed more predictors, such as ’partnership satisfaction’, ’financial status’, ’social network’, ’happiness’ and ’child desire’ will be utilized as predictors in this research.

3	Method

3.1	Datasets Description

LISS data provided the datasets for the PreFer data challenge. A total of three datasets will be utilized for this research.

3.1.1	First Dataset

The first dataset is a merged dataset that combines 13 individual datasets related to the subject ’Family and Household’, ’Economic Situation: Assets’, ’Economic Situation: Housing’, ’Economic Situation: Income’, ’Health’, ’Personality’, ’Politics and Values’, ’Religion and Ethnicity’, ’Social Integration and Leisure’ and ’Work and Schooling’ from the years 2008 to 2020. The data was obtained through an online survey completed by the members of the LISS data panel. The merged dataset is named ’PreFer_train_data.csv’ and it consists of 6418 rows and 31624 columns. The dataset contains the following main predictors: ’partnership status’, ’work’, ’unpleasant postpartum’, ’division household tasks’, ’partnership satisfaction’, ’financial status’, ’social network’, ’happiness’ and ’child desire’. Additional details regarding the main predictor can be read in section 3.2.3.

3.1.2	Second Dataset

The second dataset is named ’PreFer_train_background_data.csv’ and it consists of 758873 rows and 33 columns. This dataset provides information about the background information of each LISS data panel member from November 2007 to December 2020. On a monthly basis, LISS data publishes an updated list of the panel members to verify their continued participation in the surveys. From November 2007 to December 2020, there are a total of 158 datasets. All these datasets have been combined into a single dataset ’PreFer_train_background_data.csv’. Dataset ’PreFer_train_background_data.csv’ has 12854 unique numbers for the LISS panel members from 4950 households who participated from November 2007 to December 2020. The dataset contains the following predictors: ’age’, ’gender’, ’income’, ’education’ and ’dwelling’. Additional details regarding the main predictor in the ’PreFer_train_background_data.csv’ dataset can be read in section 3.2.5.

3.1.3	Third Dataset

The third dataset is named ’PreFer_train_outcome.csv’ and it consists of 6418 rows and 2 columns. The target variable ’new_child’ can be found in this dataset and the variable ’nomem_encr’ will be utilized for linking to the ’PreFer_train_data.csv’ dataset and the ’PreFer_train_background_data.csv’ dataset. The variable ’nomem_encr’ is present in all three datasets and will be utilized for merging in a later stage. After merging the three datasets, the variable ’nomem_encr’ will be excluded from the prediction process due to the absence of predictive power to predict the target variable.

3.2	Data Preparation: Cleaning and Preprocessing

3.2.1	Train-, Validation- and Test set

Prior to going into the explanation of data cleaning and preprocessing, it is crucial to understand the established choice for the train and test set. The entire merged dataset consists of around 6900 respondents, whereas around 1400 respondents will be utilized for the train- and test set. Approximately 5500 respondents with unknown responses (NaN) will not be taken into consideration in train set and test set, see section 3.2.2 for additional information. PreFer organizers divided the dataset into a train set (70%) and a test set (30%), following its challenge criteria. The train set has 987 samples and the test set 400 samples. Participants will not have access to the test set, therefore PreFer will allow for intermediate submissions on ’Next’ platform to assess model performance. In order to validate, the train set will be split into a train set (60%) and a validation set (10%) (Muraina, 2022). After the training and validation split, the train set has 888 samples and the validation set 99 samples.

3.2.2	Data Cleaning in ’PreFer_train_outcome.csv’: Filtering for Target Variable (Done by the PreFer Organisers)

The target variable, named ’new_child’, is defined in the dataset ’PreFer_train_outcome.csv’. It represents whether individuals born between 1975 and 2002 will have a new child, with data collected from the LISS Panel surveys in 2021, 2022, and 2023. Due to non-participation in these surveys, the outcome for many in the target group could not be determined, resulting in three possible responses: ’yes’, ’no’, and ’NaN (Not a Number)’. NaNs, representing undefined or missing values, are excluded from the analysis. Following the approach used in the study by Kim (2023), which recommends disregarding inaccurate and missing samples, the dataset has been filtered by the PreFer organizers to retain only the responses ’yes’ and ’no’. This results in a binary classification problem with 987 responses: 212 for ’yes’ (21.5% of the total) and 775 for ’no’ (78.5% of the total).

3.2.3	Data Cleaning in ’PreFer_train_data.csv’: Filtering Predictors

Based on literature review, these main predictors have been determined and it is as follows: ’partnership status’, ’work’, ’unpleasant postpartum’, ’division household tasks’, ’partnership satisfaction’, ’financial status’, ’social network’, ’happiness’ and ’child desire’. Nevertheless, these main predictors can be subdivided into sub-predictors, indicating that each main predictor includes one and/or several sub-predictors. For instance, the main predictor, ’partnership status’ is further split into sub-predictors such as ’Do you currently have a partner? (2019)’, ’Do you currently have a partner? (2020)’, ’Are you married to this partner? (2019)’, and ’Are you married to this partner? (2020)’. All these sub-predictors can be found in ’PreFer_train_data.csv’. A total of 173 columns from this dataset ’PreFer_train_data.csv’ are utilized as predictors.

3.2.4	Data Cleaning in ’PreFer_train_data.csv’: Filtering ’outcome_available’

The predictor ’outcome_available’ is important as it indicates whether individuals completed the survey in 2021, 2022, and/or 2023. The possible answers are ’0’ to indicate no and ’1’ to indicate yes. By utilizing a filter to exclude all instances of ’0’, a total of 987 rows will be shown, which corresponds exactly to the number of rows in the target variable ’new_child’. This is important for the process of merging datasets at a later stage.

3.2.5	Data Cleaning in ’PreFer_train_background_data.csv’: Filtering Predictors

The predictors ’age’, ’gender’, ’income’, ’education’ and ’dwelling’ can be found in the ’PreFer_train_background_data.csv’ file. These predictors ’age’, ’gender’ and ’income’ are fixed variables, indicating that they are generated and cleaned by the PreFer organizers. These predictors have no presence of any NaN (Not a Number) values. Regarding the predictors ’age’ and ’birth year’, the meaning of both predictors captures identical data. Therefore, ’birth year’ will be utilized instead of ’age’. The predictor ’birth year’ is easier and is straightforward to utilize, as it regards specifically to individuals born between 1975 and 2002. As indicated, the ’income’ predictor has been imputed, meaning PreFer organizers filled in all missing data. Income may increase, decrease, or remain unchanged. In order to ensure that each individual in ’nomem_encr’ had a single income, the mean was calculated and utilized. Consequently, the ’PreFer_train_background_data.csv’ file contains a total of 987 rows. A total of 7 columns (including ’nomem_encr’) are utilized as predictors from this dataset ’PreFer_train_background_data.csv’.

3.2.6	Data Cleaning in ’PreFer_train_background_data.csv’: Remove Duplicates

By utilizing the ’groupby()’ function in combination with ’.agg()’ for the above mentioned predictors (’birth year’, ’gender’, ’income’, ’education’ and ’dwelling’), the outcome is that each ’nomem_encr’ value becomes unique. However, in order to confirm that ’nomem_encr’ contains only one row and for its uniqueness, the ’drop_duplicates()’ function was utilized to remove duplicated data.

3.2.7	Data Cleaning: Renaming All Predictors

All the columns in the ’PreFer_train_data.csv’ file and the ’PreFer_train _background_data.csv’ file consist of not real representative names (code names). The provided information lacks clarity regarding the content of each column. Therefore, all columns are renamed to match the respective question and/or information.

3.2.8	Data Cleaning: Merging three datasets

All three datasets include the variable ’nomem_encr’, which has been utilized to merge all three datasets. The ’PreFer_train_outcome.csv’ dataset contains 987 responses for the target variable ’new_child’. Therefore, it is necessary for the other two datasets to exhibit identical responses of 987. Upon the combining of three datasets, the final dataset contains a total of 184 columns and 987 rows.

3.2.9	Data Cleaning: Missing Data

Among the total of 181 columns, only 6 columns do not have any missing values. Upon careful consideration, the missing values have been substituted with the numerical value of ’0’. The mean, median, and/or mode are not applicable in this case as they would yield a positive result, when the desired outcome is negative. As an illustration, one column is labeled as: ’Do you currently have a partner in 2020?’ The available answer possibilities are 1 for ’yes’ and 2 for ’no’. It is illogical to obtain a decimal number as an output. In addition, removing the rows containing missing values was not feasible due to the excessive number of missing values in the 175 columns. This may result in the loss of crucial information. In addition, the train set consists of 987 rows, and removing rows could result in a more unbalanced dataset. In order to get an accurate prediction, it is necessary to utilize all 987 rows in the final merged dataset.

3.2.10	Feature Importance

The three original datasets all combined, have a total of 31667 columns. Based on literature review, 327 predictors will be utilized for model training, as described in section 3.2.3 and 3.2.5. Nevertheless, as stated by Ng (1998) and Speiser et al. (2019), it is essential to utilize a limited number of features during the model training process. An excessive number of features can result in errors and a deceptive output. Feature selection is a process that eliminates redundant features and identifies the significant ones, hence improving prediction results. Feature importance from the Random Forest Classifier will be utilized to determine the relevance of all 327 features by checking their feature importance scores. In the research and in previous studies, it showed that the predictors such as ’income’, ’birth year’, ’education’, ’partnership status’ ’partnership satisfaction’, ’financial status’ and ’child desire were indicated as highly relevant features (section 2.4). In the end, a total of 179 features will be utilized for training the models. The target variable ’new_child’ and the non-predictor ’nomem_encr’ have been excluded.

3.3	Algorithms and Hyperparameter Tuning

This section provides an explanation of the selected classification models. The Naive Bayes model is utilized as the baseline model, while XGBoost and Random Forest models are utilized as the comparison models. Additionally, this section explains the hyperparameter tuning for each model.

3.3.1	Naive Bayes

Even though literature review did not show any relevant studies which Machine Learning Algorithm should be the baseline model, according to Sueno et al. (2020), Naive Bayes is well-known for its simplicity and efficiency, particularly in managing large datasets. Due to its computational efficiency and simple implementation, it is an appealing choice for solving binary classification and multi-class classification challenges. Thus, Naive Bayes was chosen as the baseline model.

3.3.2	XGBoost

XGBoost, abbreviation for ’eXtreme Gradient Boosting’, was developed by Chen and Guestrin (2016). XGBoost is well-known for utilizing and building decision trees as base learners, as demonstrated by the research conducted by J. Li et al. (2022) and Chen and Guestrin (2016). The trees are built in a gradual way that corrects the errors made by the prior trees. Implementing this approach may successfully reduce overfitting and significantly improve the prediction powers of the model. The study conducted by Chen and Guestrin (2016) showed that XGBoost is capable of efficiently processing huge datasets without consuming excessive memory. Considering that XGBoost has the ability to learn and correct errors in the prior trees, as well as mitigate overfitting, this model will be utilized for this research.

3.3.3	Random Forest

Leo Breiman (2001) developed the Random Forest algorithm. Random Forest is a robust approach that can be utilized to address both classification and regression problems. It is capable of carrying out both binary classification and multi-class classification. The algorithm operates by creating numerous random decision trees and aggregating their predictions through majority voting for classification problems. Random Forest is widely recognized for its simplicity, low demand for hyperparameter tuning, and offering insights into feature importance. Moreover, Random Forest is popular for its accuracy and the ability to handle small datasets with numerous features (Biau & Scornet, 2016; Breiman, 2001). Considering the characteristics, the Random Forest algorithm is well-suited for utilize in the research.

3.3.4	Hyperparameter Tuning

The GridsearchCV() method is utilized for hyperparameter tuning in order to optimize the accuracy of the classification task. GridsearchCV performs a grid search to identify the most optimal parameters. The process involves iterating through various combinations of cross validation and parameter adjustment to enhance the performance of the models. When the cross-validation parameter is not explicitly defined in the GridsearchCV() method, it will automatically utilize a default setup of 5-fold crossvalidation. When the user specifies the cross-validation parameter in the code, the GridsearchCV will not apply the default cross-validation setting, but instead applies the cross-validation parameter provided by the user. GridsearchCV is acknowledged for its effectiveness, while it suffers by its time-consuming process. It may not be efficient for algorithms that are capable of processing an enormous hyperparameter space (Isa et al., 2019). In this research, GridsearchCV will be implemented to identify the most suitable hyperparameters for Naive Bayes, XGBoost, and Random Forest.

Hyperparameter Tuning (Naive Bayes)

For Naive Bayes, the hyperparameters that were utilized, are ’alpha’ (0.001, 0.01, 0.1, 1, 10, 15, 20, 25) and ’binarize’ (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6). 

Hyperparameter Tuning (XGBoost)

Hyperparameter optimization can greatly enhance the performance of XGBoost, leading to a significant improvement in prediction accuracy. The hyperparameters ’max_depth’ (3, 4, 5, 6, 7), ’min_child_weight’ (1, 3, 5, 7, 9), ’learning_rate’ (0.01, 0.1, 0.2) and ’n_estimators’ (50, 100, 150) are utilized to train the XGBoost model.

Hyperparameter Tuning (Random Forest)

Despite the fact that Random Forest requires minimal hyperparameter tuning, the ’n_estimators’ (30, 40, 50, 60, 70, 80, 90), ’max_depth’ (None, 10, 20, 30), ’min_samples_split’ (2, 3, 4, 5, 10) and ’min_samples_leaf’ (1, 2, 3, 4) hyperparameters are implemented for the Random Forest model.

3.4	Experimental Setup

3.4.1	Experiment 1

The first experiment, which serves as the baseline experiment, involves training the Naive Bayes, XGBoost, and Random Forest models utilizing the imbalanced dataset. The target variable in the dataset exhibits a notable imbalance, with 191 responses indicating ’yes’ and 697 responses indicating ’no’. Experiment 1 seeks to determine the impact, if any, of the class imbalance. Additionally, it is crucial to determine whether experiment 2 and experiment 3 would improve or worsen the performances.

3.4.2	Experiment 2

Experiment 2 involves applying the technique of re-sampling to balance the class distribution before training the algorithms. Re-sampling is a frequently utilized preprocessing technique that includes modifying the distribution of classes (Kaur et al., 2019). In this research, oversampling has been preferred over undersampling since oversampling avoids the potential loss of significant instances. Oversampling contains of two distinct approaches: Random over-sampling and Synthetic Minority Over-sampling Technique (SMOTE). Random over-sampling involves duplicating existing samples of the minority class to achieve a more balanced ratio across both classes (Kaur et al., 2019). SMOTE, on the other hand, involves creating synthetic samples of the minority class by increasing existing samples (Bajer et al., 2019). Both approaches are utilized to equalize the distribution of the class. SMOTE has been chosen over Random over-sampling because it introduces greater variation into the dataset by creating new synthetic samples instead of merely replicating existing ones. This diversity contributes to the development of a more robust model that can effectively generalize to unknown data. Furthermore, SMOTE can mitigate the potential risk of overfitting by generating new observation points, whereas Random over-sampling can result in overfitting due to the duplication of the same samples. Moreover, SMOTE is commonly acknowledged as the foremost oversampling technique due to its simplicity and proven effectiveness (Bajer et al., 2019). Consequently, due to the greater diversity in the dataset and the mitigation of overfitting risks, utilizing SMOTE could result in improved model performance, while Random over-sampling may negatively impact the model’s performance (Kaur et al., 2019). SMOTE was chosen as the resampling method to equalize the classes because of its effectiveness, and will be implemented in the models utilized in experiment 1.

3.4.3	Experiment 3

The objective of the third experiment is to integrate all the optimal parameters into the models from experiment 1 and experiment 2 in order to assess if utilizing these parameters would improve the performance of the models or not. GridsearchCV() is utilized to discover the best parameters for each model by performing hyperparameter tuning along with the cross-validation parameter. The default setting for the crossvalidation parameter is a 5-fold cross-validation.

SMOTE must only be applied to the training set and not to the validation set. Applying SMOTE to both the training and validation sets may lead to inaccurate outcomes and predictions. Moreover, this results in that the models have prior exposure and knowledge of the validation subset, potentially resulting in data leakage. In order to prevent any data leakage during the implementation of the GridsearchCV technique, a pipeline will be utilized with SMOTE in it. The pipeline with SMOTE will be applied to only models where SMOTE is applied (Sigurdardóttir, 2021). Firstly, the dataset will be divided into a training set and a validation set. Subsequently, the pipeline with SMOTE will be implemented. When utilizing GridsearchCV() and incorporating the pipeline with SMOTE, SMOTE will be applied individually to each training fold throughout the cross-validation process. Therefore, the validation set or unseen dataset will be remain unseen.

3.5	Evaluation Methods

3.5.1	Performance Metrics

The objective of performance metrics in binary classification tasks is to accurately predict one of two potential classes and evaluate machine learning model performance across both. Once deficiencies are identified, improvements can be implemented through resampling, threshold adjustments, and hyperparameter tuning (Sokolova & Lapalme, 2009).

Research by Wibowo and Fatichah (2021) indicates that the accuracy metric may not be suitable for imbalanced datasets. Therefore, this research primarily focuses on the F1-score as the main performance metric. This is also the primary goal of the PreFer challenge, which aims to achieve the highest F1 score on the test set. However, other performance metrics such as accuracy, recall and precision, will also be considered and discussed. Additionally, the confusion matrix will be utilized to provide a detailed evaluation of how well the classification model performs on unseen data.

3.5.2	Out-of-Sample Evaluation

Performance Metric on Train Set

In order to comprehend the behavior of the models, it is crucial to analyze and evaluate their performance on the training set. Overfitting occurs when a model performs exceptionally well on the training set but performs poorly on an unseen dataset (H. Li et al., 2024). Models may be biased. The objective is to ensure that the performance metric of both the training set and validation should be comparable in order to prevent overfitting. This demonstrates that the model has acquired patterns that can be applied to data that it has not previously seen. 

K-fold Cross-Validation

This data challenge/research only has access to the train set, which is split into a training and validation sets (section 3.2.1). Training models with a restricted number of data samples might be a challenge. K-fold cross-validation is utilized to evaluate the models’ generalizability, reduce bias, and ensure model reliability. K-fold cross-validation (specifically 5-fold cross-validation) is applied to the entire dataset to be utilized for both training and validation (Wong & Yeh, 2019). 

The 5-fold cross-validation is chosen since the test set, which accounts for 30% of the data, is not accessible. This method ensures that the training and validation split, which accounts for 70% of the data, is able to generalize well to new data and/or a test set. K-fold cross-validation assesses the performance of models on several subsets of the data to determine their effectiveness.

4	Results

This section provides detailed information regarding the results of experiment 1 (section

4.1.1), experiment 2 (section 4.1.2) and experiment 3 (section 4.1.3) conducted on both the train and validation sets utilizing 181 columns for the third submission. Error analysis per class and the disparate group analysis can be read in section 4.2. In addition, first and second submissions for the PreFer data challenge were made on Github and on the ’Next’ page throughout the research period (section 4.3). Hence, this section contains the results of the test set specifically the results of the first submission (section 4.3.2), and the results of the second submission (section 4.3.4). Information about the third submission (deadline: June 3rd 2024) can be read in section 4.3.5.

4.1	Results: Experiment 1, Experiment 2 and Experiment 3 with 181 columns

4.1.1	Results Experiment 1: Model Performance with Imbalanced Dataset on Training and Validation Sets

For the initial experiment, all three models, namely Naive Bayes, XGBoost, and Random Forest, were tuned utilizing an imbalanced dataset with 181 columns. The Naive Bayes model has an accuracy of 0.68. Nevertheless, XGBoost and Random Forest generate better results, achieving accuracies of roughly 0.89 and 0.92, respectively. Furthermore, Naive Bayes has a precision of 0.35, XGBoost has a precision of 0.71, and Random Forest has the highest precision among the three models, with a value of 0.84. The recall values for Naive Bayes, XGBoost, and Random Forest are 0.62, 0.81, and 0.76, respectively. The results reveal a significant enhancement in prediction when utilizing XGBoost and Random Forest in comparison to Naive Bayes. The F1-scores for the classifiers are as follows: 0.45 for Naive Bayes, 0.76 for XGBoost, and 0.80 for Random Forest. According to the performance metrics, it is evident that both the XGBoost and Random Forest models far exceed the baseline model Naive Bayes in terms of accuracy and all other measured performance metrics. The results illustrate the success of these models in managing the imbalanced dataset. Based on the aforementioned results for accuracy, precision, recall, and F1-score in experiment 1, it is evident that the Random Forest model performs the best with an imbalanced dataset, closely followed by XGboost.

4.1.2 Results Experiment 2: Model Performance with SMOTE on Training and Validation Sets

In the second experiment, SMOTE was applied to all models from the first experiment. The introduction of SMOTE did not significantly influence the dynamics within the models. Naive Bayes’ accuracy dropped to 0.56, while XGBoost and Random Forest maintained their accuracies around 0.90. With SMOTE, precision decreased: Naive Bayes to 0.26, XGBoost to 0.68, and Random Forest to 0.81, indicating all models faced more difficulties in accurately predicting positive cases. XGBoost’s recall remained unchanged at 0.81, whether with SMOTE or without. In contrast, Random Forest’s recall increased to 0.81, while Naive Bayes’ fell to 0.57. The F1-scores confirm these findings: XGBoost scored 0.74, Random Forest 0.81, and Naive Bayes only 0.35.
Overall, the utilization of SMOTE had an effect. Random Forest showed robust and effective handling of the minority class, confirming its suitability for situations that demand careful managing of the class imbalances. XGBoost showed a slightly change in the metrics, but it handles the minority class effectively as well. This analysis highlights the importance of utilizing SMOTE to guarantee equitable and precise predictions across applications. Therefore, it could be said that among the models with (and without) SMOTE, Random Forest performed the best.

4.1.3 Results Experiment 3: Model Performance with SMOTE and Hyperparameter Tuning on Training and Validation Sets

In the third experiment, hyperparameter tuning was applied to all the trained models generated in the first and second experiment. The optimal hyperparameters per model are as follows: Naïve Bayes with imbalanced data and hyperparameter tuning, the optimal hyperparameters for alpha is 15 and binarize is 0.0. Naïve Bayes with SMOTE and hyperparameter tuning, the optimal hyperparameters for alpha is 15 and binarize is 0.0. 

XGBoost with imbalanced data and hyperparameter tuning, the optimal hyperparameters for learning_rate is 0.1, max_depth is 5, min_child_weight is 1 and n_estimators is 100. XGBoost with SMOTE and hyperparameter tuning, the optimal hyperparameters for learning_rate is 0.3, max_depth is 3, min_child_weight is 1 and n_estimators is 50.

Random Forest with imbalanced data and hyperparameter tuning, the optimal hyperparameters for max_depth is 20, min_samples_leaf is 1, min_samples_split is 4 and n_estimators is 40. Random Forest with SMOTE and hyperparameter tuning, the optimal hyperparameters for	max_depth is None, min_samples_leaf is 1, min_samples_split is 4 and n_estimators: 50.

Hyperparameter tuning has been applied to all the Naive Bayes models, resulting in both modest improvements and declines. The Naive Bayes model trained on an imbalanced dataset with hyperparameter tuning achieved an accuracy of 0.68, a precision of 0.35, a recall of 0.62, and an F1-score of 0.45. This model showed better results compared to the Naive Bayes with SMOTE and hyperparameter tuning, which achieved an accuracy of 0.56, a precision of 0.27, a recall of 0.62, and an F1-score of 0.37. The Naive Bayes model with SMOTE and hyperparameter tuning exhibited a low precision value of 0.26, indicating difficulty in correctly predicting positive classes. However, the recall value of 0.62 is acceptable, suggesting that the model is relatively effective at identifying relevant cases. Among the four trained models, the Naive Bayes model with the imbalanced dataset performs the best.

The XGBoost model, when applied to an imbalanced dataset and with hyperparameter tuning, achieved the highest accuracy (0.91), precision (0.80), recall (0.76), and f1-score (0.78). However, despite using XGBoost with SMOTE and optimizing the hyperparameters, it yielded the worst outcomes among the four models. The accuracy score is 0.86, the precision score is 0.63, the recall score is 0.81, and the F1-score is 0.71. Nevertheless, the decrease in precision and maintaining the overall F1-score in the ’XGBoost model with SMOTE’ and ’XGBoost model with SMOTE and hyperparameter tuning’ indicate a possible overfitting to the minority class. This could be explained by the model becoming excessively focused on the features of the minority class, thus affecting its ability to be applied to other cases.

After the implementation of hyperparameter tuning, the ’Random Forest with SMOTE and hyperparameter tuning’ model demonstrates better results (accuracy: 0.91, precision: 0.80, recall: 0.76, and F1-score: 0.78), in comparison to the ’Random Forest with imbalanced dataset and hyperparameter tuning’ model (accuracy: 0.90, precision: 0.79, recall: 0.71, and F1-score: 0.75). This enhancement highlights the success of SMOTE in equalizing the distribution of classes, which generally improves the model’s ability to detect the minority class. However, the ’Random Forest with SMOTE’ model demonstrates the highest performance among the four models, achieving an accuracy of 0.92, precision of 0.81, recall of 0.81, and F1-score of 0.81. This indicates that the first implementation of SMOTE successfully dealt with the issue of class imbalance, improving both precision and recall without requiring any modifications such as hyperparameter tuning. In contrast to other Random Forest models, where hyperparameter tuning has been utilized to optimize performance by fine-tuning the parameters, in this case, the application of SMOTE alone resulted in the most balanced improvement across all performance metrics. In conclusion, Random Forest with SMOTE performed the best comparing to Naive Bayes and XGBoost, and their algorithms with combination of SMOTE and hyperparameter tuning.

4.2	Error Analysis and Disparate Group Analysis

Based on the outcomes of all three experiments, it is evident that Random Forest with SMOTE achieved the highest outcome. Therefore, it is desirable to conduct an in-depth error analysis and disparate group analysis with the Random Forest model including SMOTE.

Error Analysis per Class

The outcomes of the validation set, which consisted of 99 samples. The True Positive (TP - upper left) indicates that there were 17 samples that were true ’yes’ and were correctly predicted as ’yes’. The False Negative (FN - upper right) indicates that there are 4 true ’yes’ samples, but they were incorrectly predicted as ’no’. The False Positive (FP - lower left) indicates that there are 4 samples where the actual samples are ’no’, but they were incorrectly predicted as ’yes’. The True Negative (TN lower right) indicates that there are 74 true samples classified as ’no’ and the model correctly predicted them as ’no’ as well. 

The evaluation of the metrics for each class is as follows: The precision for class 0 is 0.95, signifying that 95% of the samples were correctly predicted as ’no’ and were indeed ’no’. The precision for class 1 is 0.81, meaning that 81% of the samples were correctly predicted as ’yes’ when they were actually ’yes’. The recall for class 0 is 0.95, indicating that the model correctly predicted 95% of the samples where the actual class was ’no’. The recall for class 1 is 0.81, implying that the model correctly predicted 81% of the samples when the actual class was ’no’. The F1-score for class 0 (0.95) and the F1-score for class 1 (0.81) represent the overall harmonic mean of precision and recall.

The Random Forest model properly classifies 74 out of 78 ’no’ responses. The ’yes’ responses were accurately identified in 17 out of 21 responses, with comparable results. The proportional distribution of errors, 4 misclassifications in both false positives and false negatives, suggests a well-balanced performance in both classes, which is typically difficult to achieve in datasets with imbalanced dataset, even with the use of SMOTE.

Disparate Group Analysis

In the disparate group analysis, separate analysis was conducted for each gender, specifically male and female. Upon further analysis per gender, it is evident that males have a higher accuracy score (0.93) compared to females (0.89). Males have a higher score in precision, scoring 0.88, in comparison to females who score 0.75. Nevertheless, the recall metrics for males and females are nearly identical, with a value of 0.78 for males and 0.75 for females. The f1-score for males is 0.82, whereas for females it is 0.75, suggesting that males exhibit a more desirable balance between precision and recall. The data demonstrates that the model is more reliability in accurately predicting positive cases for males than for females.

The outcomes of the validation set for males, which consisted of 43 samples. The True Positive (TP - upper left) indicates that there were 7 samples that were ’yes’ and were correctly predicted as ’yes’. The False Negative (FN - upper right) indicates that there are 2 ’yes’ samples, but they were incorrectly predicted as ’no’. The False Positive (FP - lower left) indicates that there are 1 samples where the actual samples are ’no’, but they were incorrectly predicted as ’yes’. The True Negative (TN - lower right) indicates that there are 33 samples classified as ’no’ and the model correctly predicted them as ’no’ as well.

The outcomes of the validation set for females, which consisted of 56 samples. The True Positive (TP) indicates that 9 samples correctly identified as ’yes’. The False Negative (FN) shows that 3 ’yes’ samples were incorrectly predicted as ’no’. The False Positive (FP) reveals that 3 ’no’ samples were incorrectly predicted as ’yes’. The True Negative (TN) confirms that 41 samples were accurately classified as ’no’ and also predicted as ’no’. 

4.3	Submission and Results on Test Set (PreFer Data Challenge)

PreFer Data Challenge Submission Criteria

In order to meet the submission criteria of the PreFer data challenge, it is necessary to submit a total of 4 files prior to the deadline. Three coding files must be uploaded to Github for the purpose of evaluating their compatibility with the test set (hold out data). To avoid hardcoding, it is required to arrange the code into three functions: clean_df, train_save_model, and predict_outcomes. This approach allows for a more efficient and modular implementation of the codes. The function clean_df should contain the code for preprocessing, data cleaning, and merging datasets. The function train_save_model is accountable for saving the trained model. Once the function train_save_model has been saved, the function that is created will be saved to a file named ’model.joblib’. The function predict_outcomes is utilized to predict whether individuals will have a child or not. In the aforementioned function, the file ’model.job’ will be read and the saved model inside it will be utilized for making predictions. Once all the files have been verified for compatibility on the test set, the final Github link with the latest revisions should be uploaded on the ’Next’ platform in order to obtain results on the test set.

4.3.1	First Submission

On April 30th 2024, the first submission on the ’Next’ page took place. For the first submission, the following preparations and coding were carried out: A total of 226 variables were selected for training the models. All preprocessing and data cleaning tasks, as described in sections 3.2.3 to 3.2.10, have been completed. Ultimately, XGBoost with SMOTE and hyperparameter tuning was chosen for the first submission because it shows the best performance compared to Naïve Bayes and Random Forest.

4.3.2	First Submission Results

On May 1st 2024, the result of the first submission of the model’s performance on the test set was made available online. For all the four metrics, the results were satisfactory, but there is still potential for improvement in the second submission.

4.3.3	Second Submission

The first submission gave satisfactory results, but there is room for improvement in the second submission. Since the XGBoost algorithm already incorporates SMOTE and hyperparameter tuning, it was decided to enhance the model by adding additional features. There was awareness, that it could improve or weaken the model’s performance by incorporating an excessive number of features. For this instance, a total of 299 variables were chosen. Upon comparing the first submission with the second submission, it is noticeable that there are an additional 73 variables in the second version. Its model approach has remained unchanged in comparison to the first submission. There is confidence that utilizing XGBoost with SMOTE and hyperparameter tuning would result in good or potentially better performance on the test set. 

4.3.4	Second Submission Results

On May 15th 2024, the results of the second submission of the model’s performance on the test set were published online. Overall, the performance of XGBoost is considered reasonably satisfactory, but it might be argued that the addition of more features led to a somewhat diminished model performance when comparing results from both submissions. The decrease in performance could be attributed to the insufficient relevance of the added features and/or an inappropriate ratio between the number of features (299) and the total number of rows in the training data (888). This could potentially lead to overfitting the model.

4.3.5	Third submission

The deadline for the third submission is set for June 3rd 2024. Upon adding an additional 73 features in the second submission, the test set performance deteriorated. However, the findings from the three experiments (section 4.1.1, 4.1.2 and 4.1.3) indicated that utilizing a smaller number of features (181) resulted in improved performance of the model compared to using 226 or 299 features (section 4.3.1 and 4.3.3). Reducing the number of features helps to mitigate overfitting. Despite both the first and second submissions being submitted utilizing XGBoost with SMOTE and hyperparameter tuning, Random Forest with SMOTE demonstrated greater performance utilizing 181 features. The Random Forest algorithm will be considered in the third submission because to its better results compared to the other two machine learning algorithms.

5	Discussion, Limitations and Approach on Large Register Data

5.1	Discussion

The objective of this data challenge and this research is to explore the current state of predicting fertility intentions, to enhance our awareness of fertility behaviour, and to obtain more understanding regarding important features and predictive machine learning models that can explain fertility intentions.

Among the three experiments conducted utilizing Naive Bayes, XGBoost, and Random Forest, the findings indicate that the utilization of various models and techniques, including the incorporation of SMOTE and hyperparameter tuning, impacts the accuracy, precision, recall, and F1-score of different algorithms. In conclusion, the findings of this research suggest that the Random Forest model, especially when combined with SMOTE, shows the best results. XGBoost performed well but was second in terms of performance. Similarly, M. Li and Xu (2022) found that Random Forest slightly outperforms XGBoost. However, the distinguishing factor between the analysis by M. Li and Xu (2022) and this research is the number of features incorporated. M. Li and Xu is less likely to overfit when utilizing 11 predictors, whereas this research applies a total of 181 features, increasing the risk of overfitting.

Moreover, the error analysis will point out the distinctions in its occurrence across different classes. Out of a sample size of 99, 74 out of 78 were accurately predicted as ’no’, whereas 17 out of 21 samples were correctly predicted as ’yes’. Although the performance metric for class 0, which is the majority class, displays better results, the performance metric for class 1, also known as the minority class, remains strong when SMOTE is utilized. Random Forest with SMOTE may obtain a high overall accuracy of 92% for both classes, while also maintaining great performance in terms of precision and recall for both classes.

Additionally, conducting a disparate group analysis helps to determine whether there are any gender-based differences in accuracy. The accuracy for females is 89%, while males have a slightly higher accuracy of 93%. Previous research by Lappegård et al. (2022) has indicated that there is no visible gender disparity in their findings. However, Ahinkorah et al. (2021) suggest a gender difference in fertility desire, with males exhibiting a higher fertility desire, which could result in increased fertility intentions. This research also reveals a modest difference in fertility intentions between genders, namely for males.

In conclusion, accurate predictions can be achieved by carefully considering feature selection and the overall quantity of features. Having an excessive number of features could result in overfitting the model. Regarding the model’s performance, there are still instances of False Positive and False Negative in the predictions, indicating that the algorithms are not achieving accurate predictions. The performance of Naive Bayes has been worse when compared to XGBoost and Random Forest. SMOTE yielded inferior performance for the Naive Bayes and XGBoost models, but it demonstrated superior outcomes for the Random Forest model. Despite incorporating hyperparameter tuning into the best performed model Random Forest, there was no significant enhancement in its performance. In summary, machine learning algorithms have the potential and the power to effectively predict fertility intentions in the Netherlands. However, further improvement can be accomplished by conducting additional research.

5.2	Limitations

There are three limitations to this research. First, the test set was inaccessible for evaluating the performance of the models. By splitting the train set into train and validation sets, there is a decrease in the amount of data available for training and validation purposes. From the beginning of the data challenge till the deadline of the third submission, there were two intermediate submissions to evaluate on the test set. Significant modifications were made to the feature selection part during the training process without prior knowledge of the impact on the test set. Second, there has been limited research conducted on the prediction of fertility intentions. Obtaining papers on predicting fertility intentions proved to be challenging, perhaps leading to erroneous decisions in selecting appropriate machine learning methods and appropriate features to be utilized. Third, this research had to deal with an excessive number of features (31,657) compared to the total sample size in the training and validation sets (987). Deciding on the number of features to utilize was challenging because selecting too many features could lead to overfitting. Moreover, the model might not learn effectively due to the complexity introduced by too many features.

5.3	Approach on Large Register Data

The approach utilized in this research for predicting fertility intentions in the Netherlands could potentially be more effective when applied to larger register data. This expectation is based on the assumption that it would perform better with a bigger dataset with more samples. As mentioned in the Limitations section (5.2), the sample size of the small survey data was too limited. However, using larger register data provides a bigger sample size, which allows the model to learn the underlying patterns more effectively and reduces the risk of overfitting. Additionally, a larger dataset offers greater diversity, which helps the model to encounter and learn from a wide range of scenarios. This is particularly important when dealing with a large number of features. Achieving a balance between the number of features and samples is crucial in predictive modeling and can significantly impact the success of machine learning projects.

6	Conclusion

This study aimed to explore and to understand predictive models for fertility intentions, utilizing machine learning algorithms like Naive Bayes, XGBoost, and Random Forest. The findings reveal that Random Forest, especially when enhanced with SMOTE, outperforms others in accuracy, precision, recall, and F1-score. Therefore, Random Forest with SMOTE was subsequently applied to both class predictions and genderbased analysis. It demonstrated robust results across both applications, achieving notable scores in the metrics in distinguishing between classes (’no’ and ’yes’) and between genders, with males slightly outperforming females. These results underscore the effectiveness of machine learning in understanding and predicting fertility behaviors, offering insights into significant features and model behaviors. These insights affirm the utility of Random Forest in demographic research such as fertility intentions, showcasing its capability to handle complex predictive tasks effectively. This research contributes to the ongoing discourse on applying machine learning to fertility intentions studies, highlighting the approach on large register data, particularly in feature selection and model tuning to avoid overfitting and enhance prediction accuracy.

References

Ahinkorah, B. O., Seidu, A.-A., Budu, E., Agbaglo, E., Adu, C., Dickson, K. S., Ameyaw, E. K., Hagan Jr, J. E., & Schack, T. (2021). Which factors predict fertility intentions of married men and women? results from the 2012 niger demographic and health survey. PLoS One, 16(6), e0252281.

Ajzen, I., & Klobas, J. (2013). Fertility intentions: An approach based on the theory of planned behavior. Demographic Research, 29, pp. 203–232. Retrieved May 12, 2024, from https://www.jstor.org/stable/26348152

Bajer, D., Zonć, B., Dudjak, M., & Martinović, G. (2019). Performance analysis of smote-based oversampling techniques when dealing with data imbalance. 2019 International Conference on Systems, Signals and Image Processing (IWSSIP), 265–271. https://doi.org/10.1109/IWSSIP.2019.8787306

Beaujouan, E., & Berghammer, C. (2019). The gap between lifetime fertility intentions and completed fertility in europe and the united states: A cohort approach. Population Research and Policy Review, 38, 507–535.

Biau, G., & Scornet, E. (2016). A random forest guided tour. Test, 25, 197–227.

Birch Petersen, K., Hvidman, H. W., Sylvest, R., Pinborg, A., Larsen, E. C., Macklon, K. T., Andersen, A. N., & Schmidt, L. (2015). Family intentions and personal considerations on postponing childbearing in childless cohabiting and single women aged 35–43 seeking fertility assessment and counselling. Human Reproduction, 30(11), 2563–2574.

Brakel, M., Portegijs, W., & Hermans, B. (2020). Emancipatiemonitor 2020: En ze leefden nog lang...? Retrieved March 16, 2024, from https://digitaal.scp.nl/ emancipatiemonitor2020/en-ze-leefden-nog-lang

Breiman, L. (2001). Random forests. Machine learning, 45, 5–32.

Bylander, T. (2002). Estimating generalization error on two-class datasets using out-ofbag estimates. Machine learning, 48, 287–297.

Chen, T., & Guestrin, C. (2016). Xgboost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785–794. https://doi.org/10.1145/2939672.2939785

Ciritel, A.-A., De Rose, A., & Arezzo, M. F. (2019). Childbearing intentions in a low fertility context: The case of romania. Genus, 75(1), 4.

Hashemzadeh, M., Shariati, M., Mohammad Nazari, A., & Keramat, A. (2021). Childbearing intention and its associated factors: A systematic review. Nursing open, 8(5), 2354–2368.

Isa, S. M., Suwandi, R., & Andrean, Y. P. (2019). Optimizing the hyperparameter of feature extraction and machine learning classification algorithms. International Journal of Advanced Computer Science and Applications, 10(3), 69–76.

Kaur, H., Pannu, H. S., & Malhi, A. K. (2019). A systematic review on imbalanced data challenges in machine learning: Applications and solutions. ACM Comput. Surv., 52(4). https://doi.org/10.1145/3343440

Kim, T. (2023). The impact of working hours on pregnancy intention in childbearing-age women in korea, the country with the world’s lowest fertility rate. PloS one, 18(7), e0288697.

Lappegård, T., Kristensen, A. P., Dommermuth, L., Minello, A., & Vignoli, D. (2022). The impact of narratives of the future on fertility intentions in norway. Journal of Marriage and Family, 84(2), 476–493.

Li, H., Rajbahadur, G. K., Lin, D., Bezemer, C.-P., Ming, Z., et al. (2024). Keeping deep learning models in check: A history-based approach to mitigate overfitting. arXiv preprint arXiv:2401.10359.

Li, J., An, X., Li, Q., Wang, C., Yu, H., Zhou, X., & Geng, Y.-a. (2022). Application of xgboost algorithm in the optimization of pollutant concentration. Atmospheric Research, 276, 106238. https://doi.org/https://doi.org/10.1016/j.atmosres. 2022.106238

Li, M., & Xu, X. (2022). Fertility intentions for a second child and their influencing factors in contemporary china. Frontiers in Psychology, 13, 883317.

Miller, W. B. (2011a). Comparing the tpb and the t-d-i-b framework. Vienna Yearbook of Population Research, 9, 19–29. Retrieved May 12, 2024, from http://www. jstor.org/stable/41342799

Miller, W. B. (2011b). Differences between fertility desires and intentions: Implications for theory, research and policy. Vienna Yearbook of Population Research, 9, 75–98. Retrieved March 16, 2024, from http://www.jstor.org/stable/41342806

Muraina, I. (2022). Ideal dataset splitting ratios in machine learning algorithms: General concerns for data scientists and data analysts. 7th International Mardin Artuklu Scientific Research Conference, 496–504.

Mussino, E., Gabrielli, G., Ortensi, L. E., & Strozza, S. (2023). Fertility intentions within a 3-year time frame: A comparison between migrant and native italian women. Journal of International Migration and Integration, 24(Suppl 1), 233–260.

Ng, A. Y. (1998). On feature selection: Learning with exponentially many irreverent features as training examples [Doctoral dissertation, Massachusetts Institute of Technology].

Preis, H., Tovim, S., Mor, P., Grisaru-Granovsky, S., Samueloff, A., & Benyamini, Y. (2020). Fertility intentions and the way they change following birth-a prospective longitudinal study. BMC pregnancy and childbirth, 20, 1–11.

Riederer, B., Buber-Ennser, I., & Brzozowska, Z. (2019). Fertility intentions and their realization in couples: How the division of household chores matters. Journal of Family Issues, 40(13), 1860–1882.

Sigurdardóttir, S. A. (2021). Experimental research on a continuous integrating pipeline with a machine learning approach: Master thesis done in collaboration with electronic arts.

Sivak, E., Pankowska, P., Mendrik, A., Emery, T., Garcia-Bernardo, J., Hocuk, S., Karpinska, K., Maineri, A., Mulder, J., Nissim, M., & Stulp, G. (2024a). Combining the strengths of dutch survey and register data in a data challenge to predict fertility (prefer).

Sivak, E., Pankowska, P., Mendrik, A., Emery, T., Garcia-Bernardo, J., Hocuk, S., Karpinska, K., Maineri, A., Mulder, J., Nissim, M., & Stulp, G. (2024b). Combining the strengths of dutch survey and register data in a data challenge to predict fertility (prefer). Journal of Computational Social Science, 1–29.

Sivak, E., Pankowska, P., Mendrik, A., Emery, T., Garcia-Bernardo, J., Hocuk, S., Karpinska, K., Maineri, A., Mulder, J., Nissim, M., & Stulp, G. (2024c). The task, goal, and research questions. Retrieved March 16, 2024, from https:
//stulp.gmw.rug.nl/prefer/details/overview/1research_questions.html

Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. Information processing & management, 45(4), 427–437.

Speiser, J. L., Miller, M. E., Tooze, J., & Ip, E. (2019). A comparison of random forest variable selection methods for classification prediction modeling. Expert systems with applications, 134, 93–101.

Stulp, G. (2024). Kinderloosheid in nederland. Retrieved March 16, 2024, from https: //www.gertstulp.com/posts/posts/2023-09-13-childlessness

Sueno, H. T., Gerardo, B. D., & Medina, R. P. (2020). Multi-class document classification using support vector machine (svm) based on improved naıve bayes vectorization technique. International Journal of Advanced Trends in Computer Science and Engineering, 9(3).

Wibowo, P., & Fatichah, C. (2021). An in-depth performance analysis of the oversampling techniques for high-class imbalanced dataset. Register, 7(1), 63–71.

Wong, T.-T., & Yeh, P.-Y. (2019). Reliable accuracy estimates from k-fold cross validation. IEEE Transactions on Knowledge and Data Engineering, 32(8), 1586–1594.

Zhu, C., Yan, L., Wang, Y., Ji, S., Zhang, Y., & Zhang, J. (2022). Fertility intention and related factors for having a second or third child among childbearing couples in shanghai, china. Frontiers in public health, 10, 879672.



