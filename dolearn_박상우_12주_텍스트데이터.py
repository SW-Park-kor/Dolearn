
import pandas as pd
import numpy as np

# df = pd.read_csv('customer_data(filtered)_generated.csv', encoding = 'cp949')
# # print(df.head())

from konlpy.tag import Twitter
twitter = Twitter()

def tokenize(text):
    stems = []
    tagged = twitter.pos(text)
    for i in range (0, len(tagged)): 
        if (tagged[i][1]=='Noun' or tagged[i][1]=='Adjective') :
            stems.append(tagged[i][0])
    return stems

# tagged = twitter.pos(df['Review'][0])
# for i in range (0, len(tagged)): 
#         if (tagged[i][1]=='Noun') :
#             print(tagged[i])

from sklearn.feature_extraction.text import TfidfVectorizer

# text_data_list = df['Review'].astype(str).tolist()
# text_data_arr = np.array([''.join(text) for text in text_data_list])

vectorizer = TfidfVectorizer(min_df=2, tokenizer=tokenize, norm='l2')
# text_data = vectorizer.fit_transform(text_data_arr)

# df_tfidf = pd.DataFrame(text_data.A, columns=vectorizer.get_feature_names())
# print(df_tfidf.head())


#%%
# import matplotlib.pyplot as plt
# import seaborn as sns
# from matplotlib import font_manager, rc

# font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
# rc('font', family=font_name)

# g = sns.factorplot('SNS', data=df, kind='count', size=5)
# g.set_xlabels()

# g = sns.factorplot('Addr', data=df, kind='count', size=5)
# g.set_xticklabels(rotation=90)
# g.set_xlabels()

# g = sns.factorplot('Score', data=df, kind='count', size=5)
# g.set_xlabels()

# df = df.dropna(subset=['Score']) #점수가 없는 데이터 제거
# df.index = range(0,len(df))
# df['Score2'] = ''

# for i in range(0,len(df)) : # 1,2:bad, 3:normal, 4,5:good
#     if(df['Score'][i] < 3) :
#         df['Score2'][i] = 'bad'
#     elif (df['Score'][i] > 3) :
#         df['Score2'][i] = 'good'
#     elif (df['Score'][i] == 3) :
#         df['Score2'][i] = 'normal'
# print(df.head())

# g = sns.factorplot('Score2', data=df, kind='count', size=5)
# g.set_xlabels()

# df.to_csv('customer_data(filtered)_generated2.csv')


#%%






#####분석

df = pd.read_csv('customer_data(filtered)_generated2.csv', encoding='utf-8')
# df.head()

review_data = df['Review'].astype(str).tolist()
review_label = df['Score2'].astype(str).tolist()

trainset_size = int(round(len(review_data)*0.80))

x_train = np.array([''.join(data) for data in review_data[0:trainset_size]])
y_train = np.array([data for data in review_label[0:trainset_size]])
x_test = np.array([''.join(data) for data in review_data[trainset_size+1:len(review_data)]])
y_test = np.array([data for data in review_label[trainset_size+1:len(review_label)]])

X_train = vectorizer.fit_transform(x_train)
X_test = vectorizer.transform(x_test)

# df_per = pd.DataFrame(columns=['Classifier', 'F-Measure', 'Accuracy'])
# df_per

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB


nb_classifier = MultinomialNB().fit(X_train, y_train)
nb_pred = nb_classifier.predict(X_test)

# print('\n Confusion Matrix \n')
# print(confusion_matrix(y_test, nb_pred))
# print('\n Classification Report \n')
# print(classification_report(y_test, nb_pred))
# print('\n Accuracy \n')
# print(round(accuracy_score(y_test, nb_pred, normalize=True),2))


fm = round(f1_score(y_test, nb_pred, average='weighted'), 2)
ac = round(accuracy_score(y_test, nb_pred, normalize=True), 2)
df_per.loc[len(df_per)] = ['Naive Bayes', fm, ac]
print(df_per)


#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dt_classifier = DecisionTreeClassifier().fit(X_train, y_train)
dt_pred = dt_classifier.predict(X_test)

print('\n Confusion Matrix \n')
print(confusion_matrix(y_test, dt_pred))
print('\n Classification Report \n')
print(classification_report(y_test, dt_pred))
print('\n Accuracy \n')
print(round(accuracy_score(y_test, dt_pred, normalize=True),2))

fm = round(f1_score(y_test, dt_pred, average='weighted'), 2)
ac = round(accuracy_score(y_test, dt_pred, normalize=True), 2)
df_per.loc[len(df_per)] = ['Decison Tree', fm, ac]
# df_per


#Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train, y_train)
rf_pred = rf_classifier.predict(X_test)

print('\n Confusion Matrix \n')
print(confusion_matrix(y_test, rf_pred))
print('\n Classification Report \n')
print(classification_report(y_test, rf_pred))
print('\n Accuracy \n')
print(round(accuracy_score(y_test, rf_pred, normalize=True),2))
ac = round(accuracy_score(y_test, rf_pred, normalize=True), 2)
df_per.loc[len(df_per)] = ['Random Forest', fm, ac]
# df_per


#SVM
from sklearn.svm import LinearSVC
svm_classifier = LinearSVC().fit(X_train, y_train)
svm_pred = svm_classifier.predict(X_test)

print('\n Confusion Matrix \n')
print(confusion_matrix(y_test, svm_pred))
print('\n Classification Report \n')
print(classification_report(y_test, svm_pred))
print('\n Accuracy \n')
print(round(accuracy_score(y_test, svm_pred, normalize=True),2))

fm = round(f1_score(y_test, svm_pred, average='weighted'), 2)
ac = round(accuracy_score(y_test, svm_pred, normalize=True), 2)
df_per.loc[len(df_per)] = ['Support Vector Machine', fm, ac]
df_per


df_per_1 = df_per.set_index('Classifier')
df_per_1
ax = df_per_1[['F-Measure','Accuracy']].plot(kind='bar', title ='Performance', figsize=(10, 7), legend=True, fontsize=12)
ax.set_xlabel('Classifier', fontsize=12)
plt.show()
