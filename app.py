import nltk
nltk.download('stopwords')
nltk.download('wordnet')

import pandas as pd
import streamlit as st
import docx
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from wordcloud import WordCloud, STOPWORDS

import warnings
warnings.filterwarnings("ignore")

#Tile
st.title("Resume Classification")

path = r'E:\Data Science\Datascience Intership Project\Project DS3\resum'

#Uploaded file
uploaded_file = st.file_uploader("Chose a file",type=['Docx','pdf'])
if uploaded_file is not None:



     #extract file path
     name = uploaded_file.name
     file_path = f"{path}\{name}"

     #Store file in local disk
     with open(file_path, "wb") as f:
         f.write(uploaded_file.getbuffer())

     #Read txt data from file
     resume = []
     doc = docx.Document(file_path)
     for para in doc.paragraphs:
         resume.append(para.text)
     resume = ' '.join(resume)
     st.subheader('Preview')
     st.write(resume)

     #Import data to train the model
     Data = pd.read_csv("data.csv",encoding='latin')

     #Created df for uploded file
     data = {'Resumes' : resume,
             'Category' : None}
     df = pd.DataFrame(data, index=[0])

     #Merge both df
     Data = pd.concat([Data,df], axis=0)
     Data.reset_index(drop=True, inplace=True)
     #Data = Data.drop(columns='Unnamed: 0',axis=1)

     Data['Cleaned_Resume'] = Data['Resumes'].str.replace("[a-zA-Z]"," ")

     Data['Cleaned_Resume'] = Data['Cleaned_Resume'].str.replace("http[^\s][^s]+"," ")

     Data['Cleaned_Resume'] = Data['Cleaned_Resume'].apply(lambda x: ' '.join(i.lower() for i in x.split()))

     Data['Cleaned_Resume'] = Data['Cleaned_Resume'].apply(lambda x: ' '.join([i for i in x.split() if i not in stopwords.words('english')]))

     lemma = WordNetLemmatizer()
     Data['Cleaned_Resume'] = Data['Cleaned_Resume'].apply(lambda x: ' '.join(lemma.lemmatize(i) for i in x.split()))

     freq = pd.Series(' '.join(Data['Cleaned_Resume']).split()).value_counts()
     least_freq = freq[freq.values == 1].index
     Data['Cleaned_Resume'] = Data['Cleaned_Resume'].apply(lambda  x: ' '.join(i for i in x.split() if i not in least_freq))

     text = ' '.join(Data['Cleaned_Resume'][77:])
     skills = ['html','css','jsx','react','javascript','git','node','npm','redux','rdbms','json','python','rest','graphql','swaagger',
               'gcp','mysql','sql','mssql','ssis','ssrs','ssas','rtl','oracle','spark','mangodb','apache','agile','jquery','scrum','plsql',
               'database','tsql','hcm','crm','anp','sqr','peoplesoft','peoplecode','hr','payroll','hrms','sdlc','sap','birt','forcasting',
               'word','excel','powerpoint','commmunication','problemsolving','analytical','debugging']



     tfidf = TfidfVectorizer()
     x = tfidf.fit_transform(Data['Cleaned_Resume'])
     x.toarray()
     tfidf.get_feature_names_out()

     df2 = pd.DataFrame(x.toarray(), columns=tfidf.get_feature_names_out())
     df2['Category'] = Data['Category']

     train = df2.iloc[:-1,:]
     le = LabelEncoder()

     train['Category'] = le.fit_transform(train['Category'])

     predict_data = df2.iloc[77:,:-1]

     xtrain = train.iloc[:,:-1]
     ytrain = train['Category']
     svc = SVC()

     svc.fit(xtrain,ytrain)

     pickle.dump(svc, open('final_svm_model.pkl','wb'))

     load = open('final_svm_model.pkl','rb')
     model = pickle.load(load)

     st.subheader('Check Category')
     if st.button('Check'):
          result = model.predict(predict_data)

          for r in result:
               if r == 0:
                    result = 'Peoplesoft'
               elif r == 1:
                    result = 'React JS Developer'
               elif r == 2:
                    result = 'SQL Developer'
               else:
                    result = 'Workday'

          st.success('Category: {}'.format(result))
