from flask import Flask, render_template, send_from_directory, url_for ,request,redirect
from flask_uploads import UploadSet, configure_uploads, IMAGES
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField
from flask_wtf import FlaskForm
import pandas as pd
import random
import os
from PIL import Image
import pytesseract
import torch
from sentence_transformers import SentenceTransformer, losses, models, util
import re
from konlpy.tag import Okt
import numpy as np
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from werkzeug.utils import secure_filename
import pymysql
import mysql.connector
from datetime import datetime


app = Flask(__name__)
app.secret_key = "0000"

# DB 연결
host='localhost',
user='ysh',
password='tndgus12!',
database='newsdb'

df = pd.read_csv('./finish_df.csv',index_col=0)
df = df.reset_index(drop=True)
app.config['UPLOAD_FOLDER'] = './uploads/'  # 업로드된 파일을 저장할 폴더 경로
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'gif'}  # 허용되는 파일 확장자
app.config['SECRET_KEY'] = 'eco' # secret_key 생성
# 현재 파일의 디렉토리 경로를 가져옴
base_dir = os.path.abspath(os.path.dirname(__file__))
print(base_dir)
# 업로드된 사진을 저장할 폴더 경로 설정
upload_folder = os.path.join(base_dir, 'uploads')
print(upload_folder)# 수정된 경로 설정
os.makedirs(upload_folder, exist_ok=True)  # 폴더가 없을 경우 생성

app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'
photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)

class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos, 'Only images are allowed'),
            FileRequired('File field should not be empty')
        ]
    )
    submit = SubmitField('Upload')

@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTO_DEST'], filename)

def get_random_word():
    words = ['가상화폐 - 컴퓨터 등에 정보 형태로 남아 실물 없이 사이버상으로만 거래되는 전자화폐의 일종',
        '불성실공시 - 공시제도는 기업으로 하여금 이해관계자(주주, 채권자, 투자자 등)를 위해 해당 기업의 재무내용 등 권리행사나 투자판단에 필요한 자료를 알리도록 의무화하는 제도이다. 상장. 등록법인이 공시의무를 성실히 이행하지 않는 것',
        '비트코인 - 비트코인은 암호화폐이자, 디지털 결제 시스템',
        '오퍼 - 매매당사자의 한편이 상대에게 가격조건을 적어 보내는 거래요청서',
        '청약 - 쌍방간의 청약 내용을 기록한 서식. 청약은 계약을 성립시키겠다는 의지의 표현',
        '도시계획 - 도시 내 사람들의 주거 및 각종 활동과 관련하여 공간의 배치 및 제도 및 규칙을 세우는 일']
    return random.choice(words)

@app.route('/')
def index():
    word = get_random_word()
    contents={
        'news_1_title':df['title'][0],
        'news_1':df['content'][0],
        'news_1_date':df['date'][0],
        'news_2_title':df['title'][1],
        'news_2':df['content'][1],
        'news_2_date':df['date'][1],
        'news_3_title':df['title'][2],
        'news_3':df['content'][2],
        'news_3_date':df['date'][2],
        'news_4_title':df['title'][3],
        'news_4':df['content'][3],
        'news_4_date':df['date'][3],
        'news_5_title':df['title'][4],
        'news_5':df['content'][4],
        'news_5_date':df['date'][4],
        
        'random_word': word
        }

    return render_template('index.html',contents = contents)



@app.route('/newswith6/<title>')
# main기사와 6개 유사 기사를 render
def newswith6(title):
    title_i = 0
    for i in range(len(df)):
        if df['title'][i] == title:
            title_i = i
            break
    print(title,title_i)
    new_contents = {'no':'ECONEXUS NEWS', 'news_title':df['title'][title_i], 'news_body':df['content'][title_i], 
    'news_date':df['date'][title_i], 'news_reporter':df['reporter'][title_i], 
    'news_company':df['store_name'][title_i], 'news_img':df['image_url'][title_i]}
    
    # 요약모델
    tokenizer = PreTrainedTokenizerFast.from_pretrained("ainize/kobart-news")
    model = BartForConditionalGeneration.from_pretrained("ainize/kobart-news")
    count = 0
    for a in re.finditer(" ", new_contents["news_body"]):
        count+=1
        place = a.start()
        if count > 400:
            break 
        news_text_limit = new_contents["news_body"][:place]
        
    input_ids = tokenizer.encode(news_text_limit, return_tensors="pt")
    
    summary_text_ids = model.generate(
    input_ids=input_ids,
    bos_token_id=model.config.bos_token_id,
    eos_token_id=model.config.eos_token_id,
    length_penalty=1.0, # 길이에 대한 penalty. 1보다 작은 경우 더 짧은 문장을 생성하도록 유도하며, 1보다 클 경우 길이가 더 긴 문장을 유도
    max_length=256,     # 요약문의 최대 길이 설정
    min_length=64,      # 요약문의 최소 길이 설정
    num_beams=2)         # 문장 생성시 다음 단어를 탐색하는 영역의 개수)
    
    summary_text =   tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
    # content 값을  모델에 넣고 본문 5개 나오면 그 본문에 맞는 row를 contents_2 에 대입 
    # 유사한 뉴스 불러오기
    model_name = "klue/roberta-base"
    embedding_model = models.Transformer  (model_name)
    pooler = models.Pooling(
        embedding_model.get_word_embedding_dimension(),
        pooling_mode_mean_tokens = True,
        pooling_mode_cls_token = False,
        pooling_mode_max_tokens = False,
    )
    model = SentenceTransformer(modules = [embedding_model, pooler])
    device = torch.device('cpu')
    model.load_state_dict(torch.load("./model_end.pt", map_location = device))
        
    #df = pd.read_csv("./finish_df.csv")
    document_embeddings = np.load("./model.npy")
    temp =[]
    for i in df.content:
        temp.append(i)
    
    query_data = new_contents["news_body"]
    query_data = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", " ", query_data) 
    pos_tagger = Okt()
    query_data = re.sub('.* 기자', "", query_data)
    pos = []
    tmp = [i[0] for i in pos_tagger.pos(query_data) if ((i[1]=="Noun")and(len(i[0])>1))]
    pos.append(" ".join(tmp))
    query_embedding = model.encode(pos[0])

    top_k = 15

        # 입력 문장 - 문장 후보군 간 코사인 유사도 계산 후,
    cos_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]

    # 코사인 유사도 순으로 `top_k` 개 문장 추출
    top_results = torch.topk(cos_scores, k=top_k)
    print(top_results)
    #print(f"입력 문장: {query_data[0]}")
    #print(f"\n<입력 문장과 유사한 {top_k} 개의 문장>\n")
    return_text=[]
    for i, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
        return_text.append(temp[idx])
    
    return_df =pd.DataFrame(columns={"title","date" , "content", "reporter", "image_url", "store_name", "label"})
    titles = []
    dates = []
    contents = []
    reporters=[]
    image_urls = []
    store_names = []
    labels = []
    for i in range(len(df)):
        for j in return_text:
            if  j== df.content[i]:
                #print(df.content[i])
                titles.append(df.title[i])
                dates.append(df.date[i])
                contents.append(df.content[i])
                reporters.append(df.reporter[i])
                image_urls.append(df.image_url[i])
                store_names.append(df.store_name[i])
                labels.append(df.label[i])
    new_df = pd.DataFrame({"title":titles ,"date" :dates, "content":contents, "reporter":reporters, "image_url":image_urls, "store_name":store_names, "label":labels})   
    dataf = pd.concat([return_df, new_df])
    dataf = dataf.loc[~dataf.content.duplicated(keep='last')]
    dataf =  dataf.reset_index()

    
    contents_2 = [
        {'no':1, 'news_title':dataf['title'][1], 'news_body':dataf['content'][1], 'news_date':dataf['date'][1], 'news_reporter':dataf['reporter'][1], 'news_company':dataf['store_name'][1], 'news_img':dataf['image_url'][1]},
        {'no':2, 'news_title':dataf['title'][2], 'news_body':dataf['content'][2], 'news_date':dataf['date'][2], 'news_reporter':dataf['reporter'][2], 'news_company':dataf['store_name'][2], 'news_img':dataf['image_url'][2]},
        {'no':3, 'news_title':dataf['title'][3], 'news_body':dataf['content'][3], 'news_date':dataf['date'][3], 'news_reporter':dataf['reporter'][3], 'news_company':dataf['store_name'][3], 'news_img':dataf['image_url'][3]},
        {'no':4, 'news_title':dataf['title'][4], 'news_body':dataf['content'][4], 'news_date':dataf['date'][4], 'news_reporter':dataf['reporter'][4], 'news_company':dataf['store_name'][4], 'news_img':dataf['image_url'][4]},
        {'no':5, 'news_title':dataf['title'][5], 'news_body':dataf['content'][5], 'news_date':dataf['date'][5], 'news_reporter':dataf['reporter'][5], 'news_company':dataf['store_name'][5], 'news_img':dataf['image_url'][5]},
        {'no':6, 'news_title':dataf['title'][6], 'news_body':dataf['content'][6], 'news_date':dataf['date'][6], 'news_reporter':dataf['reporter'][6], 'news_company':dataf['store_name'][6], 'news_img':dataf['image_url'][6]},
    ]
    return render_template('newswith6.html', summary_text=summary_text ,new_contents = new_contents, contents_2 = contents_2)


@app.route('/search' )
def search():
    filtered_df = df[df['title'].str.contains('전세사기')]
    filtered_df = filtered_df.reset_index(drop=True)
    print(filtered_df)
    return render_template('search.html' ,filtered_df=filtered_df)





@app.route('/finance')
def finance():
    fin = df[df['label']=='금융']
    fin = fin.reset_index(drop=True)
    new_contents =[
        {'no':1, 'news_title':fin.title[0], 'news_body':fin.content[0], 'news_date':fin.date[0], 'news_reporter':fin.reporter[0], 'news_company':fin.store_name[0], 'news_img':fin.image_url[0]},
        {'no':2, 'news_title':fin.title[1], 'news_body':fin.content[1], 'news_date':fin.date[1], 'news_reporter':fin.reporter[1], 'news_company':fin.store_name[1], 'news_img':fin.image_url[1]},
        {'no':3, 'news_title':fin.title[2], 'news_body':fin.content[2], 'news_date':fin.date[2], 'news_reporter':fin.reporter[2], 'news_company':fin.store_name[2], 'news_img':fin.image_url[2]},
        {'no':4, 'news_title':fin.title[3], 'news_body':fin.content[3], 'news_date':fin.date[3], 'news_reporter':fin.reporter[3], 'news_company':fin.store_name[3], 'news_img':fin.image_url[3]},
        {'no':5, 'news_title':fin.title[4], 'news_body':fin.content[4], 'news_date':fin.date[4], 'news_reporter':fin.reporter[4], 'news_company':fin.store_name[4], 'news_img':fin.image_url[4]},
        {'no':6, 'news_title':fin.title[5], 'news_body':fin.content[5], 'news_date':fin.date[5], 'news_reporter':fin.reporter[5], 'news_company':fin.store_name[5], 'news_img':fin.image_url[5]},
    ]
    return render_template('finance.html', new_contents = new_contents)

@app.route('/stock')
def stock():
    sto = df[df['label']=='증권']
    sto = sto.reset_index(drop=True)
    new_contents =[
        {'no':1, 'news_title':sto.title[0], 'news_body':sto.content[0], 'news_date':sto.date[0], 'news_reporter':sto.reporter[0], 'news_company':sto.store_name[0], 'news_img':sto.image_url[0]},
        {'no':2, 'news_title':sto.title[1], 'news_body':sto.content[1], 'news_date':sto.date[1], 'news_reporter':sto.reporter[1], 'news_company':sto.store_name[1], 'news_img':sto.image_url[1]},
        {'no':3, 'news_title':sto.title[2], 'news_body':sto.content[2], 'news_date':sto.date[2], 'news_reporter':sto.reporter[2], 'news_company':sto.store_name[2], 'news_img':sto.image_url[2]},
        {'no':4, 'news_title':sto.title[3], 'news_body':sto.content[3], 'news_date':sto.date[3], 'news_reporter':sto.reporter[3], 'news_company':sto.store_name[3], 'news_img':sto.image_url[3]},
        {'no':5, 'news_title':sto.title[4], 'news_body':sto.content[4], 'news_date':sto.date[4], 'news_reporter':sto.reporter[4], 'news_company':sto.store_name[4], 'news_img':sto.image_url[4]},
        {'no':6, 'news_title':sto.title[5], 'news_body':sto.content[5], 'news_date':sto.date[5], 'news_reporter':sto.reporter[5], 'news_company':sto.store_name[5], 'news_img':sto.image_url[5]},
    ]
    return render_template('stock.html', new_contents=new_contents)


@app.route('/property')
def property():
    sto = df[df['label']=='부동산']
    sto = sto.reset_index(drop=True)
    new_contents =[
        {'no':1, 'news_title':sto.title[0], 'news_body':sto.content[0], 'news_date':sto.date[0], 'news_reporter':sto.reporter[0], 'news_company':sto.store_name[0], 'news_img':sto.image_url[0]},
        {'no':2, 'news_title':sto.title[1], 'news_body':sto.content[1], 'news_date':sto.date[1], 'news_reporter':sto.reporter[1], 'news_company':sto.store_name[1], 'news_img':sto.image_url[1]},
        {'no':3, 'news_title':sto.title[2], 'news_body':sto.content[2], 'news_date':sto.date[2], 'news_reporter':sto.reporter[2], 'news_company':sto.store_name[2], 'news_img':sto.image_url[2]},
        {'no':4, 'news_title':sto.title[3], 'news_body':sto.content[3], 'news_date':sto.date[3], 'news_reporter':sto.reporter[3], 'news_company':sto.store_name[3], 'news_img':sto.image_url[3]},
        {'no':5, 'news_title':sto.title[4], 'news_body':sto.content[4], 'news_date':sto.date[4], 'news_reporter':sto.reporter[4], 'news_company':sto.store_name[4], 'news_img':sto.image_url[4]},
        {'no':6, 'news_title':sto.title[5], 'news_body':sto.content[5], 'news_date':sto.date[5], 'news_reporter':sto.reporter[5], 'news_company':sto.store_name[5], 'news_img':sto.image_url[5]},
    ]
    return render_template('property.html', new_contents=new_contents)

@app.route('/policy')
def policy():
    sto = df[df['label']=='경제일반/정책']
    sto = sto.reset_index(drop=True)
    new_contents =[
        {'no':1, 'news_title':sto.title[0], 'news_body':sto.content[0], 'news_date':sto.date[0], 'news_reporter':sto.reporter[0], 'news_company':sto.store_name[0], 'news_img':sto.image_url[0]},
        {'no':2, 'news_title':sto.title[1], 'news_body':sto.content[1], 'news_date':sto.date[1], 'news_reporter':sto.reporter[1], 'news_company':sto.store_name[1], 'news_img':sto.image_url[1]},
        {'no':3, 'news_title':sto.title[2], 'news_body':sto.content[2], 'news_date':sto.date[2], 'news_reporter':sto.reporter[2], 'news_company':sto.store_name[2], 'news_img':sto.image_url[2]},
        {'no':4, 'news_title':sto.title[3], 'news_body':sto.content[3], 'news_date':sto.date[3], 'news_reporter':sto.reporter[3], 'news_company':sto.store_name[3], 'news_img':sto.image_url[3]},
        {'no':5, 'news_title':sto.title[4], 'news_body':sto.content[4], 'news_date':sto.date[4], 'news_reporter':sto.reporter[4], 'news_company':sto.store_name[4], 'news_img':sto.image_url[4]},
        {'no':6, 'news_title':sto.title[5], 'news_body':sto.content[5], 'news_date':sto.date[5], 'news_reporter':sto.reporter[5], 'news_company':sto.store_name[5], 'news_img':sto.image_url[5]},
    ]
    return render_template('policy.html', new_contents=new_contents)


@app.route('/industry')
def industry():
    sto = df[df['label']=='산업/재계']
    sto = sto.reset_index(drop=True)
    new_contents =[
        {'no':1, 'news_title':sto.title[0], 'news_body':sto.content[0], 'news_date':sto.date[0], 'news_reporter':sto.reporter[0], 'news_company':sto.store_name[0], 'news_img':sto.image_url[0]},
        {'no':2, 'news_title':sto.title[1], 'news_body':sto.content[1], 'news_date':sto.date[1], 'news_reporter':sto.reporter[1], 'news_company':sto.store_name[1], 'news_img':sto.image_url[1]},
        {'no':3, 'news_title':sto.title[2], 'news_body':sto.content[2], 'news_date':sto.date[2], 'news_reporter':sto.reporter[2], 'news_company':sto.store_name[2], 'news_img':sto.image_url[2]},
        {'no':4, 'news_title':sto.title[3], 'news_body':sto.content[3], 'news_date':sto.date[3], 'news_reporter':sto.reporter[3], 'news_company':sto.store_name[3], 'news_img':sto.image_url[3]},
        {'no':5, 'news_title':sto.title[4], 'news_body':sto.content[4], 'news_date':sto.date[4], 'news_reporter':sto.reporter[4], 'news_company':sto.store_name[4], 'news_img':sto.image_url[4]},
        {'no':6, 'news_title':sto.title[5], 'news_body':sto.content[5], 'news_date':sto.date[5], 'news_reporter':sto.reporter[5], 'news_company':sto.store_name[5], 'news_img':sto.image_url[5]},
    ]
    return render_template('industry.html', new_contents=new_contents)


# OCR 실행 함수
def perform_ocr(image_path):
    pytesseract.pytesseract.tesseract_cmd = "./Tesseract-OCR/tesseract.exe"
    text = pytesseract.image_to_string(Image.open(image_path), lang='kor')
    text = text.replace(" ", "")
    query_data = text
    query_data = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", " ", query_data) 
    pos_tagger = Okt()
    query_data = re.sub('.* 기자', "", query_data)
    pos = []
    tmp = [i[0] for i in pos_tagger.pos(query_data) if ((i[1]=="Noun")and(len(i[0])>1))]
    pos.append(" ".join(tmp))
    pre_pos = pos[0]
    text = text.replace(" ", "")
    #print(text)
    return (text,pre_pos)

@app.route('/ocr', methods=['GET', 'POST'])
def ocr():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.photo.data)
        file_url = photos.url(filename)

        # OCR 코드 추가
        image_path = './uploads/' + filename  # 이미지 파일 경로
        result = perform_ocr(image_path)  # OCR 실행 함수 호출
        text =result[0]
        text1=result[1]
        # 요약모델
        tokenizer = PreTrainedTokenizerFast.from_pretrained("ainize/kobart-news")
        model = BartForConditionalGeneration.from_pretrained("ainize/kobart-news")
        # count = 0
        # news_text_limit =""
        # for a in re.finditer(" ", text1):
        #     count+=1
        #     place = a.start()
        #     if count > 400:
        #         break 
        #     news_text_limit += text1[:place]
        print(text)    
        input_ids = tokenizer.encode(text, return_tensors="pt")
        
        summary_text_ids = model.generate(
        input_ids=input_ids,
        bos_token_id=model.config.bos_token_id,
        eos_token_id=model.config.eos_token_id,
        length_penalty=1.0, # 길이에 대한 penalty. 1보다 작은 경우 더 짧은 문장을 생성하도록 유도하며, 1보다 클 경우 길이가 더 긴 문장을 유도
        max_length=256,     # 요약문의 최대 길이 설정
        min_length=32,      # 요약문의 최소 길이 설정
        num_beams=2)         # 문장 생성시 다음 단어를 탐색하는 영역의 개수)
        
        summary_text =   tokenizer.decode(summary_text_ids[0], skip_special_tokens=True)
        
        model_name = "klue/roberta-base"
        embedding_model = models.Transformer(model_name)
        pooler = models.Pooling(
            embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens = True,
            pooling_mode_cls_token = False,
            pooling_mode_max_tokens = False,
        )
        model = SentenceTransformer(modules = [embedding_model, pooler])
        device = torch.device('cpu')
        model.load_state_dict(torch.load("./model_end.pt", map_location = device))
            
        document_embeddings = np.load("./model.npy")
        temp =[]
        for i in df.content:
            temp.append(i)
        
        print("유사도들어가기전",":", text)    
        query_data = text
        query_data = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", " ", query_data) 
        pos_tagger = Okt()
        query_data = re.sub('.* 기자', "", query_data)
        pos = []
        tmp = [i[0] for i in pos_tagger.pos(query_data) if ((i[1]=="Noun")and(len(i[0])>1))]
        pos.append(" ".join(tmp))
        print("유사도 전처리전",":",pos[0])
        query_embedding = model.encode(pos[0])

        top_k =  15

            # 입력 문장 - 문장 후보군 간 코사인 유사도 계산 후,
        cos_scores = util.pytorch_cos_sim(query_embedding, document_embeddings)[0]

        # 코사인 유사도 순으로 `top_k` 개 문장 추출
        top_results = torch.topk(cos_scores, k=top_k)
        print("인덱스결과값",top_results)
        #print(f"입력 문장: {query_data[0]}")
        #print(f"\n<입력 문장과 유사한 {top_k} 개의 문장>\n")
        return_text=[]
        for i, (score, idx) in enumerate(zip(top_results[0], top_results[1])):
            return_text.append(temp[idx])
        
        return_df =pd.DataFrame(columns={"title","date" , "content", "reporter", "image_url", "store_name", "label"})
        titles = []
        dates = []
        contents = []
        reporters=[]
        image_urls = []
        store_names = []
        labels = []
        for i in range(len(df)):
            for j in return_text:
                if  j== df.content[i]:
                    #print(df.content[i])
                    titles.append(df.title[i])
                    dates.append(df.date[i])
                    contents.append(df.content[i])
                    reporters.append(df.reporter[i])
                    image_urls.append(df.image_url[i])
                    store_names.append(df.store_name[i])
                    labels.append(df.label[i])
        new_df = pd.DataFrame({"title":titles ,"date" :dates, "content":contents, "reporter":reporters, "image_url":image_urls, "store_name":store_names, "label":labels})   
        dataf = pd.concat([return_df, new_df])
        #dataf = dataf.loc[~dataf.title.duplicated(keep='last')]
        dataf =  dataf.reset_index()
        print("유사도뉴스df",":",dataf)
        #print(dataf)
        contents_2 = [
        {'no':1, 'news_title':dataf['title'][1], 'news_body':dataf['content'][1], 'news_date':dataf['date'][1], 'news_reporter':dataf['reporter'][1], 'news_company':dataf['store_name'][1], 'news_img':dataf['image_url'][1]},
        {'no':2, 'news_title':dataf['title'][2], 'news_body':dataf['content'][2], 'news_date':dataf['date'][2], 'news_reporter':dataf['reporter'][2], 'news_company':dataf['store_name'][2], 'news_img':dataf['image_url'][2]},
        {'no':3, 'news_title':dataf['title'][3], 'news_body':dataf['content'][3], 'news_date':dataf['date'][3], 'news_reporter':dataf['reporter'][3], 'news_company':dataf['store_name'][3], 'news_img':dataf['image_url'][3]},
        {'no':4, 'news_title':dataf['title'][4], 'news_body':dataf['content'][4], 'news_date':dataf['date'][4], 'news_reporter':dataf['reporter'][4], 'news_company':dataf['store_name'][4], 'news_img':dataf['image_url'][4]},
        {'no':5, 'news_title':dataf['title'][5], 'news_body':dataf['content'][5], 'news_date':dataf['date'][5], 'news_reporter':dataf['reporter'][5], 'news_company':dataf['store_name'][5], 'news_img':dataf['image_url'][5]},
        {'no':6, 'news_title':dataf['title'][6], 'news_body':dataf['content'][6], 'news_date':dataf['date'][6], 'news_reporter':dataf['reporter'][6], 'news_company':dataf['store_name'][6], 'news_img':dataf['image_url'][6]},
    ]

        return render_template('ocr.html', summary_text=summary_text,form=form, file_url=file_url, text=text , contents_2=contents_2)
    else:
        file_url = None
    return render_template('ocr.html', form=form, file_url=file_url)
    
#로그인
@app.route('/login', methods=['GET', 'POST'])
def login():
    # username이랑 password 받아오기
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        print("여기")
        
        
        # username이 None이거나 빈 문자열인 경우 다시 로그인 페이지로 리디렉션
        if not username:
            return redirect('/login')
        
        # 데이터베이스 연결 및 쿼리 실행
        connection = pymysql.connect(host='localhost',user='ysh',password='tmdgus12!',database='newsdb')
        cursor = connection.cursor()
        # db에서 select로 username으로 맞는게 있는지 확인
        query = f'select * from user where username="{username}" and password="{password}"'
        cursor.execute(query)
        # 결과값 result에 저장
        result=cursor.fetchall()
        connection.close()

        print("login len", result)
        # 만약에 유저가 존재하면 len이 1개이다. 그러면 홈페이지로 이동
        if len(result)==1:
            return redirect('/list')
        # db에 유저가 존재하지 않으면 일단 비번을 잘못 쳤다 생각하고 html에 alert으로 보여주기 위해서 msg를 보내준다.
        else:
            return render_template('login.html',msg='login error.')
        
    # GET 요청일 경우 회원가입 페이지 표시
    return render_template('login.html')


# 회원가입
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = request.form['name']
        
        # username이 None이거나 빈 문자열인 경우 다시 회원가입 페이지로 리디렉션
        if not username:
            return redirect('/register')
        
        # 데이터베이스 연결 및 쿼리 실행
        connection = pymysql.connect(host='localhost',user='ysh',password='tmdgus12!',database='newsdb')
        cursor = connection.cursor()
        query = "INSERT INTO user (username, password, name) VALUES (%s, %s, %s)"
        cursor.execute(query, (username, password, name))
        connection.commit()
        connection.close()
        
        # 회원가입 후 로그인 페이지로 이동
        return redirect('/login')
    
    # GET 요청일 경우 회원가입 페이지 표시
    return render_template('register.html')

@app.route('/write', methods=['GET', 'POST'])
def write():
     if request.method == 'POST':
        post_title = request.form['title']
        post_text = request.form['text']
        post_nickname = request.form['nickname']
        post_password = request.form['password']
        post_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        conn = pymysql.connect(host='localhost',user='ysh',password='tmdgus12!',database='newsdb')
        cursor = conn.cursor()
        query = "INSERT INTO board2 (title, nickname, password, text, date) VALUES (%s, %s, %s, %s, %s)"
        cursor.execute(query, (post_title, post_nickname, post_password, post_text, post_date))
        conn.commit()

        cursor.close()
        conn.close()
        return redirect('/list')
     return render_template('write.html')

@app.route('/list', methods=['GET', 'POST'])
def list():
    result = None  # 변수 초기화
    conn = pymysql.connect(host='localhost',user='ysh',password='tmdgus12!',database='newsdb')
    cursor = conn.cursor()
    query = "SELECT * FROM board2"
    cursor.execute(query)
    result = cursor.fetchall()
    cursor.close()
    conn.close()

    return render_template('list.html', board2=result)
def get_pymysql_connection():
    # MySQL 데이터베이스 연결 정보 설정
    # host = '127.0.0.1'
    # user = 'root'
    # password = '0000'
    # database = 'user'

    # pymysql을 사용하여 MySQL에 연결
    connection = pymysql.connect(
        host='localhost',
        user='ysh',
        password='tmdgus12!',
        database='newsdb',
        cursorclass=pymysql.cursors.DictCursor
    )
    return connection

@app.route('/view/<int:post_id>')
def view(post_id):
    # pymysql을 사용하여 MySQL에 연결
    connection = get_pymysql_connection()
    cursor = connection.cursor()

    # 쿼리 실행
    query = "SELECT * FROM board2 WHERE id = %s"
    cursor.execute(query, (post_id,))
    result = cursor.fetchone()
    print("result:", result)

    cursor.close()
    connection.close()

    return render_template('view.html', board2=result)

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=5000)