def total():
    import crawling
    import pandas as pd
    import crawling
    import torch
    from sentence_transformers import SentenceTransformer, losses, models, util
    import pandas as pd
    import numpy as np
    import re
    from konlpy.tag import Okt
    data = crawling.crawling(2)
    print("크롤링한 데이터",": ",len(data),data)
    df = pd.read_csv("./finish_df.csv",index_col=0)
    print(df.index)
    dataf = pd.DataFrame(columns={"title","date" , "content", "reporter", "image_url", "store_name", "label"})
    for i in range(len(data)):
        if data.title[i] not in df.title.values:
            title = data.title[i]
            date = data.date[i]
            content = data.content[i]
            reporter = data.reporter[i]
            image_url = data.image_url[i]
            store_name = data.store_name[i]
            label = data.label[i]
            dataf = dataf.append({"title": title, "date": date, "content": content, "reporter": reporter,
                                    "image_url": image_url, "store_name": store_name, "label": label}, ignore_index=True)
    dataf.drop_duplicates(subset=["title"], inplace=True)       
    dataf.reset_index(drop=True)
    print("크롤링한 중복제거 데이터",": ",len(dataf),dataf)
    
    last_df = pd.concat([dataf, df])
    last_df=last_df.reset_index(drop=True)
    last_df.to_csv("./finish_df.csv")
    print("중복제거 데이터와 원래 df 합친 ",": ",len(last_df))
    
    if len(dataf)>0:
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
        contents = []
        for i in dataf.content:
            contents.append(i)
        for i in contents:
            re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", " ", i) 
        pos=[]
        pos_tagger = Okt()
        for content in contents:
            content = re.sub('.* 기자', "", content)
            tmp = [i[0] for i in pos_tagger.pos(content) if ((i[1]=="Noun")and(len(i[0])>1))]
            pos.append(" ".join(tmp))

        for i in range(len(pos)):
            if len(list(pos[i].split()))>256:
                a = list(pos[i].split())[:256]
                b = " ".join(a)
                pos[i]=b
        print("모델에 들어갈 데이터",": ",len(pos))        
        document_embeddings = model.encode(pos)
        model_data = np.load("./model.npy")
        last_model_data= np.concatenate((document_embeddings,model_data),axis=0)
        np.save("model.npy",last_model_data)
        print(last_df.shape,last_model_data.shape)
        return total()
    else:
        return total()
            