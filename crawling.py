def crawling(page):
    import pandas as pd
    from bs4 import BeautifulSoup
    import requests
    dataf = pd.DataFrame(columns={"title","date" , "content", "reporter", "image_url", "store_name", "label"})
    # 금융
    for i in range(1,((page-1)*10)+1,10):
            
            url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=%EA%B8%88%EC%9C%B5&sort=1&photo=0&field=0&pd=0&ds=&de=&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:dd,p:all,a:all"+ "&start=" + str(i)
            response = requests.get(url, headers={'User-Agent':'mozilla/5.0 (windows nt 10.0; win64; x64) applewebkit/537.36 (khtml, like gecko) chrome/114.0.0.0 safari/537.36'})
            html = response.text
            soup = BeautifulSoup(html, "html.parser")
            articles = soup.select("div.info_group")
            for article in articles:
                links = article.select("a.info")
                if len(links) >= 2:
                    url = links[1].attrs["href"]
                    response = requests.get(url, headers={'User-Agent':'mozilla/5.0 (windows nt 10.0; win64; x64) applewebkit/537.36 (khtml, like gecko) chrome/114.0.0.0 safari/537.36'})      
                    html = response.text                                                   
                    soup = BeautifulSoup(html, "html.parser")

                    if "article" in response.url:
                            titles = []
                            dates = []
                            contents = []
                            reporters=[]
                            image_urls = []
                            store_names = []
                            labels = []
                            # 기사제목
                            title    = soup.select_one(".media_end_head_headline")
                            if title != None:
                                title = title.text.strip()
                            else:
                                title = "제목를 알수없음."
                            titles.append(title)
                            # 기사 본문 들고오기
                            content  = soup.select_one("#dic_area")
                            if content != None:
                                content  = content.text 
                            else:
                                content = "본문을 알수없음."
                            contents.append(content)
                            # 날짜 들고와서 수정날짜는 수정날짜대로 날짜 본문은 본문대로 들고오기
                            two_date =   soup.select_one("#ct > div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp")
                            if two_date != None:
                                date = two_date.text
                                dates1 = []
                                dates1.append(date)
                                for i in dates1:
                                    if len(i)>=33:
                                        for_date = i.split("\n\n\n")[1].split("정")[1].split("\n")[0]
                                        dates.append(for_date)
                                    else:
                                        for_date = i.split("력")[1].split("\n")[0]
                                        dates.append(for_date)
                            else:
                                dates.append("날짜를 알수없음.")
                            # 기자명 들고오기
                            reporter = soup.select_one("div#contents > div.byline > p.byline_p > span")                   
                            if reporter != None:
                                reporter = reporter.text.strip()
                            else:
                                reporter = "기자를 알수없음."
                            reporters.append(reporter)
                            # 이미지 url 들고오기
                            image = soup.select_one("meta[property='og:image']")
                            if image!= None:
                                image_url = image["content"]                               
                            else:
                                image_urls = "이미지를 알수없음."
                            image_urls.append(image_url)
                            # 신문사명 들고오기
                            store_name = soup.select_one("#contents > div.copyright > div > p")
                            if store_name !=None :
                                store_name = store_name.text.split(".")[0].split("ⓒ ")[1]                                
                            else:
                                store_name = "신문사명을 알수없음."
                            store_names.append(store_name)
                            
                            labels.append("금융")
                            df = pd.DataFrame({"title":titles ,"date" :dates, "content":contents, "reporter":reporters, "image_url":image_urls, "store_name":store_names, "label":labels})
                            dataf = pd.concat([dataf, df])
                            #print("합체완료")
    
    # 증권
    for i in range(1,((page-1)*10)+1,10): 
            url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=%EC%A6%9D%EA%B6%8C&sort=1&photo=0&field=0&pd=0&ds=&de=&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:dd,p:all,a:all"+ "&start=" + str(i)
            response = requests.get(url, headers={'User-Agent':'mozilla/5.0 (windows nt 10.0; win64; x64) applewebkit/537.36 (khtml, like gecko) chrome/114.0.0.0 safari/537.36'})
            html = response.text
            soup = BeautifulSoup(html, "html.parser")
            articles = soup.select("div.info_group")
            for article in articles:
                links = article.select("a.info")
                if len(links) >= 2:
                    url = links[1].attrs["href"]
                    response = requests.get(url, headers={'User-Agent':'mozilla/5.0 (windows nt 10.0; win64; x64) applewebkit/537.36 (khtml, like gecko) chrome/114.0.0.0 safari/537.36'})      
                    html = response.text                                                   
                    soup = BeautifulSoup(html, "html.parser")

                    if "article" in response.url:
                            titles = []
                            dates = []
                            contents = []
                            reporters=[]
                            image_urls = []
                            store_names = []
                            labels = []
                            # 기사제목
                            title    = soup.select_one(".media_end_head_headline")
                            if title != None:
                                title = title.text.strip()
                            else:
                                title = "제목를 알수없음."
                            titles.append(title)
                            # 기사 본문 들고오기
                            content  = soup.select_one("#dic_area")
                            if content != None:
                                content  = content.text 
                            else:
                                content = "본문을 알수없음."
                            contents.append(content)
                            # 날짜 들고와서 수정날짜는 수정날짜대로 날짜 본문은 본문대로 들고오기
                            two_date =   soup.select_one("#ct > div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp")
                            if two_date != None:
                                date = two_date.text
                                dates1 = []
                                dates1.append(date)
                                for i in dates1:
                                    if len(i)>=33:
                                        for_date = i.split("\n\n\n")[1].split("정")[1].split("\n")[0]
                                        dates.append(for_date)
                                    else:
                                        for_date = i.split("력")[1].split("\n")[0]
                                        dates.append(for_date)
                            else:
                                dates.append("날짜를 알수없음.")
                            # 기자명 들고오기
                            reporter = soup.select_one("div#contents > div.byline > p.byline_p > span")                   
                            if reporter != None:
                                reporter = reporter.text.strip()
                            else:
                                reporter = "기자를 알수없음."
                            reporters.append(reporter)
                            # 이미지 url 들고오기
                            image = soup.select_one("meta[property='og:image']")
                            if image!= None:
                                image_url = image["content"]                               
                            else:
                                image_urls = "이미지를 알수없음."
                            image_urls.append(image_url)
                            # 신문사명 들고오기
                            store_name = soup.select_one("#contents > div.copyright > div > p")
                            if store_name !=None :
                                store_name = store_name.text.split(".")[0].split("ⓒ ")[1]                                
                            else:
                                store_name = "신문사명을 알수없음."
                            store_names.append(store_name)
                            
                            labels.append("증권")
                            df = pd.DataFrame({"title":titles ,"date" :dates, "content":contents, "reporter":reporters, "image_url":image_urls, "store_name":store_names, "label":labels}) 
                            dataf = pd.concat([dataf, df])
    
    # 부동산
    for i in range(1,((page-1)*10)+1,10):
            
            url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=%EB%B6%80%EB%8F%99%EC%82%B0%20%EB%89%B4%EC%8A%A4&sort=1&photo=0&field=0&pd=0&ds=&de=&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:dd,p:all,a:all"+ "&start=" + str(i)
            response = requests.get(url, headers={'User-Agent':'mozilla/5.0 (windows nt 10.0; win64; x64) applewebkit/537.36 (khtml, like gecko) chrome/114.0.0.0 safari/537.36'})
            html = response.text
            soup = BeautifulSoup(html, "html.parser")
            articles = soup.select("div.info_group")
            for article in articles:
                links = article.select("a.info")
                if len(links) >= 2:
                    url = links[1].attrs["href"]
                    response = requests.get(url, headers={'User-Agent':'mozilla/5.0 (windows nt 10.0; win64; x64) applewebkit/537.36 (khtml, like gecko) chrome/114.0.0.0 safari/537.36'})      
                    html = response.text                                                   
                    soup = BeautifulSoup(html, "html.parser")

                    if "article" in response.url:
                            titles = []
                            dates = []
                            contents = []
                            reporters=[]
                            image_urls = []
                            store_names = []
                            labels = []
                            # 기사제목
                            title    = soup.select_one(".media_end_head_headline")
                            if title != None:
                                title = title.text.strip()
                            else:
                                title = "제목를 알수없음."
                            titles.append(title)
                            # 기사 본문 들고오기
                            content  = soup.select_one("#dic_area")
                            if content != None:
                                content  = content.text 
                            else:
                                content = "본문을 알수없음."
                            contents.append(content)
                            # 날짜 들고와서 수정날짜는 수정날짜대로 날짜 본문은 본문대로 들고오기
                            two_date =   soup.select_one("#ct > div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp")
                            if two_date != None:
                                date = two_date.text
                                dates1 = []
                                dates1.append(date)
                                for i in dates1:
                                    if len(i)>=33:
                                        for_date = i.split("\n\n\n")[1].split("정")[1].split("\n")[0]
                                        dates.append(for_date)
                                    else:
                                        for_date = i.split("력")[1].split("\n")[0]
                                        dates.append(for_date)
                            else:
                                dates.append("날짜를 알수없음.")
                            # 기자명 들고오기
                            reporter = soup.select_one("div#contents > div.byline > p.byline_p > span")                   
                            if reporter != None:
                                reporter = reporter.text.strip()
                            else:
                                reporter = "기자를 알수없음."
                            reporters.append(reporter)
                            # 이미지 url 들고오기
                            image = soup.select_one("meta[property='og:image']")
                            if image!= None:
                                image_url = image["content"]                               
                            else:
                                image_urls = "이미지를 알수없음."
                            image_urls.append(image_url)
                            # 신문사명 들고오기
                            store_name = soup.select_one("#contents > div.copyright > div > p")
                            if store_name !=None :
                                store_name = store_name.text.split(".")[0].split("ⓒ ")[1]                                
                            else:
                                store_name = "신문사명을 알수없음."
                            store_names.append(store_name)
                            
                            labels.append("부동산")
                            df = pd.DataFrame({"title":titles ,"date" :dates, "content":contents, "reporter":reporters, "image_url":image_urls, "store_name":store_names, "label":labels})
                            dataf = pd.concat([dataf, df])
    
    # 경제일반/정책
    for i in range(1,((page-1)*10)+1,10):
            
            url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=%EA%B2%BD%EC%A0%9C%EB%89%B4%EC%8A%A4&sort=1&photo=0&field=0&pd=0&ds=&de=&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:dd,p:all,a:all"+ "&start=" + str(i)
            response = requests.get(url, headers={'User-Agent':'mozilla/5.0 (windows nt 10.0; win64; x64) applewebkit/537.36 (khtml, like gecko) chrome/114.0.0.0 safari/537.36'})
            html = response.text
            soup = BeautifulSoup(html, "html.parser")
            articles = soup.select("div.info_group")
            for article in articles:
                links = article.select("a.info")
                if len(links) >= 2:
                    url = links[1].attrs["href"]
                    response = requests.get(url, headers={'User-Agent':'mozilla/5.0 (windows nt 10.0; win64; x64) applewebkit/537.36 (khtml, like gecko) chrome/114.0.0.0 safari/537.36'})      
                    html = response.text                                                   
                    soup = BeautifulSoup(html, "html.parser")

                    if "article" in response.url:
                            titles = []
                            dates = []
                            contents = []
                            reporters=[]
                            image_urls = []
                            store_names = []
                            labels = []
                            # 기사제목
                            title    = soup.select_one(".media_end_head_headline")
                            if title != None:
                                title = title.text.strip()
                            else:
                                title = "제목를 알수없음."
                            titles.append(title)
                            # 기사 본문 들고오기
                            content  = soup.select_one("#dic_area")
                            if content != None:
                                content  = content.text 
                            else:
                                content = "본문을 알수없음."
                            contents.append(content)
                            # 날짜 들고와서 수정날짜는 수정날짜대로 날짜 본문은 본문대로 들고오기
                            two_date =   soup.select_one("#ct > div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp")
                            if two_date != None:
                                date = two_date.text
                                dates1 = []
                                dates1.append(date)
                                for i in dates1:
                                    if len(i)>=33:
                                        for_date = i.split("\n\n\n")[1].split("정")[1].split("\n")[0]
                                        dates.append(for_date)
                                    else:
                                        for_date = i.split("력")[1].split("\n")[0]
                                        dates.append(for_date)
                            else:
                                dates.append("날짜를 알수없음.")
                            # 기자명 들고오기
                            reporter = soup.select_one("div#contents > div.byline > p.byline_p > span")                   
                            if reporter != None:
                                reporter = reporter.text.strip()
                            else:
                                reporter = "기자를 알수없음."
                            reporters.append(reporter)
                            # 이미지 url 들고오기
                            image = soup.select_one("meta[property='og:image']")
                            if image!= None:
                                image_url = image["content"]                               
                            else:
                                image_urls = "이미지를 알수없음."
                            image_urls.append(image_url)
                            # 신문사명 들고오기
                            store_name = soup.select_one("#contents > div.copyright > div > p")
                            if store_name !=None :
                                store_name = store_name.text.split(".")[0].split("ⓒ ")[1]                                
                            else:
                                store_name = "신문사명을 알수없음."
                            store_names.append(store_name)
                            
                            labels.append("경제일반/정책")
                            df = pd.DataFrame({"title":titles ,"date" :dates, "content":contents, "reporter":reporters, "image_url":image_urls, "store_name":store_names, "label":labels})
                            dataf = pd.concat([dataf, df])
    
    # 산업/재계
    for i in range(1,((page-1)*10)+1,10):
            
            url = "https://search.naver.com/search.naver?where=news&sm=tab_pge&query=%EC%82%B0%EC%97%85%2F%EC%9E%AC%EA%B3%84%20%EB%89%B4%EC%8A%A4&sort=1&photo=0&field=0&pd=0&ds=&de=&mynews=0&office_type=0&office_section_code=0&news_office_checked=&nso=so:dd,p:all,a:all"+ "&start=" + str(i)
            response = requests.get(url, headers={'User-Agent':'mozilla/5.0 (windows nt 10.0; win64; x64) applewebkit/537.36 (khtml, like gecko) chrome/114.0.0.0 safari/537.36'})
            html = response.text
            soup = BeautifulSoup(html, "html.parser")
            articles = soup.select("div.info_group")
            for article in articles:
                links = article.select("a.info")
                if len(links) >= 2:
                    url = links[1].attrs["href"]
                    response = requests.get(url, headers={'User-Agent':'mozilla/5.0 (windows nt 10.0; win64; x64) applewebkit/537.36 (khtml, like gecko) chrome/114.0.0.0 safari/537.36'})      
                    html = response.text                                                   
                    soup = BeautifulSoup(html, "html.parser")

                    if "article" in response.url:
                            titles = []
                            dates = []
                            contents = []
                            reporters=[]
                            image_urls = []
                            store_names = []
                            labels = []
                            # 기사제목
                            title    = soup.select_one(".media_end_head_headline")
                            if title != None:
                                title = title.text.strip()
                            else:
                                title = "제목를 알수없음."
                            titles.append(title)
                            # 기사 본문 들고오기
                            content  = soup.select_one("#dic_area")
                            if content != None:
                                content  = content.text 
                            else:
                                content = "본문을 알수없음."
                            contents.append(content)
                            # 날짜 들고와서 수정날짜는 수정날짜대로 날짜 본문은 본문대로 들고오기
                            two_date =   soup.select_one("#ct > div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp")
                            if two_date != None:
                                date = two_date.text
                                dates1 = []
                                dates1.append(date)
                                for i in dates1:
                                    if len(i)>=33:
                                        for_date = i.split("\n\n\n")[1].split("정")[1].split("\n")[0]
                                        dates.append(for_date)
                                    else:
                                        for_date = i.split("력")[1].split("\n")[0]
                                        dates.append(for_date)
                            else:
                                dates.append("날짜를 알수없음.")
                            # 기자명 들고오기
                            reporter = soup.select_one("div#contents > div.byline > p.byline_p > span")                   
                            if reporter != None:
                                reporter = reporter.text.strip()
                            else:
                                reporter = "기자를 알수없음."
                            reporters.append(reporter)
                            # 이미지 url 들고오기
                            image = soup.select_one("meta[property='og:image']")
                            if image!= None:
                                image_url = image["content"]                               
                            else:
                                image_urls = "이미지를 알수없음."
                            image_urls.append(image_url)
                            # 신문사명 들고오기
                            store_name = soup.select_one("#contents > div.copyright > div > p")
                            if store_name !=None :
                                store_name = store_name.text.split(".")[0].split("ⓒ ")[1]                                
                            else:
                                store_name = "신문사명을 알수없음."
                            store_names.append(store_name)
                            
                            labels.append("산업/재계")
                            df = pd.DataFrame({"title":titles ,"date" :dates, "content":contents, "reporter":reporters, "image_url":image_urls, "store_name":store_names, "label":labels})
                            dataf = pd.concat([dataf, df])
    
    dataf = dataf.reset_index(drop=True)
    return dataf
    