import uuid
from bidi.algorithm import get_display
import arabic_reshaper
import pymysql.cursors
import datetime
import pymongo


class AppPipeline(object):

    def __init__(self):
        self.db = pymysql.connect(host="localhost",  # your host
                                  user="root",  # username
                                  passwd="root",  # password
                                  db="presscrawler",  # name of the database
                                  charset='utf8mb4',
                                  cursorclass=pymysql.cursors.DictCursor)
        cursor = self.db.cursor()
        self.db.set_charset("utf8")
        cursor.execute('SET NAMES utf8;')
        cursor.execute('SET CHARACTER SET utf8;')
        cursor.execute('SET character_set_connection=utf8;')
        cursor.close()
        self.db.commit()

    def process_item(self, item, spider):
        i = dict(item)
        cursor = self.db.cursor()
        cursor.execute("SELECT id FROM article WHERE link=%s", (i["link"]))
        results = cursor.rowcount
        if results == 0:
            ida = uuid.uuid4().__str__()
            title = get_display(arabic_reshaper.reshape(u'' + i["title"])).replace("'", "")
            author = get_display(arabic_reshaper.reshape(u'' + i["author"])).replace("'", "")
            link = i["link"]
            photo = i["photo"]
            journal = i["journal"]
            publication = datetime.datetime.now()
            try:
                cursor.execute(
                    query="INSERT INTO article(id, journal, title, author, link, photo, publication) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                    args=(ida, journal, title, author, link, photo, publication))
            except Exception as e:
                print(e)
            finally:
                self.db.commit()
                cursor.close()
        cursor.close()

    def spider_closed(self, spider):
        self.db.close()


class AppPipelineNoSQL(object):

    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client["kolchipress"]
        self.collection = self.db["articles"]

    def process_item(self, item, spider):
        item = dict(item)
        result = self.collection.find({"link": item["link"]}).count()
        if result == 0:
            row = {
                "_id": uuid.uuid4().__str__(),
                "title": get_display(arabic_reshaper.reshape(u'' + item["title"])).replace("'", ""),
                "author": get_display(arabic_reshaper.reshape(u'' + item["author"])).replace("'", ""),
                "link": item["link"],
                "photo": item["photo"],
                "journal": item["journal"],
                "publication": datetime.datetime.now()
            }
            print(row)
            inserted = self.collection.insert_one(row)
            print(inserted.inserted_id)
            print("------")
