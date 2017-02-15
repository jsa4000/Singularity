import pymongo
from pymongo import MongoClient as mongodb
import datetime

# The ideas with this class is to familiarizate with db data bases and operations:
#
# 0. Create new data base
# 1. Create new collection
# 2. Add/Update/Remove item from collection
# 3. Querys (optimzed for Non-SQL)

# See for further documentation
# http://www.tutorialspoint.com/mongodb/mongodb_create_database.htm

# Good Tutorial about MongoDB using python 
# http://api.mongodb.com/python/current/tutorial.html

#Connect to an exisisting data base
#mongodb://localhost:27017
client = mongodb("mongodb://localhost:27017")

# Get all the databases in the current Client
dbs = client.database_names()
#db = client.get_default_database() # Error with no database defined
#print (db)

print (dbs)
# -> [u'local', u'test', u'test2']


# Get the current data base.
db = client.test
db = client["test"]

#Create new collection
articles = db.Articles
post = {"name":"The Title", "description":"This is something", "date":datetime.datetime.utcnow()}
document = articles.insert_one(post)
id_document = document.inserted_id

#List all the collection in the data base
print (db.collection_names(include_system_collections=False))

post = {"name":"The Title2", "description":"This is something2", "date":datetime.datetime.utcnow()}
document = articles.insert_one(post)

print (articles.find_one({"name":"The Title2"}))
print (articles.find_one({"name":"error"})) # No result found

print (id_document)
articles.find_one({"_id": id_document})


# Insert multiple documents into a collection
new_articles = [{"author": "Mike",
              "text": "Another post! But not the first one",
              "tags": ["bulk", "insert"],
              "date": datetime.datetime(2009, 11, 12, 11, 14)},
             {"author": "Eliot",
              "title": "MongoDB is fun, but not the first one",
              "text": "and pretty easy too!",
              "date": datetime.datetime(2009, 11, 10, 10, 45)}]
result = articles.insert_many(new_articles)
print (result)

print (articles.find_one({"author":"Eliot"}))

# Query in where the response will result multiple number
print (articles.find({"author": "Mike"}).count())


print (articles.find().count())

# The result obtaind from the query is yield into a generator so it must be obtained this way.
for article in articles.find():
    print (article)

# The result obtaind from the query is yield into a generator so it must be obtained this way.
for article in articles.find({"author": "Mike"}):
    print (article)

# Update 

print (articles.update_one({"author":"Eliot"}, {"$set": {"text": "Another post! But not the first one and not the second one"}}))
# The result obtaind from the query is yield into a generator so it must be obtained this way.
for article in articles.find({"author": "Eliot"}):
    print (article)


#For multiple update is also possible to specify if the docuemnts don't exist then insert them (upsert = true)
# update_many(filter, update, upsert=False, bypass_document_validation=False) -> HEARDER of the function

print (articles.update_many({"author":"Mike"}, {"$set": {"text": "Multiple Update"}}))
# The result obtaind from the query is yield into a generator so it must be obtained this way.
for article in articles.find({"author": "Mike"}):
    print (article)

# Delete

result = articles.delete_one({"author": "Javier"})
print (result.deleted_count)
print (articles.find({"author": "Javier"}).count())
for article in articles.find({"author": "Javier"}):
    print (article)

# Delete multiple documents
result = articles.delete_many({"author": "Javier"})
print (result.deleted_count)
print (articles.find({"author": "Javier"}).count())
for article in articles.find({"author": "Javier"}):
    print (article)

# Indexing

# In order to create the indexes first the index must be created.
result = articles.create_index([('article_id', pymongo.ASCENDING)], unique=True)
print (result)

print (list(articles.index_information()))

# MongoDB also use an id to indentify each document, but these indexes prevent to the database to not insert documents that are already created.

# In order to inserte a new article we must provide the index create with the .
new_articles = [{"user_id": 101,
              "author": "Javier",
              "text": "Another post! But not the first one",
              "tags": ["bulk", "insert"],
              "date": datetime.datetime(2009, 11, 12, 11, 14)},
             {"user_id": 102,
              "author": "Pedro",
              "title": "The last of US",
              "text": "and pretty easy too!",
              "date": datetime.datetime(2009, 11, 10, 10, 45)}]
result = articles.insert_many(new_articles)
print (result)
print (articles.find().count())

# Remove indexes


# Remove the index specified
articles.dropIndex( { "article_id": 1 } )


#Remove all the idnexes
articles.dropIndexes()

# Map Reduce, Authorization, etc..