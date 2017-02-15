
##################
## MONGO BASICS ##
##################

****************
* INSTALLATION *
****************

	- Installation is srightforward. Just extract the binaries or install the desired package onto the computer.
	
	$.\bin\mongod.exe --dbpath "d:\set up\mongodb\data"			# To init the data base

	$.\bin\mongod.exe --auth --dbpath "d:\set up\mongodb\data"	# To init the data base with auth mode enabled

	$.\bin\mongod --auth --port 27017 --dbpath /data/db1		# Run the database with Auth enabled, current port and database path 

	$.\bin\mongo.exe 											# To access to the data base (client)

	Data base will be deployed at "mongodb://localhost:27017"


***************
* DATA BASES *
***************

	>show dbs						 # Show all created data bases in mongodb. Data base must have some data already to show up.

	>use mydb						 #used a data base or create a new one

	>db						  		 # Show the current database being used

	>db.dropDatabase()				 # Remove the current database


*********
* USERS *
*********

	>db.auth( <username>, <password> )															# To authenticate into the data base

	>db.createUser({user: "accountUser",pwd: "password",roles: [ "readWrite", "dbAdmin" ]})		# Create a user with roles
	>db.createUser( { user: "reportsUser", pwd: "password", roles: [ ] })						# Creates a user withot roles

	>db.updateUser( "appClient01",{customData : { employeeId : "0x3039" }roles : [{ role : "read", db : "assets"  } ]} ) # Updates user data

	>db.changeUserPassword("accountUser", "SOh3TbYhx8ypJPxmt1oOfL")								# Changes used password

	>db.dropUser("reportUser1")																# Remove the user from database


***************
* COLLECTIONS *
***************

	>show collections								 # Show all collections created in the current db 
	
	>db.movie.insert({"name":"tutorials point"})	 # Creates a new collection called movie and insert a new item (document).

	>db.createCollection("films")					 #Creates a new collection. Not neccesary since it will be created autmatically when insert the first item.
	
*************
* DOCUMENTS *
*************

	 - INSERT

		>db.movie.insert({"name":"tutorials point"})							# Creates a new collection called movie and insert a new item (document).
	
		>db.movie.insert([{"name":"Jurassic Park"},{"name":"Spiderman"}])		# Insert Multiple elements, like an array

	- UPDATE

		>db.mycol.update({'title':'MongoDB Overview'},{$set:{'title':'New MongoDB Tutorial'}}					# Update the element that match with the find #1 and set the new params

		>db.mycol.update({'title':'MongoDB Overview'}, {$set:{'title':'New MongoDB Tutorial'}},{multi:true})	# Update multiple elements. By default only the first match is updated

		>db.mycol.save({_id:ObjectId(),NEW_DATA})																# Replace the entire object with the ID

	- REMOVE

		>db.mycol.remove({'title':'MongoDB Overview'})					# Remove the doucment with the criteria													

	- QUERY

		>db.movie.find()										# It returns all the iem in the collection "movie". 

		>db.movie.find().pretty()								# It returns all the iem in the collection "movie". Uee pretty to get a formatted input.

		>db.movie.find("name":"tutorials point"}).pretty()		# Find an item. For more operations see http://www.tutorialspoint.com/mongodb/mongodb_query_document.htm
	
																	# quality 	{<key>:<value>} 	db.mycol.find({"by":"tutorials point"}).pretty() 	where by = 'tutorials point'
																	# Less Than 	{<key>:{$lt:<value>}} 	db.mycol.find({"likes":{$lt:50}}).pretty() 	where likes < 50
																	# Less Than Equals 	{<key>:{$lte:<value>}} 	db.mycol.find({"likes":{$lte:50}}).pretty() 	where likes <= 50
																	# Greater Than 	{<key>:{$gt:<value>}} 	db.mycol.find({"likes":{$gt:50}}).pretty() 	where likes > 50
																	# Greater Than Equals 	{<key>:{$gte:<value>}} 	db.mycol.find({"likes":{$gte:50}}).pretty() 	where likes >= 50
																	# Not Equals 	{<key>:{$ne:<value>}} 	db.mycol.find({"likes":{$ne:50}}).pretty() 	where likes != 50

																	# AND: db.mycol.find({"by":"tutorials point","title": "MongoDB Overview"}).pretty()
																	# OR:  db.mycol.find({$or:[{key1: value1}, {key2:value2}]}).pretty()
																	# AND OR TOGETHER: db.mycol.find({"likes": {$gt:10}, $or: [{"by": "tutorials point"}, {"title": "MongoDB Overview"}]}).pretty()


		>db.mycol.find({},{"title":1,_id:0}).sort({"title":-1})		# Sort the find values by title key


**********
* BACKUP *
**********

	$.\bin\mongodump.exe --db test													# This will create a dump of the database in json format creating also a "\dump" folder. The database is given in the paramater.
	$.\bin\mongodump.exe --db test --out - | gzip > dump_`date "+%Y-%m-%d"`.gz		# Zip the dump files

	$.\bin\mongorestore -d test ".\dump\test"										# 
	

**************
* MAP-REDUCE *
**************

Accordingly to Wikipedia, MapReduce is a programing model of processing and generating large data sets with a Parallel, Distributed algorithm on a 
cluster. The input data is processed in pararel between the different workers that will generate the data requiered ofr each step of Mapreduce.

    - "Map" step: Each worker node applies the "map()" function to the local data, and writes the output to a temporary storage. A master node ensures 
	that only one copy of redundant input data is processed.
  
    - "Shuffle" step: Worker nodes redistribute data based on the output keys (produced by the "map()" function), such that all data belonging to one 
	key is located on the same worker node.

    - "Reduce" step: Worker nodes now process each group of output data, per key, in parallel.


MongoDB also has the functionality to apply a MapRecuce function to the data. Map Reduce is used to compute and store the data in parallel between nodes (Master and slaves). 
Because the amount of information and the number of sources some data requires this paradigm has been developed to optimize this process. 

- First we need the data that is going to be stored into the Database. The data sets will contain a number of fields (data-scheme) and tuples with the data.

- The MapReduce process is divided into two operations: Map and Reduce.
	
		Map: is a function that returns a <key, value> pair for each tuple of data stored. This will be chosen depending on the requeriments of the data and the needs.
			-> Map function is originally created to change the domain of the data into another, more convenient to the main focus and purpose of the data.
			-> Because the key could not be unique, the Map function will group all the results into a list of values for each key.
			-> the final result is a list of tuples with a <key,value> pair.

		Since the infrastrucutre is originall created to be a networks node base with multiple "workers" and one Master that manage all the process. The system will have
		a list will all the Maps created for each workers. For this reason the Reduce function takes place.

		(Each Key is located in the same worker node. This step is called (shuffle steps))

		Reduce: is another function, that takes each of the <key, list<values>> and generate another value for eeach key. The list of values for each key will depend on the 
		distributed systems and the number of nodes. 

- THe idea behind this algorithm isthat  we could have different mappings operatons independent from each other. So we could access to the information that we need in
short period of time.


-the Logical View of Map convert one data pair domain into another data pair domain.  Map(k1,v1) ? list(k2,v2)  -> See it returns a list of key par values


EXAMPLE:
	
	INPUTS

		- We have a document that have words in it. e.g.
			Line 234 "the cat is on the table"
			Line 235 "because the bed is broken"

			# In this case the document are ordered using the following domain.
			Domain -> Key: number of line
					  Value: phrase with worids

	MAP

		- For each worker the input will be distributed in parallel to be processed.

		- The pseudocode of the Map function, and for this example, is the following:

			def map(key, Value)
				For each word in input
					# where 1 is the word counter to be used later by the Reduce function... is the simples use case
					emit (word, 1)						

		- This function basically will transform the input domain <key,value> (line, phrase) into another a different domain <word, count>

		- Map function will generates the following output in the first iteration:

			[(the,1), (cat,1), (is,1), (on,1), (the,1), (table,1), (because,1), (the,1), (bed,1), (is,1), (broken,1)]

		- Each <key, pair> will be grouped into a list.
			[(the, [1,1]),
			(cat, [1]),
			(is, [1,1]),
			(on, [1]),
			...
			(broken, [1]
			
			)]
	REDUCE

		- Finally Reduce function will take the outputs from all the workers and will execute the Reduce function. This function will generate a <key, value> pair element.

			def reduce (key, list <values>)
				set count = 0
				for each item in list <values>
					count += item

				emit (key, count)

		- The output from this function will be:
			( (the,2), (cat, 1), (is,2), (on, 1), .. , (broken, 1))
		


	