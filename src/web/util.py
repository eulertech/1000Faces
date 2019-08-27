import sys
import os
import os.path
import psycopg2
import json
def get_connection():
    try:
        db_conn = psycopg2.connect(host="thsndfaces.c9dfyqjobtqf.us-east-1.rds.amazonaws.com",
                                   port=5432,
                                   database="thsndfaces",
                                   user="postgres",
                                   password="1000faces")
    except:
        print ("Connection Error.")
        sys.exit(1)

def get_employee ( emp_id ):
   db_conn = get_connection()
   cursor = db_conn.cursor()
   emp_id_str = [str(i) for i in emp_id]
   sqlString = r"select * from public.employee where id in ({}) ".format(','.join(emp_id_str))
   cursor.execute(sqlString)
   rows = cursor.fetchall()
   columns = ('firstname', 'degree', 'pastprojectsid', 'skills', 'lastname', 'managerid', 'profilepicturename', 'yearsincompany',
              'hobbies', 'id','hub')
   data = []
   for row in rows:
       data.append(dict(zip(columns, row)))
   return json.dumps(data)

def get_project (pid):
   db_conn = get_connection()
   cursor = db_conn.cursor()
   p_id_str = [str(i) for i in pid]
   cursor.execute (
       "select * from public.projects where pid in ({}) ").format(','.join(p_id_str))
   rows = cursor.fetchall()
   columns = ('firstname', 'degree', 'pastprojectsid', 'skills', 'lastname', 'managerid', 'profilepicturename', 'yearsincompany',
              'hobbies', 'id','hub')
   data = []
   for row in rows:
       data.append(dict(zip(columns, row)))
   return json.dumps(data)

def get_node (eid ):
   db_conn = get_connection()
   cursor = db_conn.cursor()
   emp_id_str = [str(i) for i in eid]
   sqlString = r"select 'hub' as type, 'Tech' as name, -1 id, 'images/hub/tech.png' as url union " \
               r"select 'hub' as type, 'Business' as name, -2 id, 'images/hub/business.png' as url union " \
               r"select 'hub' as type, 'QA' as name, -3 id, 'images/hub/qa.png' as url union " \
               r"select 'hub' as type, 'ML' as name, -4 id, 'images/hub/ml.png' as url union " \
               r"select 'person' as type, firstname as name,id, '/imgs/' ||  profilepicturename || '.jfif' as url from public.employee where id in ({}) ".format(','.join(emp_id_str))
   cursor.execute(sqlString)
   rows = cursor.fetchall()
   columns = (
   'type', 'name', 'id', 'url')
   data = []
   for row in rows:
       data.append(dict(zip(columns, row)))
   columns = ('source', 'target')
   sqlString = r"select hub as source, firstname as target from public.employee where id in ({}) ".format(
       ','.join(emp_id_str))
   cursor.execute(sqlString)
   rows = cursor.fetchall()
   data1 = []
   for row in rows:
       data1.append(dict(zip(columns, row)))
   return json.dumps(data) +',' +  json.dumps(data1)

if __name__ == "__main__":
   try:
       db_conn = psycopg2.connect(host="thsndfaces.c9dfyqjobtqf.us-east-1.rds.amazonaws.com",
                                  port=5432,
                                  database="thsndfaces",
                                  user="postgres",
                                  password="1000faces")
   except:
       print("Connection Error.")
       sys.exit(1)
   # get_employee([12126, 12127], db_conn
   get_node([12126, 12127], db_conn)
   # get_project([1, 2], db_conn)
