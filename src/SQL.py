import psycopg2
import datetime
import torch
import io

# try to establish connection
conn = psycopg2.connect(dbname="RL", host="localhost", user="dumpresults", password="unsecure")
conn.close()


def insert(columnData, blob):
    # establish connection
    conn = psycopg2.connect(dbname="RL", host="localhost", user="dumpresults", password="unsecure")
    db = conn.cursor()


db = conn.cursor()
db.execute("insert into test (bullshit) values (%s)", ("foo",))
conn.commit()
