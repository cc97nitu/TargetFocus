import psycopg2
import datetime
import torch
import io

# try to establish connection
conn = psycopg2.connect(dbname="RL", host="localhost", user="dumpresults", password="unsecure")
conn.close()


def insert(columnData, trainBlob):
    # merge into single dictionary
    data = {**columnData, "trainBlob": trainBlob}

    # establish connection
    conn = psycopg2.connect(dbname="RL", host="localhost", user="dumpresults", password="unsecure")
    db = conn.cursor()
    insertStatement = "insert into Agents (stateDefinition, actionSet) values (%(stateDefinition)s, %(actionSet)s);"
    db.execute(insertStatement, columnData)


if __name__ == "__main__":
    # fetch pre-trained agents
    trainResults = torch.load(
        "/home/conrad/RL/TempDiff/TargetFocus/src/dump/REINFORCE/6d-states-normalized/ConstantRewardPerStep/6d-norm_9A_RR_Cat3_constantRewardPerStep_2000_agents.tar")

    columnData = trainResults["environmentConfig"]
    insert(columnData, trainResults)
