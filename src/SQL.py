import psycopg2
import torch
import io
import pickle

# try to establish connection
credentials = {"dbname": "RL", "host": "192.168.30.66", "user": "dumpresults", "password": "unsecure"}
conn = psycopg2.connect(**credentials)
conn.close()


def insert(columnData, trainBlob):
    # merge into single dictionary
    data = {**columnData, "trainBlob": trainBlob}

    # establish connection
    conn = psycopg2.connect(**credentials)
    db = conn.cursor()
    insertStatement = "insert into Agents (stateDefinition, actionSet, rewardFunction, acceptance, targetDiameter, maxStepsPerEpisode, successBounty, failurePenalty, device, BATCH_SIZE, GAMMA, TARGET_UPDATE, EPS_START, EPS_END, EPS_DECAY, MEMORY_SIZE, algorithm, network, optimizer, stepSize, trainEpisodes, trainBlob)" \
                      " values (%(stateDefinition)s, %(actionSet)s, %(rewardFunction)s, %(acceptance)s, %(targetDiameter)s, %(maxStepsPerEpisode)s, %(successBounty)s, %(failurePenalty)s, %(device)s, %(BATCH_SIZE)s, %(GAMMA)s, %(TARGET_UPDATE)s, %(EPS_START)s, %(EPS_END)s, %(EPS_DECAY)s, %(MEMORY_SIZE)s, %(algorithm)s, %(network)s, %(optimizer)s, %(stepSize)s, %(trainEpisodes)s, %(trainBlob)s)" \
                      "returning id;"
    db.execute(insertStatement, data)
    conn.commit()
    print("inserted into Agents with id {}".format(db.fetchone()[0]))


def retrieve(row_id):
    # establish connection
    conn = psycopg2.connect(**credentials)
    db = conn.cursor()

    queryStatement = "select * from Agents where id = %s;"
    db.execute(queryStatement, (row_id,))
    data = db.fetchone()

    buffer = io.BytesIO(data[-1])  # assumes train blob is located in last column
    return torch.load(buffer)


def insertBenchmark(agents_id, algorithm, bench_episodes, benchBlob):
    # merge into single dictionary
    data = {"agents_id": agents_id, "algorithm": algorithm, "bench_episodes": bench_episodes, "benchBlob": benchBlob}

    # establish connection
    conn = psycopg2.connect(**credentials)
    db = conn.cursor()
    insertStatement = "insert into Benchmarks (agents_id, algorithm, bench_episodes, benchBlob) values (%(agents_id)s, %(algorithm)s, %(bench_episodes)s, %(benchBlob)s)" \
                      "returning bench_id;"
    db.execute(insertStatement, data)
    conn.commit()
    print("inserted into Benchmarks with id {}".format(db.fetchone()[0]))


def retrieveBenchmark(row_id):
    # establish connection
    conn = psycopg2.connect(**credentials)
    db = conn.cursor()

    queryStatement = "select * from Benchmarks where bench_id = %s;"
    db.execute(queryStatement, (row_id,))
    data = db.fetchone()

    buffer = io.BytesIO(data[-1])  # assumes train blob is located in last column
    return pickle.load(buffer)


if __name__ == "__main__":
    # retrieve a result and unpickle it
    data = retrieve(1)
