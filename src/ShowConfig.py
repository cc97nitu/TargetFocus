import SQL

def showConfig(row_ids: list):
    for id in row_ids:
        rowData = SQL.retrieve(id)

        # print config
        print("Row: {}".format(id))

        print(rowData["environmentConfig"])
        print(rowData["hyperParameters"])
        print("\n")


def dictCompare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o : (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
    same = set(o for o in intersect_keys if d1[o] == d2[o])
    return added, removed, modified, same


def compareRowConfig(row_ids: list):
    # get rows
    rows: list = list()
    for row_id in row_ids:
        rows.append(SQL.retrieve(row_id))

    # find common keys
    hypParamKeyIntersection: set = set(rows[0]["hyperParameters"])
    envConfigKeyIntersection: set = set(rows[0]["environmentConfig"])

    for row in rows[1:]:
        hyperParamSet = set(row["hyperParameters"])
        hypParamKeyIntersection.intersection_update(hyperParamSet)

        envConfigSet = set(row["environmentConfig"])
        envConfigKeyIntersection.intersection_update(envConfigSet)

    # find non-common keys
    hypParamKeyNonCommon: set = set()
    envConfigKeyNonCommon: set = set()

    for row in rows:
        hyperParamSet = set(row["hyperParameters"])
        hypParamKeyNonCommon.update(hyperParamSet.difference(hypParamKeyIntersection))

        envConfigSet = set(row["environmentConfig"])
        envConfigKeyNonCommon.update(envConfigSet.difference(envConfigKeyIntersection))

    # find common keys with non-common values
    hypParamNonCommon: set = set()
    envConfigNonCommon: set = set()

    for key in hypParamKeyIntersection:
        for i in range(1, len(rows)):
            if rows[i - 1]["hyperParameters"][key] != rows[i]["hyperParameters"][key]:
                hypParamNonCommon.add(key)

    for key in envConfigKeyIntersection:
        for i in range(1, len(rows)):
            if rows[i - 1]["environmentConfig"][key] != rows[i]["environmentConfig"][key]:
                envConfigNonCommon.add(key)

    # condense
    nonCommonKeys: set = hypParamKeyNonCommon.union(envConfigKeyNonCommon)
    nonCommonValues: set = hypParamNonCommon.union(envConfigNonCommon)

    return nonCommonKeys, nonCommonValues

if __name__ == "__main__":
    # # show config for rows
    # rows = [131, 170, 171, 172, 174]
    # showConfig(rows)

    # compare multiple rows
    rows = [173, 153, 156, 151, 154, 152, 175]

    print(compareRowConfig(rows))