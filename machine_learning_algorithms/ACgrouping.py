# function for grouping overall labels into similaur cases then grouping into main ones

import pandas as pd
import numpy as np
import copy

def AC_clustergroup(ks,cuttoff):
    # strips the important classification data and put it into a large string for the computer
    maindic = {}
    totalsims = 0
    for values in ks:

        x = values[0].strip("_H").split("_H")
        tot = ""
        for s in x:
            # strips the empty classifications off the label
            if not s.isdigit():
                s = s.strip("1234567890\n\t")
                tot = tot + s
        # add a big dictyionary of the labels
        maindic.update({tot: values[1]})
        totalsims += values[1]

    # check for labels that are too small and useless for our needs
    list = []
    deletelist = []
    maxlist = []
    for values in maindic:
        n = len(values)
        # delete any labels where only the 3rd label
        if not (n == 3 * 2 or n == 2 * 2 or n == 2 or n == 2 * 4):
            list.append(values)
        else:

            deletelist.append(values)

        if n == 9 * 2:
            maxlist.append(values)

    # seletes all the small labels from the main dic
    for values in deletelist:
        maindic.pop(values)

    # makes a list of the keys in the main dic
    listtot = []
    for keys in maindic:
        listtot.append(keys)

    # creates an empty database to store all the information in the database
    maind_df = pd.DataFrame(data=np.zeros((len(listtot), len(listtot))), index=list, columns=list)

    bigdic = {}
    N = 0
    for maxs in maxlist:
        list = []
        # adds the max labels into the database
        maind_df.loc[maxs][maxs] = 1

        # appends subset labels into database row
        for values in maindic:
            if maxs.startswith(values) and maxs != values:
                maind_df.loc[maxs][values] = 1
                list.append(values)

        bigdic.update({maxs: list})

    # adds similaur labels to the matrix for the MAX labels
    similaurdic = {}
    for values in bigdic:
        Nt = len(values)
        for comp in bigdic:
            if comp != values:
                ii = 2
                j = 2
                for s in comp[2:]:  # skips the DC
                    # bug here for when the group labels change from (a_) to (b_)
                    if values[ii] == s:
                        j += 1
                    ii += 1
                # if DC is different breaks the groups up definatly
                # This is done for steady state and surface confined which have specific when it won't work
                if comp[:2] != values[:2]:
                    j = 0


                # Can do two difference but looks like it may group things a little twoo much
                if  Nt - j == 2 or Nt - j == 1 or Nt - j == 0:

                    listold = similaurdic.get(values)
                    if type(listold) == type(None):
                        listold = [values, comp]
                    else:
                        listold.append(comp)
                    similaurdic.update({values: listold})
                    maind_df.loc[values][comp] = 1  # puts main value as sam in row
                    # maind_df.loc[comp][values] = 1
                    # puts similaur ones into main dic
                    for val in bigdic.get(comp):
                        maind_df.loc[values][val] = 1  # puts subsets of main value as same row
                        # maind_df.loc[val][values] = 1

    x = pd.DataFrame(np.dot(maind_df.values, pd.DataFrame.from_dict(maindic, orient="index").values), index=listtot)
    # deletes duplicate matrixes based of sum of the simulations
    j = 0
    for values in listtot:

        l1 = maind_df.loc[values][:].values.tolist()
        # n = int(x.loc[values][0])
        for values2 in listtot[j:]:
            l2 = maind_df.loc[values2][:].values.tolist()
            if l2 == l1 and values != values2 and sum(l1) != 0 and sum(l2) != 0:
                if values[2:] != values2[2:]:  # exemption for dc different
                    maind_df[values2][:] = 0
        j += 1

    # Deals with union positions and moves larger stuff to smaller
    dictionaryvalues = pd.DataFrame.from_dict(maindic, orient="index")

    # sorts the main classes by largest to smallest
    blah = np.dot(maind_df.values, pd.DataFrame.from_dict(maindic, orient="index").values).tolist()
    valuedic = {}
    i = 0
    for values in maindic:
        valuedic.update({values: blah[i][0]})
        i += 1

    # sorts the main classes by largest to smallest
    x1 = {k1: v for k1, v in sorted(valuedic.items(), key=lambda item: item[1], reverse=True)}
    dictionaryvalues = dictionaryvalues.values

    # put label columns into largest group possible
    for i, v in x1.items():

        ni = maind_df.loc[:][i].values
        ln = sum(ni)
        x = pd.DataFrame(np.dot(maind_df.values, dictionaryvalues), index=listtot)
        if ln != 1 and ln != 0:
            listg = []
            for j in range(len(listtot)):
                if maind_df.loc[listtot[j]][i] == 1:
                    listg.append(listtot[j])

            listn = []
            for values in listg:
                listn.append(x.loc[values][0])

            n = listn[0]
            nk = 0

            for j in range(len(listg)):
                if listn[j] > n:
                    nk = j
                    n = listn[j]

            for j in listg:
                if j != listg[nk]:  # and j != i:
                    maind_df.loc[j][i] = 0

        else:
            pass

    ###################################################################
    ### puts lonely labels into the nearest allocation within reason###
    ###################################################################

    # sorts the main classes by largest to smallest
    blah = np.dot(maind_df.values, pd.DataFrame.from_dict(maindic, orient="index").values).tolist()
    valuedic = {}
    i = 0
    for values in maindic:
        valuedic.update({values: blah[i][0]})
        i += 1

    # sorts the main classes by largest to smallest
    x1 = {k1: v for k1, v in sorted(valuedic.items(), key=lambda item: item[1], reverse=True)}

    ###################################################################

    # sums main dictionary to see if label is lonely
    sumv = copy.deepcopy(maind_df.sum(axis=0))
    # puts sums into a dictionary
    sumdic = {}
    for values in maindic:
        sumdic.update({values: sumv.loc[values]})

    simdiccoef = {}
    for values, items in sumdic.items():
        if items == 0:
            Nt = len(values)
            for comp in bigdic:  # itterates through large labels
                if comp != values:
                    ii = 2
                    j = 2
                    for s in values[2:]:  # skips the DC
                        if comp[ii] == s:
                            j += 1
                        ii += 1

                    simdiccoef.update({comp: Nt - j})

            # TRY AND PUT IT INTO SOMETHING WITH A SIMILAURITY OF 1
            allocated = False
            for values2 in x1:  # iterate through largest groups first
                if simdiccoef.get(values2) == 1:
                    for items3 in x1:
                        if maind_df.loc[values2][items3] == 1:
                            maind_df.loc[values2][values] = 1
                            allocated = True
                            break
                if allocated:
                    break

            # TRY AND PUT IT INTO SOMETHING WITH A SIMILAURITY OF 1
            if not allocated:
                for values2 in x1:  # iterate through largest groups first
                    if simdiccoef.get(values2) == 2:
                        for items3 in x1:
                            if maind_df.loc[values2][items3] == 1:
                                maind_df.loc[values2][values] = 1
                                allocated = True
                                break
                    if allocated:
                        break

    # allocate the dictionary with a list of similaur ones
    x = pd.DataFrame(np.dot(maind_df.values, dictionaryvalues), index=listtot)
    n = 0
    dicoverall = {}
    for i in listtot:
        if x.loc[i][0] != 0:
            lister = [i]  # adds the main one
            for j in listtot:
                if maind_df.loc[i][j] == 1 and i != j:
                    lister.append(j)
                    n += 1
            dicoverall.update({i: lister})

    # fixes an issue where large groups would disapear randomly due to it being sorted into larger singular group
    xvals = copy.deepcopy(dicoverall)
    for i, v in xvals.items():
        if len(v) == 1:
            dicoverall.pop(i)

    # count the number in each group
    ntot = 0
    n_classes = 0
    nh = 0
    previous = []
    print("counts groups")
    dicgroupedfinal = {}
    for values, item in dicoverall.items():
        nx = 0
        for stuff in item:
            previous.append(stuff)
            nx += maindic.get(stuff)

        ntot += nx
        if nx > cuttoff:
            nh += nx
            n_classes += 1
            print(values + " : " + str(nx))
            dicgroupedfinal.update({values:item})

    return dicgroupedfinal